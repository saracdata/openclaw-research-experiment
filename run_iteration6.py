"""Iteration #6 — QuantStart advanced frontiers: Signature-based ML (Rough Paths),
Rough Volatility (fBM/RFSV), Market Microstructure (LOB), Systematic TAA enhancements.
Outputs: iter6_*.csv, iter6_*.png, updated REPORT.md
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/root/quant')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import load, backtest, sharpe, max_dd, perf
from stats import nw_tstat, report_validation
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

COST = 10
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'AGG',
           'VNQ', 'DBC', 'XLP', 'XLU', 'XLE', 'XLF', 'IEI', 'VIG', 'SCHD',
           'MDY', 'XLK', 'XLV', 'VTI', 'VEA', 'VWO', 'GOVT', 'SHY', 'BIL',
           'LQD', 'HYG', 'SMH']
px = pd.DataFrame({t: load(t)['Close'] for t in TICKERS}).dropna()
rets = px.pct_change().dropna()

# ======================================================================
# Base signals (from previous iterations)
# ======================================================================
def rebalanced_portfolio(weights_dict):
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    total = sum(weights_dict.values())
    norm_w = {k: v/total for k, v in weights_dict.items()}
    months = px.resample('ME').last().index
    for d in months:
        if d in px.index:
            idx = px.index.get_loc(d)
            if idx + 1 < len(px.index):
                for k, v in norm_w.items():
                    if k in w.columns:
                        w.loc[px.index[idx+1], k] = v
    w = w.ffill().fillna(0)
    return w

# SMA200
w_sma = (px[['SPY']] > px[['SPY']].rolling(200).mean()).astype(float).reindex(columns=px.columns).fillna(0)

# GEM
mom = px[['SPY','EFA','AGG']] / px[['SPY','EFA','AGG']].shift(126) - 1
w_gem = pd.DataFrame(0.0, index=px.index, columns=px.columns)
valid = mom.dropna().index
for t in valid:
    if mom.loc[t, 'SPY'] > 0:
        w_gem.loc[t, mom.loc[t, ['SPY','EFA']].idxmax()] = 1.0
    else:
        w_gem.loc[t, 'AGG'] = 1.0

# TSMOM + Risk Parity
mom12 = px / px.shift(252 + 21) - 1
signs = (mom12 > 0).astype(float) * 2 - 1
vol = rets.rolling(63).std() * np.sqrt(252)
inv_vol = 1.0 / vol.replace(0, np.nan)
w_rp = signs.mul(inv_vol).div(inv_vol.abs().sum(axis=1), axis=0).fillna(0)

# XSec Momentum (top 5)
RISK = ['SPY','QQQ','IWM','EFA','EEM','VNQ','DBC','XLK','XLF','XLE','XLV','MDY','SMH','VIG']
RISK = [t for t in RISK if t in px.columns]
mom_xs = px[RISK] / px[RISK].shift(252) - 1
ranks = mom_xs.rank(axis=1, ascending=False)
w_xs = (ranks <= 5).astype(float)
w_xs = w_xs.div(w_xs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).reindex(columns=px.columns).fillna(0)

# 60/40
w_6040 = rebalanced_portfolio({'SPY': 0.6, 'AGG': 0.4})

# All Weather (Dalio simplified)
w_aw = rebalanced_portfolio({'SPY': 0.3, 'TLT': 0.4, 'IEF': 0.15, 'GLD': 0.075, 'DBC': 0.075})

# ======================================================================
# E1. Signature-based features for regime classification (Rough Paths Part 2-3)
# ======================================================================
def lead_lag_transform(path):
    """Lead-lag transformation for signature computation (from QuantStart Part 1)."""
    # path is 1D array of prices/returns
    # Returns 2D path: (time, value) with lead-lag
    t = np.arange(len(path))
    # Lead-lag: duplicate each point except first/last
    ll_path = np.zeros((2*len(path)-2, 2))
    for i in range(len(path)-1):
        ll_path[2*i] = [path[i], path[i+1]]
        ll_path[2*i+1] = [path[i+1], path[i+1]]
    return ll_path

def signature(path, order=3):
    """
    Compute truncated signature up to given order.
    For 1D path, signature is just iterated integrals.
    Simplified: for piecewise linear path, signature terms are:
    - Level 1: total increment
    - Level 2: 1/2 * (increment)^2 (for 1D)
    - Level 3: 1/6 * (increment)^3
    """
    # For multidimensional, need proper implementation
    # Here we use a practical approximation: signature of returns window
    increments = np.diff(path)
    sig = [1.0]  # level 0
    for k in range(1, order+1):
        # k-th level signature (simplified for 1D)
        sig.append(np.mean(increments**k))
    return np.array(sig)

def rolling_signatures(returns, window=63, order=3):
    """Compute rolling signature features from returns."""
    n = len(returns)
    sig_features = []
    dates = []
    for i in range(window, n):
        window_rets = returns.iloc[i-window:i].values
        # Compute signature for each asset, concatenate
        asset_sigs = []
        for j in range(window_rets.shape[1]):
            s = signature(window_rets[:, j], order=order)
            asset_sigs.extend(s[1:])  # drop level 0 (always 1)
        sig_features.append(asset_sigs)
        dates.append(returns.index[i])
    return pd.DataFrame(sig_features, index=dates)

# Compute signature features for SPY (proxy for market regime)
print("E1 Computing signature features...")
spy_rets = rets[['SPY']].dropna()
sig_df = rolling_signatures(spy_rets, window=63, order=3)
sig_df.columns = [f'sig_{i}' for i in range(sig_df.shape[1])]

# Use signature features to predict next-month regime (high/low vol)
# Target: next 21-day realized vol > median
target_vol = spy_rets.rolling(21).std().shift(-21)
target = (target_vol > target_vol.rolling(252).median()).astype(int).dropna()

# Align
common_idx = sig_df.index.intersection(target.index)
X = sig_df.loc[common_idx]
y = target.loc[common_idx]

# Simple logistic regression with walk-forward
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

train_size = int(len(X) * 0.6)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_s, y_train.squeeze())
pred = lr.predict(X_test_s)
pred_proba = lr.predict_proba(X_test_s)[:, 1]

accuracy = (pred == y_test.squeeze()).mean()
print(f"E1 Signature Regime Accuracy: {accuracy:.2%}")

# Build strategy: use signature regime prediction to adjust SMA200 exposure
w_sig = w_sma.copy()
# When predicted high vol regime, reduce equity exposure
high_vol_mask = pd.Series(pred, index=X_test.index) == 1
for t in high_vol_mask[high_vol_mask].index:
    if t in w_sig.index:
        w_sig.loc[t] *= 0.5  # cut exposure in half

r_sma = backtest(w_sma, rets, COST).dropna()
r_sig = backtest(w_sig, rets, COST).dropna()

e1_tbl = pd.DataFrame({
    'Strategy': ['SMA200 Base', 'SMA200 + Signature Regime'],
    'Sharpe': [round(sharpe(r_sma), 2), round(sharpe(r_sig), 2)],
    'AnnRet%': [round(252*r_sma.mean()*100, 2), round(252*r_sig.mean()*100, 2)],
    'MaxDD%': [round(100*max_dd((1+r_sma).cumprod()), 2), round(100*max_dd((1+r_sig).cumprod()), 2)],
})
e1_tbl.to_csv('iter6_signature_regime.csv', index=False)
print('E1 Signature Regime:\n', e1_tbl.to_string(index=False))

# ======================================================================
# E2. Rough Volatility / fBM modeling (RFSV from "Volatility is Rough")
# ======================================================================
def generate_fbm(n, H, seed=42):
    """Generate fractional Brownian motion using Davies-Harte method (simplified)."""
    rng = np.random.default_rng(seed)
    # Simple approximation: fBM with Hurst H
    # Using Cholesky of covariance matrix (slow but works for small n)
    t = np.arange(1, n+1)
    cov = 0.5 * (t[:, None]**(2*H) + t[None, :]**(2*H) - np.abs(t[:, None] - t[None, :])**(2*H))
    # Add small noise for numerical stability
    cov = cov + 1e-6 * np.eye(n)
    L = np.linalg.cholesky(cov)
    return L @ rng.normal(0, 1, n)

def rfsv_volatility_path(n, H=0.1, nu=0.5, alpha=5.0, m=0.0, dt=1/252, seed=42):
    """
    Rough Fractional Stochastic Volatility (RFSV) model:
    dX_t = nu * dW^H_t - alpha * (X_t - m) * dt
    sigma_t = exp(X_t)
    """
    rng = np.random.default_rng(seed)
    # Euler-Maruyama for rough OU
    X = np.zeros(n)
    X[0] = m
    # Generate fBM increments
    fbm = generate_fbm(n, H, seed)
    dfbm = np.diff(fbm, prepend=0)
    
    for i in range(1, n):
        X[i] = X[i-1] + nu * dfbm[i] - alpha * (X[i-1] - m) * dt
    
    vol = np.exp(X) * 0.16  # scale to ~16% annual vol
    return vol

# Estimate Hurst exponent from realized volatility (R/S analysis)
def hurst_exponent(series, max_lag=100):
    """Estimate Hurst exponent using R/S analysis."""
    lags = range(2, min(max_lag, len(series)//4))
    tau = []
    for lag in lags:
        diffs = series[lag:] - series[:-lag]
        rescaled_range = np.max(np.cumsum(diffs - diffs.mean())) - np.min(np.cumsum(diffs - diffs.mean()))
        std_d = diffs.std()
        if std_d > 0:
            tau.append(rescaled_range / std_d)
        else:
            tau.append(0)
    tau = np.array(tau)
    valid = (tau > 0) & np.isfinite(tau)
    if valid.sum() > 5:
        coeffs = np.polyfit(np.log(np.array(lags)[valid]), np.log(tau[valid]), 1)
        return coeffs[0]
    return 0.5

# Compute realized vol and estimate H
realized_vol = rets['SPY'].rolling(21).std() * np.sqrt(252)
H_est = hurst_exponent(realized_vol.dropna().values)
print(f"E2 Estimated Hurst exponent from SPY realized vol: {H_est:.3f}")

# Simulate RFSV paths and test strategy robustness
n_paths = 50
n_days = len(rets)
base_strats = {'SMA200': w_sma, 'GEM': w_gem, 'TSMOM+RP': w_rp, 'XSec Mom': w_xs}

rfsv_results = {}
for name, w in base_strats.items():
    sharpes = []
    for p in range(n_paths):
        # Generate synthetic returns with RFSV vol
        vol_path = rfsv_volatility_path(n_days, H=0.1, seed=p*100)
        # Generate returns with this vol
        syn_rets = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            mu = rets[col].mean()
            sig = rets[col].std()
            syn_rets[col] = np.random.normal(mu, sig * vol_path / 0.16, n_days)
        syn_rets = syn_rets.fillna(0)
        
        r_syn = backtest(w, syn_rets, COST).dropna()
        if len(r_syn) > 100:
            sharpes.append(sharpe(r_syn))
    if sharpes:
        rfsv_results[name] = {'mean': np.mean(sharpes), 'std': np.std(sharpes),
                              'min': np.min(sharpes), 'max': np.max(sharpes),
                              'pct_neg': np.mean(np.array(sharpes) < 0) * 100}

rfsv_tbl = pd.DataFrame(rfsv_results).T
rfsv_tbl.to_csv('iter6_rfsv_stress.csv')
print('E2 RFSV Stress:\n', rfsv_tbl.round(2).to_string())

# ======================================================================
# E3. Market Microstructure: LOB-inspired execution costs
# ======================================================================
def lob_execution_cost(turnover, spread_bps=5, impact_coeff=0.1, adv_dollars=1e9):
    """
    LOB-inspired cost model:
    - Half-spread cost (crossing spread)
    - Square-root market impact: impact_coeff * sqrt(turnover / ADV)
    - Latency slippage
    """
    # Half-spread
    spread_cost = turnover * spread_bps / 1e4 * 0.5
    # Square-root impact (Almgren et al. square-root law)
    participation = turnover / adv_dollars  # Simplified: assume $1M portfolio
    impact_cost = turnover * impact_coeff * np.sqrt(np.maximum(participation, 1e-6))
    # Latency slippage (1bp per 10% turnover)
    latency_cost = turnover * 1e-4 * np.minimum(turnover * 10, 1)
    return spread_cost + impact_cost + latency_cost

# Test on strategies with different turnover
strategies_test = {'SMA200': w_sma, 'GEM': w_gem, 'XSec Mom': w_xs, 'TSMOM+RP': w_rp}

lob_results = {}
for name, w in strategies_test.items():
    pos = w.shift(1).fillna(0)
    turnover = pos.diff().abs().sum(axis=1)
    turnover.iloc[0] = pos.iloc[0].abs().sum()
    
    # Base cost (10bp)
    base_cost = turnover * COST / 1e4
    r_base = (pos * rets).sum(axis=1) - base_cost
    
    # LOB cost
    lob_cost = lob_execution_cost(turnover)
    r_lob = (pos * rets).sum(axis=1) - lob_cost
    
    lob_results[name] = {
        'Base Sharpe': round(sharpe(r_base.dropna()), 2),
        'LOB Sharpe': round(sharpe(r_lob.dropna()), 2),
        'Avg Daily Turnover%': round(turnover.mean() * 100, 2),
    }

lob_tbl = pd.DataFrame(lob_results).T
lob_tbl.to_csv('iter6_lob_costs.csv')
print('E3 LOB Costs:\n', lob_tbl.to_string())

# ======================================================================
# E4. Systematic TAA: Multi-Signal Risk Parity with Correlation Clustering
# ======================================================================
def risk_parity_weights(cov, max_iter=1000, tol=1e-8):
    """Risk parity weights via Newton method."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        rc = w * (cov @ w)  # risk contributions
        target = rc.sum() / n
        grad = cov @ w
        hess = cov + np.outer(w, np.diag(cov))
        step = (rc - target) / np.diag(hess)
        w_new = w - step
        w_new = np.maximum(w_new, 1e-8)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w

def hclust_correlation(corr):
    """Simple hierarchical clustering of correlation matrix."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = squareform(1 - corr)
    link = linkage(dist, method='ward')
    clusters = fcluster(link, t=0.5, criterion='distance')
    return clusters

# Multi-signal risk parity
# Signals as "assets" for meta-allocation
sig_rets = pd.DataFrame({
    'SMA200': backtest(w_sma, rets, COST),
    'GEM': backtest(w_gem, rets, COST),
    'TSMOM+RP': backtest(w_rp, rets, COST),
    'XSec Mom': backtest(w_xs, rets, COST),
    '60/40': backtest(w_6040, rets, COST),
    'All Weather': backtest(w_aw, rets, COST),
}).dropna()

# Rolling risk parity on signal returns
window = 252
w_meta = pd.DataFrame(0.0, index=sig_rets.index, columns=sig_rets.columns)

for i in range(window, len(sig_rets)):
    cov = sig_rets.iloc[i-window:i].cov().values
    w_meta.iloc[i] = risk_parity_weights(cov)

# Also try HRP (hierarchical risk parity) on signals
w_hrp = pd.DataFrame(0.0, index=sig_rets.index, columns=sig_rets.columns)
for i in range(window, len(sig_rets)):
    corr = sig_rets.iloc[i-window:i].corr().values
    try:
        clusters = hclust_correlation(corr)
        # Simple cluster risk parity
        n_clusters = len(np.unique(clusters))
        cluster_w = np.zeros(n_clusters)
        for c in range(1, n_clusters+1):
            mask = clusters == c
            sub_cov = cov[mask][:, mask]
            cluster_w[c-1] = 1.0 / n_clusters  # equal risk across clusters
        # Distribute within clusters
        final_w = np.zeros(len(clusters))
        for c in range(1, n_clusters+1):
            mask = clusters == c
            final_w[mask] = cluster_w[c-1] / mask.sum()
        w_hrp.iloc[i] = final_w
    except:
        w_hrp.iloc[i] = 1.0 / len(sig_rets.columns)

# Backtest meta strategies
r_meta = (w_meta.shift(1).fillna(0) * sig_rets).sum(axis=1)
r_hrp = (w_hrp.shift(1).fillna(0) * sig_rets).sum(axis=1)
r_eq = sig_rets.mean(axis=1)  # equal weight

e4_tbl = pd.DataFrame({
    'Strategy': ['Equal Weight', 'Risk Parity', 'HRP'],
    'Sharpe': [round(sharpe(r_eq), 2), round(sharpe(r_meta.dropna()), 2), round(sharpe(r_hrp.dropna()), 2)],
    'AnnRet%': [round(252*r_eq.mean()*100, 2), round(252*r_meta.dropna().mean()*100, 2), round(252*r_hrp.dropna().mean()*100, 2)],
    'MaxDD%': [round(100*max_dd((1+r_eq).cumprod()), 2), round(100*max_dd((1+r_meta.dropna()).cumprod()), 2), round(100*max_dd((1+r_hrp.dropna()).cumprod()), 2)],
})
e4_tbl.to_csv('iter6_meta_taa.csv', index=False)
print('E4 Meta-TAA:\n', e4_tbl.to_string(index=False))

# ======================================================================
# E5. Rebalance Timing Luck (from TAA article)
# ======================================================================
def test_rebalance_timing(w_func, rets, n_offsets=21):
    """Test strategy performance across different rebalance days."""
    sharpes = []
    for offset in range(n_offsets):
        # Shift weights by offset days
        w_shifted = w_func.shift(offset).ffill().fillna(0)
        r = backtest(w_shifted, rets, COST).dropna()
        if len(r) > 100:
            sharpes.append(sharpe(r))
    return np.array(sharpes)

# Test on SMA200
sma_sharpes = test_rebalance_timing(w_sma, rets)
gem_sharpes = test_rebalance_timing(w_gem, rets)
xs_sharpes = test_rebalance_timing(w_xs, rets)

timing_tbl = pd.DataFrame({
    'Strategy': ['SMA200', 'GEM', 'XSec Mom'],
    'Mean Sharpe': [sma_sharpes.mean(), gem_sharpes.mean(), xs_sharpes.mean()],
    'Std Sharpe': [sma_sharpes.std(), gem_sharpes.std(), xs_sharpes.std()],
    'Min Sharpe': [sma_sharpes.min(), gem_sharpes.min(), xs_sharpes.min()],
    'Max Sharpe': [sma_sharpes.max(), gem_sharpes.max(), xs_sharpes.max()],
    'Range': [sma_sharpes.max()-sma_sharpes.min(), gem_sharpes.max()-gem_sharpes.min(), xs_sharpes.max()-xs_sharpes.min()],
})
timing_tbl.to_csv('iter6_timing_luck.csv', index=False)
print('E5 Timing Luck:\n', timing_tbl.to_string(index=False))

# ======================================================================
# E6. 60/40 vs TAA: Regime-Conditional Benchmarking
# ======================================================================
# Regime: SPY 12m momentum > 0 (bull) vs < 0 (bear)
spy_mom = px['SPY'] / px['SPY'].shift(252) - 1
bull_regime = (spy_mom > 0).astype(int)

# Strategy returns
strat_rets = {
    '60/40': backtest(w_6040, rets, COST),
    'All Weather': backtest(w_aw, rets, COST),
    'SMA200': backtest(w_sma, rets, COST),
    'GEM': backtest(w_gem, rets, COST),
    'XSec Mom': backtest(w_xs, rets, COST),
}

regime_perf = {}
for name, r in strat_rets.items():
    r = r.dropna()
    common = r.index.intersection(bull_regime.index)
    r = r.loc[common]
    regime = bull_regime.loc[common]
    
    bull_r = r[regime == 1]
    bear_r = r[regime == 0]
    
    regime_perf[name] = {
        'Bull Sharpe': round(sharpe(bull_r), 2) if len(bull_r) > 20 else np.nan,
        'Bear Sharpe': round(sharpe(bear_r), 2) if len(bear_r) > 20 else np.nan,
        'Bull AnnRet%': round(252*bull_r.mean()*100, 2) if len(bull_r) > 20 else np.nan,
        'Bear AnnRet%': round(252*bear_r.mean()*100, 2) if len(bear_r) > 20 else np.nan,
    }

regime_tbl = pd.DataFrame(regime_perf).T
regime_tbl.to_csv('iter6_regime_conditional.csv')
print('E6 Regime-Conditional:\n', regime_tbl.to_string())

# ======================================================================
# Compile Iteration #6 results
# ======================================================================
all_strategies = {
    'SMA200': w_sma,
    'GEM': w_gem,
    'TSMOM+RP': w_rp,
    'XSec Mom': w_xs,
    '60/40': w_6040,
    'All Weather': w_aw,
    'SMA200+Sig': w_sig,
    'Meta-RP': r_meta,
    'Meta-HRP': r_hrp,
    'Meta-EW': r_eq,
}

results = []
all_rets = {}
for name, obj in all_strategies.items():
    if isinstance(obj, pd.Series):
        r = obj
    else:
        r = backtest(obj, rets, COST).dropna()
    all_rets[name] = r
    results.append(perf(r, name))

res = pd.DataFrame(results)
res['Calmar'] = res['AnnRet%'] / res['MaxDD%'].abs()

# Walk-forward folds
n = len(px)
fb = [(k * n // 4, (k + 1) * n // 4) for k in range(4)]
wf_rows = []
for name, r in all_rets.items():
    r = r.dropna()
    if len(r) > 100:
        shs = [round(sharpe(r.iloc[a:b]), 2) for a, b in fb]
    else:
        shs = [0, 0, 0, 0]
    wf_rows.append({'Strategy': name, 'F1': shs[0], 'F2': shs[1], 'F3': shs[2], 'F4': shs[3]})
wf = pd.DataFrame(wf_rows)

# Validation
sharpes = [sharpe(r) for r in all_rets.values() if len(r) > 100]
sr_std = float(np.std(sharpes, ddof=1))
vals = [report_validation(results[i]['name'], list(all_rets.values())[i], len(all_strategies), sr_std)
        for i in range(len(results)) if len(list(all_rets.values())[i]) > 100]
val = pd.DataFrame(vals)

# Save
res.to_csv('iter6_comprehensive_perf.csv', index=False)
wf.to_csv('iter6_comprehensive_walkforward.csv', index=False)
val.to_csv('iter6_comprehensive_validation.csv', index=False)

# Plot equity curves
plt.figure(figsize=(12, 7))
for name, r in all_rets.items():
    if len(r) > 100:
        plt.plot((1+r).cumprod(), label=name, alpha=0.8, linewidth=0.8)
plt.legend(fontsize=7, ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Iteration #6 — All Strategies Equity Curves (net 10bp)')
plt.ylabel('Growth of $1'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('iter6_equity.png', dpi=110, bbox_inches='tight')

# Timing luck
plt.figure(figsize=(8, 5))
timing_tbl.set_index('Strategy')['Range'].plot(kind='bar', color='steelblue')
plt.title('Rebalance Timing Luck: Sharpe Range Across 21 Offsets')
plt.ylabel('Max - Min Sharpe'); plt.tight_layout(); plt.savefig('iter6_timing.png', dpi=110)

# RFSV stress
plt.figure(figsize=(10, 6))
rfsv_tbl['mean'].plot(kind='bar', color='coral')
plt.title('Mean Sharpe Under RFSV (H=0.1) Stress')
plt.ylabel('Sharpe'); plt.tight_layout(); plt.savefig('iter6_rfsv.png', dpi=110)

# Signature features importance
plt.figure(figsize=(8, 5))
if len(lr.coef_[0]) == len(X.columns):
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coef': lr.coef_[0]})
    coef_df = coef_df.sort_values('Coef', key=abs, ascending=True).tail(10)
    plt.barh(range(len(coef_df)), coef_df['Coef'].values)
    plt.yticks(range(len(coef_df)), coef_df['Feature'].values)
    plt.title('Signature Feature Coefficients (Regime Prediction)')
    plt.tight_layout(); plt.savefig('iter6_signature_coef.png', dpi=110)

print('\n=== ITERATION #6 PERFORMANCE ===')
print(res.to_string(index=False))
print('\n=== WALK-FORWARD ===')
print(wf.to_string(index=False))
print('\n=== VALIDATION ===')
print(val.to_string(index=False))
print('\nIteration #6 complete.')