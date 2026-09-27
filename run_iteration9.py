"""Iteration #9 — QuantStart: Kelly Criterion, Realized Volatility, SVM Regime, Forex, Advanced Metrics
Explores: Kelly optimal bet sizing, realized vol forecasting, SVM for regime, Forex carry/momentum, Sortino/Calmar/Omega ratios.
Outputs: iter9_*.csv, iter9_*.png, updated REPORT.md
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
from scipy.optimize import minimize, brentq
from scipy.stats import norm, skew, kurtosis
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
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
# Helper: rebalanced portfolio
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

# ======================================================================
# Base strategies
# ======================================================================
w_sma = (px[['SPY']] > px[['SPY']].rolling(200).mean()).astype(float).reindex(columns=px.columns).fillna(0)

mom = px[['SPY','EFA','AGG']] / px[['SPY','EFA','AGG']].shift(126) - 1
w_gem = pd.DataFrame(0.0, index=px.index, columns=px.columns)
valid = mom.dropna().index
for t in valid:
    if mom.loc[t, 'SPY'] > 0:
        w_gem.loc[t, mom.loc[t, ['SPY','EFA']].idxmax()] = 1.0
    else:
        w_gem.loc[t, 'AGG'] = 1.0

RISK = ['SPY','QQQ','IWM','EFA','EEM','VNQ','DBC','XLK','XLF','XLE','XLV','MDY','SMH','VIG']
RISK = [t for t in RISK if t in px.columns]
mom_xs = px[RISK] / px[RISK].shift(252) - 1
ranks = mom_xs.rank(axis=1, ascending=False)
w_xs = (ranks <= 5).astype(float)
w_xs = w_xs.div(w_xs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).reindex(columns=px.columns).fillna(0)

mom12 = px / px.shift(252 + 21) - 1
signs = (mom12 > 0).astype(float) * 2 - 1
vol = rets.rolling(63).std() * np.sqrt(252)
inv_vol = 1.0 / vol.replace(0, np.nan)
w_rp = signs.mul(inv_vol).div(inv_vol.abs().sum(axis=1), axis=0).fillna(0)

# ======================================================================
# E1. Kelly Criterion Optimal Bet Sizing
# ======================================================================
def kelly_fraction(returns, max_leverage=2.0):
    """Kelly fraction: f* = μ / σ² for Gaussian returns."""
    mu = returns.mean() * 252
    sigma2 = returns.var() * 252
    if sigma2 > 0:
        f = mu / sigma2
        return np.clip(f, 0, max_leverage)
    return 1.0

def kelly_fraction_full(returns, max_leverage=2.0):
    """Full Kelly using numerical optimization of E[log(1 + f*r)]."""
    def neg_log_growth(f):
        r = returns.values
        growth = np.log(1 + f * r)
        return -growth.mean()
    
    try:
        res = minimize(neg_log_growth, x0=0.5, bounds=[(0, max_leverage)])
        return res.x[0]
    except:
        return 1.0

# Test Kelly on strategies
strategies = {'SMA200': w_sma, 'GEM': w_gem, 'XSec Mom': w_xs, 'TSMOM+RP': w_rp}

kelly_results = {}
for name, w in strategies.items():
    r_base = backtest(w, rets, COST).dropna()
    
    # Gaussian Kelly
    f_gauss = kelly_fraction(r_base)
    # Full Kelly
    f_full = kelly_fraction_full(r_base)
    
    # Apply Kelly leverage
    w_kelly_g = w.mul(f_gauss).clip(upper=1.0)  # cap at 1x for long-only
    w_kelly_f = w.mul(f_full).clip(upper=1.0)
    
    r_kelly_g = backtest(w_kelly_g, rets, COST).dropna()
    r_kelly_f = backtest(w_kelly_f, rets, COST).dropna()
    
    kelly_results[name] = {
        'Base Sharpe': round(sharpe(r_base), 2),
        'Kelly(Gauss) f': round(f_gauss, 2),
        'Kelly(Full) f': round(f_full, 2),
        'Kelly(Gauss) Sharpe': round(sharpe(r_kelly_g), 2),
        'Kelly(Full) Sharpe': round(sharpe(r_kelly_f), 2),
        'Kelly(Gauss) AnnRet%': round(252*r_kelly_g.mean()*100, 2),
        'Kelly(Full) AnnRet%': round(252*r_kelly_f.mean()*100, 2),
        'Kelly(Gauss) MaxDD%': round(100*max_dd((1+r_kelly_g).cumprod()), 2),
        'Kelly(Full) MaxDD%': round(100*max_dd((1+r_kelly_f).cumprod()), 2),
    }

kelly_tbl = pd.DataFrame(kelly_results).T
kelly_tbl.to_csv('iter9_kelly_sizing.csv')
print('E1 Kelly Sizing:\n', kelly_tbl.to_string())

# ======================================================================
# E2. Realized Volatility Forecasting (from Forex articles)
# ======================================================================
def realized_vol(returns, window=30):
    """Realized volatility: rolling std of returns."""
    return returns.rolling(window).std() * np.sqrt(252)

def rv_features(px, rets, windows=[5, 10, 21, 63, 126]):
    """Create realized volatility features at multiple horizons."""
    feat = pd.DataFrame(index=px.index)
    for col in ['SPY', 'TLT', 'GLD', 'EFA']:
        if col in px.columns:
            for w in windows:
                feat[f'{col}_rv_{w}'] = realized_vol(rets[col], w)
            # Vol of vol
            for w in [21, 63]:
                rv = realized_vol(rets[col], w)
                feat[f'{col}_vov_{w}'] = rv.rolling(w).std()
            # Vol regime (high/low)
            rv_63 = realized_vol(rets[col], 63)
            feat[f'{col}_rv_regime'] = (rv_63 > rv_63.rolling(252).median()).astype(int)
    return feat.dropna()

print("E2 Creating realized vol features...")
rv_feat = rv_features(px, rets)

# Target: next 21-day realized vol
target_rv = realized_vol(rets['SPY'], 21).shift(-21)

common_idx = rv_feat.index.intersection(target_rv.dropna().index)
X = rv_feat.loc[common_idx]
y = target_rv.loc[common_idx]

split = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# SVR for realized vol prediction (from "Using SVMs to predict market regime change")
svr_models = {
    'Linear SVR': SVR(kernel='linear', C=1.0),
    'RBF SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
    'RBF SVR(C=10)': SVR(kernel='rbf', C=10.0, gamma='scale'),
}

rv_results = {}
for name, model in svr_models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    
    mse = np.mean((pred - y_test)**2)
    corr = np.corrcoef(pred, y_test)[0,1]
    dir_acc = np.mean((pred > y_test.median()) == (y_test > y_test.median()))
    
    # Strategy: reduce exposure when predicted vol high
    w_svr = w_sma.copy()
    pred_series = pd.Series(pred, index=X_test.index)
    high_vol = pred_series > pred_series.median()
    for t in high_vol[high_vol].index:
        if t in w_svr.index:
            w_svr.loc[t] *= 0.5
    
    r_svr = backtest(w_svr, rets, COST).dropna()
    
    rv_results[name] = {
        'MSE': round(mse * 1e6, 2),
        'Correlation': round(corr, 3),
        'Dir Acc': round(dir_acc, 3),
        'Strategy Sharpe': round(sharpe(r_svr), 2),
        'Strategy AnnRet%': round(252*r_svr.mean()*100, 2),
        'Strategy MaxDD%': round(100*max_dd((1+r_svr).cumprod()), 2),
    }

rv_tbl = pd.DataFrame(rv_results).T
rv_tbl.to_csv('iter9_rv_forecasting.csv')
print('E2 RV Forecasting:\n', rv_tbl.to_string())

# ======================================================================
# E3. SVM for Regime Classification (from "Using SVMs to predict market regime change")
# ======================================================================
# Define regimes based on realized vol and momentum
spy_rv = realized_vol(rets['SPY'], 21)
spy_mom = px['SPY'].pct_change(63)

regime = pd.Series(1, index=px.index)  # 0=low vol bull, 1=normal, 2=high vol bear
regime[(spy_mom > 0) & (spy_rv < spy_rv.rolling(252).quantile(0.33))] = 0
regime[(spy_mom < 0) & (spy_rv > spy_rv.rolling(252).quantile(0.67))] = 2

# Features for SVM
def svm_features(px, rets):
    feat = pd.DataFrame(index=px.index)
    for col in ['SPY', 'TLT', 'GLD']:
        if col in px.columns:
            feat[f'{col}_ret_5'] = px[col].pct_change(5)
            feat[f'{col}_ret_21'] = px[col].pct_change(21)
            feat[f'{col}_ret_63'] = px[col].pct_change(63)
            feat[f'{col}_rv_21'] = realized_vol(rets[col], 21)
            feat[f'{col}_rv_63'] = realized_vol(rets[col], 63)
            # Moving average distance
            feat[f'{col}_ma_dist_50'] = px[col] / px[col].rolling(50).mean() - 1
            feat[f'{col}_ma_dist_200'] = px[col] / px[col].rolling(200).mean() - 1
    return feat.dropna()

svm_feat = svm_features(px, rets)
common_idx = svm_feat.index.intersection(regime.dropna().index)
X_svm = svm_feat.loc[common_idx]
y_svm = regime.loc[common_idx]

split = int(len(X_svm) * 0.7)
X_svm_train, X_svm_test = X_svm.iloc[:split], X_svm.iloc[split:]
y_svm_train, y_svm_test = y_svm.iloc[:split], y_svm.iloc[split:]

X_svm_train_s = scaler.fit_transform(X_svm_train)
X_svm_test_s = scaler.transform(X_svm_test)

svc_models = {
    'Linear SVC': SVC(kernel='linear', C=1.0, probability=True),
    'RBF SVC': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
    'RBF SVC(C=10)': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True),
}

svc_results = {}
for name, model in svc_models.items():
    model.fit(X_svm_train_s, y_svm_train)
    pred = model.predict(X_svm_test_s)
    proba = model.predict_proba(X_svm_test_s)
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_svm_test, pred)
    
    # Strategy: long in regime 0, half in 1, cash in 2
    w_svc = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    pred_series = pd.Series(pred, index=X_svm_test.index)
    for t, p in pred_series.items():
        if p == 0:
            w_svc.loc[t, 'SPY'] = 1.0
        elif p == 1:
            w_svc.loc[t, 'SPY'] = 0.5
        else:
            w_svc.loc[t, 'SPY'] = 0.0
    w_svc = w_svc.ffill().fillna(0)
    
    r_svc = backtest(w_svc, rets, COST).dropna()
    
    svc_results[name] = {
        'Accuracy': round(acc, 3),
        'Strategy Sharpe': round(sharpe(r_svc), 2),
        'Strategy AnnRet%': round(252*r_svc.mean()*100, 2),
        'Strategy MaxDD%': round(100*max_dd((1+r_svc).cumprod()), 2),
    }

svc_tbl = pd.DataFrame(svc_results).T
svc_tbl.to_csv('iter9_svm_regime.csv')
print('E3 SVM Regime:\n', svc_tbl.to_string())

# ======================================================================
# E4. Forex-Style Carry & Momentum (adapted for ETFs)
# ======================================================================
# Carry: long high-yield, short low-yield (approximate with div yield / term spread)
# Momentum: cross-asset momentum

# Proxy carry using bond ETFs: long TLT (long-term), short SHY (short-term) when curve steep
# Or: long high-yield (HYG), short investment grade (LQD) when spread tight

carry_signals = {}
# 1. Term structure carry: TLT vs SHY/IEF
if 'TLT' in px.columns and 'IEF' in px.columns:
    term_spread = px['TLT'].pct_change(63) - px['IEF'].pct_change(63)
    carry_signals['Term'] = (term_spread > 0).astype(float)

# 2. Credit carry: HYG vs LQD
if 'HYG' in px.columns and 'LQD' in px.columns:
    credit_spread = px['HYG'].pct_change(63) - px['LQD'].pct_change(63)
    carry_signals['Credit'] = (credit_spread > 0).astype(float)

# 3. Equity carry: SPY vs TLT (risk-on vs risk-off)
if 'SPY' in px.columns and 'TLT' in px.columns:
    eq_spread = px['SPY'].pct_change(63) - px['TLT'].pct_change(63)
    carry_signals['Equity'] = (eq_spread > 0).astype(float)

# Combine carry signals
if carry_signals:
    carry_df = pd.DataFrame(carry_signals)
    carry_combined = carry_df.mean(axis=1)
    w_carry = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for t in carry_combined[carry_combined > 0].index:
        w_carry.loc[t, 'SPY'] = 0.5
        w_carry.loc[t, 'TLT'] = 0.5
    for t in carry_combined[carry_combined <= 0].index:
        w_carry.loc[t, 'TLT'] = 1.0
    w_carry = w_carry.ffill().fillna(0)
    
    r_carry = backtest(w_carry, rets, COST).dropna()
    print(f"E4 Carry Strategy Sharpe: {sharpe(r_carry):.2f}")

# Forex-style momentum: cross-asset 12-1 momentum
fx_assets = ['SPY', 'EFA', 'EEM', 'TLT', 'GLD', 'DBC', 'VNQ']
fx_assets = [a for a in fx_assets if a in px.columns]
fx_mom = px[fx_assets] / px[fx_assets].shift(252) - 1
fx_ranks = fx_mom.rank(axis=1, ascending=False)
w_fx_mom = (fx_ranks <= 3).astype(float)
w_fx_mom = w_fx_mom.div(w_fx_mom.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).reindex(columns=px.columns).fillna(0)

r_fx_mom = backtest(w_fx_mom, rets, COST).dropna()
print(f"E4 FX-Style Momentum Sharpe: {sharpe(r_fx_mom):.2f}")

# ======================================================================
# E5. Advanced Performance Metrics (Sortino, Calmar, Omega, Tail Ratio)
# ======================================================================
def sortino_ratio(returns, target=0):
    """Sortino ratio: excess return / downside deviation."""
    excess = returns - target/252
    downside = excess[excess < 0]
    if len(downside) > 0:
        return excess.mean() * np.sqrt(252) / downside.std()
    return np.nan

def calmar_ratio(returns):
    """Calmar: annual return / max drawdown."""
    ann_ret = returns.mean() * 252
    dd = max_dd((1+returns).cumprod())
    if dd < 0:
        return ann_ret / abs(dd)
    return np.nan

def omega_ratio(returns, threshold=0):
    """Omega ratio: probability-weighted gains / losses."""
    excess = returns - threshold/252
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses > 0:
        return gains / losses
    return np.nan

def tail_ratio(returns, p=0.05):
    """Tail ratio: 95th percentile / 5th percentile (absolute)."""
    q95 = returns.quantile(1-p)
    q05 = returns.quantile(p)
    if q05 != 0:
        return q95 / abs(q05)
    return np.nan

def gain_to_pain(returns):
    """Gain-to-Pain ratio: sum of gains / sum of losses."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses > 0:
        return gains / losses
    return np.nan

# Compute for all strategies
all_strats = {'SMA200': w_sma, 'GEM': w_gem, 'XSec Mom': w_xs, 'TSMOM+RP': w_rp,
              'FX Mom': w_fx_mom}
if carry_signals:
    all_strats['Carry'] = w_carry

metrics_results = {}
for name, w in all_strats.items():
    r = backtest(w, rets, COST).dropna()
    metrics_results[name] = {
        'Sharpe': round(sharpe(r), 2),
        'Sortino': round(sortino_ratio(r), 2),
        'Calmar': round(calmar_ratio(r), 2),
        'Omega': round(omega_ratio(r), 2),
        'TailRatio': round(tail_ratio(r), 2),
        'GainToPain': round(gain_to_pain(r), 2),
        'Skew': round(skew(r), 2),
        'Kurtosis': round(kurtosis(r), 2),
    }

metrics_tbl = pd.DataFrame(metrics_results).T
metrics_tbl.to_csv('iter9_advanced_metrics.csv')
print('E5 Advanced Metrics:\n', metrics_tbl.to_string())

# ======================================================================
# E6. Parameter Optimization with Kelly + Walk-Forward
# ======================================================================
# Optimize SMA window + Kelly fraction jointly using walk-forward
def optimize_sma_kelly_wf(px, rets, train_window=504, test_window=63, step=21):
    """Walk-forward optimization of SMA window and Kelly leverage."""
    n = len(rets)
    results = []
    
    for i in range(train_window, n - test_window, step):
        train_end = i
        test_end = min(i + test_window, n)
        
        # Grid search on training data
        best_sharpe = -np.inf
        best_window = 200
        best_kelly = 1.0
        
        for window in [100, 150, 200, 250, 300]:
            w = (px[['SPY']][:train_end] > px[['SPY']][:train_end].rolling(window).mean()).astype(float).reindex(columns=px.columns).fillna(0)
            r = backtest(w, rets[:train_end], COST).dropna()
            if len(r) > 50:
                # Gaussian Kelly
                f = kelly_fraction(r)
                w_k = w.mul(f).clip(upper=1.0)
                r_k = backtest(w_k, rets[:train_end], COST).dropna()
                s = sharpe(r_k)
                if s > best_sharpe:
                    best_sharpe = s
                    best_window = window
                    best_kelly = f
        
        # Test on OOS
        w_oos = (px[['SPY']][train_end:test_end] > px[['SPY']][train_end:test_end].rolling(best_window).mean()).astype(float).reindex(columns=px.columns).fillna(0)
        w_k_oos = w_oos.mul(best_kelly).clip(upper=1.0)
        r_k_oos = backtest(w_k_oos, rets[train_end:test_end], COST).dropna()
        
        if len(r_k_oos) > 20:
            results.append({
                'Period': f'{rets.index[train_end].date()} to {rets.index[test_end-1].date()}',
                'Window': best_window,
                'Kelly': round(best_kelly, 2),
                'OOS_Sharpe': round(sharpe(r_k_oos), 2)
            })
    
    return pd.DataFrame(results)

wf_opt = optimize_sma_kelly_wf(px, rets)
wf_opt.to_csv('iter9_wf_kelly_opt.csv', index=False)
print('E6 WF Kelly Optimization:\n', wf_opt.to_string(index=False))

# ======================================================================
# Compile Iteration #9
# ======================================================================
combined = {**strategies}
if carry_signals:
    combined['Carry'] = w_carry
combined['FX Mom'] = w_fx_mom

# Add best SVM strategy
if 'RBF SVC' in svc_results:
    model = svc_models['RBF SVC']
    model.fit(X_svm_train_s, y_svm_train)
    pred = model.predict(X_svm_test_s)
    w_svc = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    pred_series = pd.Series(pred, index=X_svm_test.index)
    for t, p in pred_series.items():
        if p == 0:
            w_svc.loc[t, 'SPY'] = 1.0
        elif p == 1:
            w_svc.loc[t, 'SPY'] = 0.5
        else:
            w_svc.loc[t, 'SPY'] = 0.0
    w_svc = w_svc.ffill().fillna(0)
    combined['SVM-Regime'] = w_svc

results = []
all_rets = {}
for name, obj in combined.items():
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
vals = [report_validation(results[i]['name'], list(all_rets.values())[i], len(combined), sr_std)
        for i in range(len(results)) if len(list(all_rets.values())[i]) > 100]
val = pd.DataFrame(vals)

# Save
res.to_csv('iter9_comprehensive_perf.csv', index=False)
wf.to_csv('iter9_comprehensive_walkforward.csv', index=False)
val.to_csv('iter9_comprehensive_validation.csv', index=False)

# Plots
plt.figure(figsize=(12, 7))
for name, r in all_rets.items():
    if len(r) > 100:
        plt.plot((1+r).cumprod(), label=name, alpha=0.8, linewidth=0.8)
plt.legend(fontsize=7, ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Iteration #9 — All Strategies Equity Curves (net 10bp)')
plt.ylabel('Growth of $1'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('iter9_equity.png', dpi=110, bbox_inches='tight')

# Kelly comparison
plt.figure(figsize=(10, 6))
kelly_tbl[['Base Sharpe', 'Kelly(Gauss) Sharpe', 'Kelly(Full) Sharpe']].plot(kind='bar')
plt.title('Kelly Sizing Impact on Sharpe')
plt.ylabel('Sharpe'); plt.tight_layout(); plt.savefig('iter9_kelly.png', dpi=110)

# Advanced metrics radar
plt.figure(figsize=(10, 6))
metrics_plot = metrics_tbl[['Sharpe', 'Sortino', 'Calmar', 'Omega']].T
metrics_plot.plot(kind='bar')
plt.title('Advanced Metrics Comparison')
plt.ylabel('Value'); plt.tight_layout(); plt.savefig('iter9_metrics.png', dpi=110)

# RV forecasting
plt.figure(figsize=(10, 6))
rv_tbl['Strategy Sharpe'].plot(kind='bar', color='coral')
plt.title('SVR Vol Forecasting: Strategy Sharpe')
plt.ylabel('Sharpe'); plt.tight_layout(); plt.savefig('iter9_rv.png', dpi=110)

# SVM regime
plt.figure(figsize=(10, 6))
svc_tbl['Strategy Sharpe'].plot(kind='bar', color='steelblue')
plt.title('SVM Regime Classification: Strategy Sharpe')
plt.ylabel('Sharpe'); plt.tight_layout(); plt.savefig('iter9_svm.png', dpi=110)

print('\n=== ITERATION #9 PERFORMANCE ===')
print(res.to_string(index=False))
print('\n=== WALK-FORWARD ===')
print(wf.to_string(index=False))
print('\n=== VALIDATION ===')
print(val.to_string(index=False))
print('\nIteration #9 complete.')