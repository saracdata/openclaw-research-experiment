"""Iteration #2 — QuantStart themes: Survivorship bias, Kelly sizing,
Execution/slippage layer, Meta-strategy ensemble, Walk-forward optimization.
Outputs: iter2_*.csv, iter2_*.png, updated REPORT.md
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

COST = 10
# Use a broader universe to test survivorship effects
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'AGG',
           'VNQ', 'DBC', 'XLP', 'XLU', 'XLE', 'XLF', 'IEI', 'VIG', 'SCHD',
           'MDY', 'XLK', 'XLV', 'VTI', 'VEA', 'VWO', 'GOVT', 'SHY', 'BIL',
           'LQD', 'HYG', 'SMH']
px = pd.DataFrame({t: load(t)['Close'] for t in TICKERS}).dropna()
rets = px.pct_change()

# ======================================================================
# Base strategies (from iteration 1, re-computed on this universe)
# ======================================================================
# 1. 60/40 SPY/AGG (monthly rebalanced, executed next day)
def rebalanced_portfolio(weights_dict):
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    total = sum(weights_dict.values())
    norm_w = {k: v/total for k, v in weights_dict.items()}
    # Month ends
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

w_6040 = rebalanced_portfolio({'SPY': 0.6, 'AGG': 0.4})
w_aw = rebalanced_portfolio({'VTI': 0.30, 'TLT': 0.40, 'IEI': 0.15, 'GLD': 0.075, 'GSG': 0.075})

# 2. Dual Momentum GEM (Antonacci exact)
mom = px[['SPY','EFA','AGG']] / px[['SPY','EFA','AGG']].shift(126) - 1
w_gem = pd.DataFrame(0.0, index=px.index, columns=px.columns)
valid = mom.dropna().index
for t in valid:
    spy_mom = mom.loc[t, 'SPY']
    if spy_mom > 0:
        best = mom.loc[t, ['SPY','EFA']].idxmax()
        w_gem.loc[t, best] = 1.0
    else:
        w_gem.loc[t, 'AGG'] = 1.0

# 3. SMA200 Trend (SPY)
w_sma = (px[['SPY']] > px[['SPY']].rolling(200).mean()).astype(float)
w_sma = w_sma.reindex(columns=px.columns).fillna(0)

# 4. RSI(2) MeanRev (SPY)
from strategies import rsi2_meanrev
w_rsi = rsi2_meanrev(px['SPY']).to_frame('SPY').reindex(columns=px.columns).fillna(0)

# 5. Multi-asset TSMOM + Risk Parity
mom12 = px / px.shift(252 + 21) - 1
signs = (mom12 > 0).astype(float) * 2 - 1
vol = rets.rolling(63).std() * np.sqrt(252)
inv_vol = 1.0 / vol.replace(0, np.nan)
w_rp = signs.mul(inv_vol).div(inv_vol.abs().sum(axis=1), axis=0).fillna(0)

# 6. Cross-sectional momentum (top 5 of 15)
RISK = ['SPY','QQQ','IWM','EFA','EEM','VNQ','DBC','XLK','XLF','XLE','XLV','MDY','SMH','VIG']
RISK = [t for t in RISK if t in px.columns]
mom_xs = px[RISK] / px[RISK].shift(252) - 1
ranks = mom_xs.rank(axis=1, ascending=False)
w_xs = (ranks <= 5).astype(float)
w_xs = w_xs.div(w_xs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
w_xs = w_xs.reindex(columns=px.columns).fillna(0)

# ======================================================================
# E1. Survivorship Bias Simulation
# ======================================================================
# Simulate survivorship bias by: (a) dropping worst performers mid-sample,
# (b) comparing results with full universe vs survivor-only universe
def simulate_survivorship_bias(px, rets, drop_frac=0.3, drop_date='2019-01-01'):
    """Drop bottom performers after drop_date to simulate survivorship bias."""
    # Identify worst performers in first half
    early_rets = rets[rets.index < drop_date]
    mean_ret = early_rets.mean()
    worst = mean_ret.nsmallest(int(len(mean_ret) * drop_frac)).index.tolist()
    # Create survivor-only universe (remove worst)
    survivor_cols = [c for c in px.columns if c not in worst]
    px_surv = px[survivor_cols]
    rets_surv = rets[survivor_cols]
    return px_surv, rets_surv, worst

px_surv, rets_surv, dropped = simulate_survivorship_bias(px, rets)
print(f'E1 Dropped tickers (survivorship sim): {dropped}')

# Test SMA200 on full vs survivor universe
w_sma_surv = (px_surv[['SPY']] > px_surv[['SPY']].rolling(200).mean()).astype(float).reindex(columns=px_surv.columns).fillna(0)
r_sma_full = backtest(w_sma.reindex(columns=px.columns).fillna(0), rets, COST).dropna()
r_sma_surv = backtest(w_sma_surv, rets_surv, COST).dropna()

surv_tbl = pd.DataFrame([
    {'Universe': 'Full', 'Sharpe': round(sharpe(r_sma_full), 2), 'AnnRet%': round(252*r_sma_full.mean()*100, 2), 'MaxDD%': round(100*max_dd((1+r_sma_full).cumprod()), 2)},
    {'Universe': 'Survivor-only', 'Sharpe': round(sharpe(r_sma_surv), 2), 'AnnRet%': round(252*r_sma_surv.mean()*100, 2), 'MaxDD%': round(100*max_dd((1+r_sma_surv).cumprod()), 2)},
])
surv_tbl.to_csv('iter2_survivorship_bias.csv', index=False)
print('E1 Survivorship bias:\n', surv_tbl.to_string(index=False))

# ======================================================================
# E2. Kelly / Optimal Position Sizing
# ======================================================================
def kelly_fraction(returns, leverage_cap=2.0):
    """Kelly fraction f = (mu - rf) / sigma^2 for Gaussian returns."""
    r = returns.dropna()
    if r.std() == 0: return 0.0
    f = (r.mean() - 0) / (r.std() ** 2) * 252  # annualized
    return float(np.clip(f, 0, leverage_cap))

def apply_kelly_sizing(weights, base_returns, lookback=252, cap=2.0):
    """Dynamically scale weights by Kelly fraction estimated on trailing window."""
    kelly = pd.Series(0.0, index=base_returns.index)
    for i in range(lookback, len(base_returns)):
        hist = base_returns.iloc[i-lookback:i]
        strat_ret = (weights.iloc[i-lookback:i] * hist).sum(axis=1)
        f = kelly_fraction(strat_ret, cap)
        kelly.iloc[i] = f
    kelly = kelly.ffill().fillna(1.0)
    return weights.mul(kelly, axis=0)

# Apply Kelly to SMA200 and GEM
r_sma_base = backtest(w_sma, rets, COST).dropna()
w_sma_kelly = apply_kelly_sizing(w_sma, rets)
r_sma_kelly = backtest(w_sma_kelly, rets, COST).dropna()

r_gem_base = backtest(w_gem, rets, COST).dropna()
w_gem_kelly = apply_kelly_sizing(w_gem, rets)
r_gem_kelly = backtest(w_gem_kelly, rets, COST).dropna()

kelly_tbl = pd.DataFrame([
    {'Strategy': 'SMA200', 'Sizing': 'Fixed', 'Sharpe': round(sharpe(r_sma_base), 2), 'AnnRet%': round(252*r_sma_base.mean()*100, 2)},
    {'Strategy': 'SMA200', 'Sizing': 'Kelly', 'Sharpe': round(sharpe(r_sma_kelly), 2), 'AnnRet%': round(252*r_sma_kelly.mean()*100, 2)},
    {'Strategy': 'GEM', 'Sizing': 'Fixed', 'Sharpe': round(sharpe(r_gem_base), 2), 'AnnRet%': round(252*r_gem_base.mean()*100, 2)},
    {'Strategy': 'GEM', 'Sizing': 'Kelly', 'Sharpe': round(sharpe(r_gem_kelly), 2), 'AnnRet%': round(252*r_gem_kelly.mean()*100, 2)},
])
kelly_tbl.to_csv('iter2_kelly_sizing.csv', index=False)
print('E2 Kelly sizing:\n', kelly_tbl.to_string(index=False))

# ======================================================================
# E3. Execution Layer with Slippage & Bid-Ask
# ======================================================================
def backtest_with_slippage(weights, returns, cost_bps=COST, slippage_bps=5, 
                           bid_ask_bps=3, adv_pct=0.05):
    """
    Extended backtest with:
    - Explicit slippage (bps per trade)
    - Bid-ask spread cost (half-spread per side)
    - Market impact: linear in participation rate (adv_pct of daily volume)
    Note: We don't have volume data for all tickers, so use a proxy.
    """
    pos = weights.shift(1).fillna(0.0)
    turnover = pos.diff().abs()
    turnover.iloc[0] = pos.iloc[0].abs()
    
    # Explicit costs
    explicit_cost = turnover.sum(axis=1) * cost_bps / 1e4
    
    # Slippage (assume we cross spread)
    slippage_cost = turnover.sum(axis=1) * slippage_bps / 1e4
    
    # Bid-ask spread (half spread paid per trade)
    spread_cost = turnover.sum(axis=1) * bid_ask_bps / 1e4
    
    # Simple market impact: assume 5 bps per 10% ADV participation
    # ADV ~ 1% of portfolio value per day for liquid ETFs
    participation = turnover.sum(axis=1) * adv_pct
    impact_cost = participation * 5 / 1e4  # 5 bps per 10% ADV
    
    total_cost = explicit_cost + slippage_cost + spread_cost + impact_cost
    strat = (pos * returns).sum(axis=1) - total_cost
    return strat

# Test on all base strategies
base_strategies = {
    '60/40': w_6040, 'All Weather': w_aw, 'GEM': w_gem,
    'SMA200': w_sma, 'RSI(2)': w_rsi, 'TSMOM+RP': w_rp, 'XSec Mom': w_xs
}

slip_tbl = []
for name, w in base_strategies.items():
    r = backtest_with_slippage(w, rets, cost_bps=10, slippage_bps=5, bid_ask_bps=3)
    r = r.dropna()
    slip_tbl.append({'Strategy': name, 'Sharpe': round(sharpe(r), 2),
                     'AnnRet%': round(252*r.mean()*100, 2),
                     'MaxDD%': round(100*max_dd((1+r).cumprod()), 2)})
slip_tbl = pd.DataFrame(slip_tbl)
slip_tbl.to_csv('iter2_execution_slippage.csv', index=False)
print('E3 Execution with slippage:\n', slip_tbl.to_string(index=False))

# ======================================================================
# E4. Meta-Strategy Ensemble (combine signals)
# ======================================================================
# Ensemble methods: equal-weight signals, volatility-weighted, correlation-weighted
signals = {
    'SMA200': w_sma,
    'GEM': w_gem,
    'TSMOM+RP': w_rp,
    'XSec Mom': w_xs,
    'RSI(2)': w_rsi,
}

# Equal-weight ensemble
w_ensemble_eq = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for s in signals.values():
    w_ensemble_eq += s
w_ensemble_eq = w_ensemble_eq.div(len(signals)).fillna(0)

# Volatility-weighted ensemble (inverse vol of each signal's returns)
sig_rets = {name: backtest(sig, rets, COST).dropna() for name, sig in signals.items()}
vols = {name: r.rolling(63).std() * np.sqrt(252) for name, r in sig_rets.items()}
inv_vols = {name: 1.0 / v.replace(0, np.nan) for name, v in vols.items()}
w_ensemble_vol = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for name, sig in signals.items():
    w_ensemble_vol += sig.mul(inv_vols[name].reindex(sig.index), axis=0)
w_ensemble_vol = w_ensemble_vol.div(w_ensemble_vol.abs().sum(axis=1).replace(0, 1), axis=0).fillna(0)

# Correlation-weighted ensemble (minimize portfolio variance)
# Use trailing 252-day correlation of signal returns
w_ensemble_corr = pd.DataFrame(0.0, index=px.index, columns=px.columns)
lookback = 252
for i in range(lookback, len(px)):
    hist = pd.DataFrame({n: r.iloc[i-lookback:i] for n, r in sig_rets.items()})
    if len(hist.dropna()) >= 60:
        cov = hist.cov() * 252
        # Minimum variance weights
        try:
            inv_cov = np.linalg.pinv(cov.values)
            ones = np.ones(len(cov))
            w_mv = inv_cov @ ones / (ones @ inv_cov @ ones)
            w_mv = np.maximum(w_mv, 0)  # long only
            w_mv = w_mv / w_mv.sum()
            # Apply to signals at i
            for j, name in enumerate(signals.keys()):
                w_ensemble_corr.iloc[i] += signals[name].iloc[i] * w_mv[j]
        except:
            pass

ensemble_strategies = {
    'Ensemble Equal': w_ensemble_eq,
    'Ensemble Vol-Weighted': w_ensemble_vol,
    'Ensemble Min-Var': w_ensemble_corr,
}

ensemble_tbl = []
for name, w in ensemble_strategies.items():
    r = backtest(w, rets, COST).dropna()
    ensemble_tbl.append({'Strategy': name, 'Sharpe': round(sharpe(r), 2),
                         'AnnRet%': round(252*r.mean()*100, 2),
                         'MaxDD%': round(100*max_dd((1+r).cumprod()), 2)})
ensemble_tbl = pd.DataFrame(ensemble_tbl)
ensemble_tbl.to_csv('iter2_ensemble.csv', index=False)
print('E4 Ensemble strategies:\n', ensemble_tbl.to_string(index=False))

# ======================================================================
# E5. Walk-Forward Optimization (Expanding Window)
# ======================================================================
def walk_forward_optimize(px, rets, train_window=504, test_window=63, 
                          reopt_every=21, cost_bps=COST):
    """
    Expanding window walk-forward:
    - Train on all data up to t
    - Optimize max-Sharpe on train
    - Apply on test_window days
    - Re-optimize every reopt_every days
    """
    cols = list(px.columns)
    n = len(px)
    w = pd.DataFrame(0.0, index=px.index, columns=cols)
    
    for start_test in range(train_window, n - test_window, reopt_every):
        end_test = min(start_test + test_window, n)
        train_rets = rets.iloc[:start_test]
        
        # Optimize on training data
        if len(train_rets) >= 252:
            mu = train_rets.mean().values * 252
            cov = train_rets.cov().values * 252
            cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))  # shrinkage
            
            def neg_sharpe(ww):
                ret = ww @ mu
                vol = np.sqrt(max(ww @ cov @ ww, 1e-12))
                return -ret / vol * np.sqrt(252)
            
            bounds = [(0.0, 1.0)] * len(cols)
            cons = [{"type": "eq", "fun": lambda ww: np.sum(ww) - 1.0}]
            w0 = np.ones(len(cols)) / len(cols)
            res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                          constraints=cons, options={"maxiter": 300})
            w_opt = np.clip(res.x, 0, 1)
            w_opt = w_opt / w_opt.sum() if w_opt.sum() > 0 else w0
            
            # Apply on test period
            for i in range(start_test, end_test):
                w.iloc[i] = w_opt
    
    w = w.ffill().fillna(0)
    return backtest(w, rets, cost_bps).dropna()

r_wfo = walk_forward_optimize(px, rets)
wfo_tbl = pd.DataFrame([{'Strategy': 'Walk-Forward Optimized', 
                          'Sharpe': round(sharpe(r_wfo), 2),
                          'AnnRet%': round(252*r_wfo.mean()*100, 2),
                          'MaxDD%': round(100*max_dd((1+r_wfo).cumprod()), 2)}])
wfo_tbl.to_csv('iter2_walkforward_opt.csv', index=False)
print('E5 Walk-forward optimization:\n', wfo_tbl.to_string(index=False))

# ======================================================================
# Compile all Iteration #2 strategies
# ======================================================================
all_strategies = {}
all_strategies.update(base_strategies)
all_strategies['SMA200 Kelly'] = w_sma_kelly
all_strategies['GEM Kelly'] = w_gem_kelly
all_strategies.update(ensemble_strategies)
all_strategies['Walk-Forward Opt'] = r_wfo  # this is already returns

# Compute performance for all
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
    shs = [round(sharpe(r.iloc[a:b]), 2) for a, b in fb]
    wf_rows.append({'Strategy': name, 'F1': shs[0], 'F2': shs[1], 'F3': shs[2], 'F4': shs[3]})
wf = pd.DataFrame(wf_rows)

# Validation
sharpes = [sharpe(r) for r in all_rets.values()]
sr_std = float(np.std(sharpes, ddof=1))
vals = [report_validation(results[i]['name'], list(all_rets.values())[i], len(all_rets), sr_std)
        for i in range(len(results))]
val = pd.DataFrame(vals)

# Save
res.to_csv('iter2_comprehensive_perf.csv', index=False)
wf.to_csv('iter2_comprehensive_walkforward.csv', index=False)
val.to_csv('iter2_comprehensive_validation.csv', index=False)

# Equity curves
plt.figure(figsize=(12, 7))
for name, r in all_rets.items():
    plt.plot((1+r).cumprod(), label=name, alpha=0.8, linewidth=0.8)
plt.legend(fontsize=7, ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Iteration #2 — All Strategies Equity Curves (net 10bp)')
plt.ylabel('Growth of $1'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('iter2_equity.png', dpi=110, bbox_inches='tight')

# Best performers bar chart
top5 = res.nlargest(5, 'Sharpe')
plt.figure(figsize=(8, 4))
plt.barh(top5['name'], top5['Sharpe'], color='steelblue')
plt.xlabel('Sharpe'); plt.title('Top 5 Strategies by Sharpe (Iteration #2)')
plt.tight_layout(); plt.savefig('iter2_top5_sharpe.png', dpi=110)

print('\n=== ITERATION #2 PERFORMANCE ===')
print(res.to_string(index=False))
print('\n=== WALK-FORWARD ===')
print(wf.to_string(index=False))
print('\n=== VALIDATION ===')
print(val.to_string(index=False))
print('\nIteration #2 complete.')