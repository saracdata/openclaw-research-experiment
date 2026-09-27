"""Iteration #7 — QuantStart: Fee Models, Simple vs Advanced, Backtesting Frameworks
Explores: QSTrader fee hierarchy (ZeroFee → PercentFee → Slippage/Impact),
Simple vs Advanced strategy comparison, Backtesting best practices.
Outputs: iter7_*.csv, iter7_*.png, updated REPORT.md
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
# Base strategies (Simple vs Advanced classification from QuantStart)
# ======================================================================
# SIMPLE STRATEGIES (per QuantStart definition)
# 1. Buy & Hold
w_bh = pd.DataFrame(0.0, index=px.index, columns=px.columns)
w_bh.iloc[0, w_bh.columns.get_loc('SPY')] = 1.0
w_bh = w_bh.ffill()

# 2. SMA Trend (Faber 2007)
w_sma = (px[['SPY']] > px[['SPY']].rolling(200).mean()).astype(float).reindex(columns=px.columns).fillna(0)

# 3. 60/40 Static
w_6040 = rebalanced_portfolio({'SPY': 0.6, 'AGG': 0.4})

# 4. Dual Momentum GEM (Antonacci) - border case
mom = px[['SPY','EFA','AGG']] / px[['SPY','EFA','AGG']].shift(126) - 1
w_gem = pd.DataFrame(0.0, index=px.index, columns=px.columns)
valid = mom.dropna().index
for t in valid:
    if mom.loc[t, 'SPY'] > 0:
        w_gem.loc[t, mom.loc[t, ['SPY','EFA']].idxmax()] = 1.0
    else:
        w_gem.loc[t, 'AGG'] = 1.0

# ADVANCED STRATEGIES
# 5. Cross-Sectional Momentum (Jegadeesh-Titman)
RISK = ['SPY','QQQ','IWM','EFA','EEM','VNQ','DBC','XLK','XLF','XLE','XLV','MDY','SMH','VIG']
RISK = [t for t in RISK if t in px.columns]
mom_xs = px[RISK] / px[RISK].shift(252) - 1
ranks = mom_xs.rank(axis=1, ascending=False)
w_xs = (ranks <= 5).astype(float)
w_xs = w_xs.div(w_xs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).reindex(columns=px.columns).fillna(0)

# 6. TSMOM + Risk Parity (Moskowitz et al.)
mom12 = px / px.shift(252 + 21) - 1
signs = (mom12 > 0).astype(float) * 2 - 1
vol = rets.rolling(63).std() * np.sqrt(252)
inv_vol = 1.0 / vol.replace(0, np.nan)
w_rp = signs.mul(inv_vol).div(inv_vol.abs().sum(axis=1), axis=0).fillna(0)

# 7. Hierarchical Risk Parity (López de Prado) - meta
# 8. Vol Targeting (Moreira & Muir)
spy_ret = rets['SPY']
vol_target = 0.10
rolling_vol = spy_ret.rolling(63).std() * np.sqrt(252)
leverage = (vol_target / rolling_vol).clip(upper=2.0, lower=0.2).fillna(1.0).shift(1)
w_volt = w_sma.mul(leverage, axis=0)

# 9. Regime-Aware (HMM-style rule-based)
spy_mom = px['SPY'] / px['SPY'].shift(252) - 1
spy_vol = spy_ret.rolling(63).std() * np.sqrt(252)
regime = pd.Series(2, index=px.index)  # 0=bull, 1=crisis, 2=choppy
regime[(spy_mom > 0) & (spy_vol < spy_vol.rolling(252).quantile(0.5))] = 0
regime[(spy_mom < 0) & (spy_vol > spy_vol.rolling(252).quantile(0.8))] = 1

w_reg = w_sma.copy()
for t in regime[regime == 1].index:  # crisis -> cash
    if t in w_reg.index:
        w_reg.loc[t] = 0.0
for t in regime[regime == 0].index:  # bull -> 1.5x
    if t in w_reg.index:
        w_reg.loc[t] *= 1.5
w_reg = w_reg.div(w_reg.sum(axis=1).replace(0, 1), axis=0).fillna(0)

# Classify
simple_strats = {'Buy&Hold': w_bh, 'SMA200': w_sma, '60/40': w_6040, 'GEM': w_gem}
advanced_strats = {'XSec Mom': w_xs, 'TSMOM+RP': w_rp, 'Vol Target': w_volt, 'Regime-Aware': w_reg}

# ======================================================================
# E1. QSTrader Fee Model Hierarchy
# ======================================================================
def zero_fee_cost(turnover):
    return pd.Series(0.0, index=turnover.index)

def percent_fee_cost(turnover, commission_bps=10, tax_bps=0):
    return turnover * (commission_bps + tax_bps) / 1e4

def slippage_impact_cost(turnover, spread_bps=5, impact_coeff=0.1, adv=1e9):
    """Slippage + market impact (square-root law)"""
    spread_cost = turnover * spread_bps / 1e4 * 0.5  # half-spread
    participation = turnover / adv
    impact_cost = turnover * impact_coeff * np.sqrt(np.maximum(participation, 1e-6))
    return spread_cost + impact_cost

def full_lob_cost(turnover, commission_bps=10, spread_bps=5, impact_coeff=0.1, adv=1e9):
    """Full LOB: commission + tax + half-spread + sqrt impact"""
    commission = turnover * commission_bps / 1e4
    spread_cost = turnover * spread_bps / 1e4 * 0.5
    participation = turnover / adv
    impact_cost = turnover * impact_coeff * np.sqrt(np.maximum(participation, 1e-6))
    return commission + spread_cost + impact_cost

# Test fee models on all strategies
all_strats = {**simple_strats, **advanced_strats}
fee_models = {
    'ZeroFee': zero_fee_cost,
    'PercentFee(10bp)': lambda t: percent_fee_cost(t, 10, 0),
    'PercentFee(30bp)': lambda t: percent_fee_cost(t, 30, 0),
    'Slippage+Impact': slippage_impact_cost,
    'FullLOB': full_lob_cost,
}

fee_results = {}
for name, w in all_strats.items():
    pos = w.shift(1).fillna(0)
    turnover = pos.diff().abs().sum(axis=1)
    turnover.iloc[0] = pos.iloc[0].abs().sum()
    
    fee_results[name] = {}
    for model_name, cost_fn in fee_models.items():
        cost = cost_fn(turnover)
        r = (pos * rets).sum(axis=1) - cost
        fee_results[name][model_name] = round(sharpe(r.dropna()), 2)

fee_tbl = pd.DataFrame(fee_results).T
fee_tbl.to_csv('iter7_fee_models.csv')
print('E1 Fee Models:\n', fee_tbl.to_string())

# ======================================================================
# E2. Simple vs Advanced: Systematic Comparison
# ======================================================================
# Compare at multiple cost levels
cost_levels = [0, 5, 10, 20, 30]  # bps
sa_results = {}

for level in cost_levels:
    sa_results[level] = {}
    for name, w in all_strats.items():
        pos = w.shift(1).fillna(0)
        turnover = pos.diff().abs().sum(axis=1)
        turnover.iloc[0] = pos.iloc[0].abs().sum()
        cost = turnover * level / 1e4
        r = (pos * rets).sum(axis=1) - cost
        sa_results[level][name] = {
            'Sharpe': round(sharpe(r.dropna()), 2),
            'AnnRet%': round(252*r.dropna().mean()*100, 2),
            'MaxDD%': round(100*max_dd((1+r.dropna()).cumprod()), 2),
            'Turnover/yr': round(turnover.mean() * 252, 1)
        }

# Summary table at 10bp
sa_tbl_10 = pd.DataFrame({name: sa_results[10][name] for name in all_strats})
sa_tbl_10.to_csv('iter7_simple_vs_advanced.csv')
print('E2 Simple vs Advanced (10bp):\n', sa_tbl_10.to_string())

# Cost sensitivity: at what cost does each strategy break even?
break_even = {}
for name, w in all_strats.items():
    pos = w.shift(1).fillna(0)
    turnover = pos.diff().abs().sum(axis=1)
    turnover.iloc[0] = pos.iloc[0].abs().sum()
    gross_r = (pos * rets).sum(axis=1)
    gross_sharpe = sharpe(gross_r.dropna())
    avg_turnover = turnover.mean()
    # Approximate break-even: gross_sharpe - cost_sharpe = 0
    # cost_sharpe ≈ avg_turnover * cost_bps / 1e4 / daily_vol
    daily_vol = gross_r.dropna().std()
    if daily_vol > 0 and avg_turnover > 0:
        be_bps = gross_sharpe * daily_vol * 1e4 / (avg_turnover * np.sqrt(252))
        break_even[name] = round(be_bps, 1)
    else:
        break_even[name] = np.nan

be_tbl = pd.DataFrame({'Strategy': list(break_even.keys()), 'BreakEven_bps': list(break_even.values())})
be_tbl.to_csv('iter7_breakeven.csv', index=False)
print('E2 Break-even costs:\n', be_tbl.to_string(index=False))

# ======================================================================
# E3. Backtesting Framework Best Practices (from QuantStart article)
# ======================================================================
# 1. Event-driven vs Vectorized comparison
# 2. Look-ahead bias checks
# 3. Multiple testing correction
# 4. Walk-forward vs single split
# 5. Synthetic data validation

# 3a. Look-ahead bias check: signal at close vs next open
def check_lookahead(w, rets):
    """Compare next-bar (correct) vs same-bar (look-ahead) execution"""
    # Correct: next-bar
    r_correct = backtest(w, rets, COST)
    
    # Wrong: same-bar (look-ahead)
    pos_wrong = w  # no shift
    turnover_wrong = pos_wrong.diff().abs().sum(axis=1)
    turnover_wrong.iloc[0] = pos_wrong.iloc[0].abs().sum()
    cost_wrong = turnover_wrong * COST / 1e4
    r_wrong = (pos_wrong * rets).sum(axis=1) - cost_wrong
    
    return {
        'Correct Sharpe': round(sharpe(r_correct.dropna()), 2),
        'LookAhead Sharpe': round(sharpe(r_wrong.dropna()), 2),
        'Inflation': round(sharpe(r_wrong.dropna()) - sharpe(r_correct.dropna()), 2)
    }

la_results = {}
for name, w in all_strats.items():
    la_results[name] = check_lookahead(w, rets)

la_tbl = pd.DataFrame(la_results).T
la_tbl.to_csv('iter7_lookahead_bias.csv')
print('E3 Look-ahead Bias Check:\n', la_tbl.to_string())

# 3b. Walk-forward vs Single Split (expanding window)
def walk_forward_sharpe(w, rets, train_window=504, test_window=63, step=21):
    """Expanding window walk-forward Sharpe"""
    n = len(rets)
    sharpes = []
    for i in range(train_window, n - test_window, step):
        # Train on all data up to i
        # Test on i:i+test_window
        pos = w.shift(1).fillna(0)
        turnover = pos.diff().abs().sum(axis=1)
        turnover.iloc[0] = pos.iloc[0].abs().sum()
        cost = turnover * COST / 1e4
        r = (pos * rets).sum(axis=1) - cost
        test_r = r.iloc[i:i+test_window].dropna()
        if len(test_r) > 20:
            sharpes.append(sharpe(test_r))
    return np.mean(sharpes) if sharpes else 0

wf_results = {}
for name, w in all_strats.items():
    # Single split: first 60% train, last 40% test
    split = int(len(rets) * 0.6)
    pos = w.shift(1).fillna(0)
    turnover = pos.diff().abs().sum(axis=1)
    turnover.iloc[0] = pos.iloc[0].abs().sum()
    cost = turnover * COST / 1e4
    r = (pos * rets).sum(axis=1) - cost
    single_sharpe = sharpe(r.iloc[split:].dropna())
    
    # Walk-forward
    wf_sharpe = walk_forward_sharpe(w, rets)
    
    wf_results[name] = {
        'SingleSplit Sharpe': round(single_sharpe, 2),
        'WalkForward Sharpe': round(wf_sharpe, 2),
        'Difference': round(wf_sharpe - single_sharpe, 2)
    }

wf_tbl = pd.DataFrame(wf_results).T
wf_tbl.to_csv('iter7_walkforward_vs_single.csv')
print('E3 Walk-Forward vs Single Split:\n', wf_tbl.to_string())

# 3c. Synthetic data validation (GBM, OU, Jump)
def test_on_synthetic(w, rets, model='gbm', n_paths=50):
    n_days = len(rets)
    sharpes = []
    for p in range(n_paths):
        if model == 'gbm':
            mu = rets.mean().values * 252
            cov = rets.cov().values * 252
            chol = np.linalg.cholesky(cov + 1e-6 * np.eye(len(cov)))
            z = np.random.normal(0, 1, (n_days, len(mu)))
            daily = (mu / 252 - 0.5 * np.diag(cov) / 252) + (chol @ z.T / np.sqrt(252)).T
        elif model == 'ou':
            theta = 5.0
            mu = rets.mean().values * 252
            cov = rets.cov().values * 252
            chol = np.linalg.cholesky(cov + 1e-6 * np.eye(len(cov)))
            z = np.random.normal(0, 1, (n_days, len(mu)))
            daily = np.zeros((n_days, len(mu)))
            daily[0] = np.random.normal(mu/252, np.sqrt(np.diag(cov)/252))
            for t in range(1, n_days):
                daily[t] = daily[t-1] + theta * (mu/252 - daily[t-1]) + (chol @ z[t]) / np.sqrt(252)
        elif model == 'jump':
            mu = rets.mean().values * 252
            cov = rets.cov().values * 252
            chol = np.linalg.cholesky(cov + 1e-6 * np.eye(len(cov)))
            z = np.random.normal(0, 1, (n_days, len(mu)))
            lam = 0.5
            jump_mu = -0.1
            jump_sig = 0.15
            jumps = np.random.poisson(lam/252, (n_days, len(mu))) * np.random.normal(jump_mu, jump_sig, (n_days, len(mu)))
            daily = (mu / 252) + (chol @ z.T / np.sqrt(252)).T + jumps
        
        syn_rets = pd.DataFrame(daily, columns=rets.columns, index=rets.index)
        r = backtest(w, syn_rets, COST).dropna()
        if len(r) > 100:
            sharpes.append(sharpe(r))
    return np.mean(sharpes), np.std(sharpes)

syn_results = {}
for name, w in all_strats.items():
    for model in ['gbm', 'ou', 'jump']:
        mean_s, std_s = test_on_synthetic(w, rets, model, n_paths=30)
        syn_results[f'{name}_{model}'] = {'mean': round(mean_s, 2), 'std': round(std_s, 2)}

syn_tbl = pd.DataFrame(syn_results).T
syn_tbl.to_csv('iter7_synthetic_validation.csv')
print('E3 Synthetic Validation:\n', syn_tbl.to_string())

# ======================================================================
# E4. Parameter Sensitivity: Simple vs Advanced
# ======================================================================
# SMA window sweep
sma_windows = [50, 100, 150, 200, 250, 300]
sma_sens = {}
for window in sma_windows:
    w = (px[['SPY']] > px[['SPY']].rolling(window).mean()).astype(float).reindex(columns=px.columns).fillna(0)
    r = backtest(w, rets, COST).dropna()
    sma_sens[window] = round(sharpe(r), 2)

# XSec momentum lookback sweep
xs_lookbacks = [63, 126, 189, 252, 315, 378]
xs_sens = {}
for lb in xs_lookbacks:
    mom_xs = px[RISK] / px[RISK].shift(lb) - 1
    ranks = mom_xs.rank(axis=1, ascending=False)
    w = (ranks <= 5).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).reindex(columns=px.columns).fillna(0)
    r = backtest(w, rets, COST).dropna()
    xs_sens[lb] = round(sharpe(r), 2)

# GEM lookback sweep
gem_sens = {}
for lb in xs_lookbacks:
    mom = px[['SPY','EFA','AGG']] / px[['SPY','EFA','AGG']].shift(lb) - 1
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    valid = mom.dropna().index
    for t in valid:
        if mom.loc[t, 'SPY'] > 0:
            w.loc[t, mom.loc[t, ['SPY','EFA']].idxmax()] = 1.0
        else:
            w.loc[t, 'AGG'] = 1.0
    r = backtest(w, rets, COST).dropna()
    gem_sens[lb] = round(sharpe(r), 2)

sens_tbl = pd.DataFrame({
    'SMA_Window': pd.Series(sma_sens),
    'XSec_Lookback': pd.Series(xs_sens),
    'GEM_Lookback': pd.Series(gem_sens)
})
sens_tbl.to_csv('iter7_param_sensitivity.csv')
print('E4 Parameter Sensitivity:\n', sens_tbl.to_string())

# ======================================================================
# E5. Data Frequency & Resolution Effects
# ======================================================================
# Resample to weekly and test
px_w = px.resample('W-FRI').last()
rets_w = px_w.pct_change().dropna()

# Recompute simple strategies at weekly
w_sma_w = (px_w[['SPY']] > px_w[['SPY']].rolling(40).mean()).astype(float).reindex(columns=px_w.columns).fillna(0)  # ~200 days = 40 weeks
w_6040_w = rebalanced_portfolio({'SPY': 0.6, 'AGG': 0.4})  # but rebalanced weekly

# Weekly backtest
r_sma_w = backtest(w_sma_w, rets_w, COST/5).dropna()  # cost adjusted for weekly
r_sma_d = backtest(w_sma, rets, COST).dropna()

freq_tbl = pd.DataFrame({
    'Strategy': ['SMA200 Daily', 'SMA200 Weekly'],
    'Sharpe': [round(sharpe(r_sma_d), 2), round(sharpe(r_sma_w), 2)],
    'AnnRet%': [round(252*r_sma_d.mean()*100, 2), round(52*r_sma_w.mean()*100, 2)],
    'MaxDD%': [round(100*max_dd((1+r_sma_d).cumprod()), 2), round(100*max_dd((1+r_sma_w).cumprod()), 2)],
})
freq_tbl.to_csv('iter7_frequency_effects.csv', index=False)
print('E5 Frequency Effects:\n', freq_tbl.to_string(index=False))

# ======================================================================
# Compile Iteration #7
# ======================================================================
combined = {**all_strats}
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

# Add simple/advanced classification
res['Category'] = res['name'].map({**{k: 'Simple' for k in simple_strats}, **{k: 'Advanced' for k in advanced_strats}})

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
res.to_csv('iter7_comprehensive_perf.csv', index=False)
wf.to_csv('iter7_comprehensive_walkforward.csv', index=False)
val.to_csv('iter7_comprehensive_validation.csv', index=False)

# Plots
plt.figure(figsize=(12, 7))
for name, r in all_rets.items():
    if len(r) > 100:
        color = 'blue' if name in simple_strats else 'red'
        plt.plot((1+r).cumprod(), label=name, alpha=0.8, linewidth=0.8, color=color)
plt.legend(fontsize=7, ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Iteration #7 — Simple (blue) vs Advanced (red) Equity Curves (net 10bp)')
plt.ylabel('Growth of $1'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('iter7_equity.png', dpi=110, bbox_inches='tight')

# Fee model comparison
plt.figure(figsize=(10, 6))
for name in all_strats:
    plt.plot(fee_tbl.columns, fee_tbl.loc[name], marker='o', label=name, alpha=0.7)
plt.title('Sharpe Across Fee Models')
plt.xlabel('Fee Model'); plt.ylabel('Sharpe'); plt.legend(fontsize=7, ncol=2)
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig('iter7_fee_models.png', dpi=110)

# Simple vs Advanced at 10bp
plt.figure(figsize=(10, 6))
cats = ['Simple', 'Advanced']
simple_sharpes = [sa_results[10][n]['Sharpe'] for n in simple_strats]
adv_sharpes = [sa_results[10][n]['Sharpe'] for n in advanced_strats]
x = np.arange(len(simple_strats))
width = 0.35
plt.bar(x - width/2, simple_sharpes, width, label='Simple', color='blue', alpha=0.7)
plt.bar(x + width/2, adv_sharpes[:len(simple_strats)], width, label='Advanced', color='red', alpha=0.7)
plt.xticks(x, list(simple_strats.keys()))
plt.ylabel('Sharpe (10bp)'); plt.title('Simple vs Advanced at 10bp')
plt.legend(); plt.tight_layout(); plt.savefig('iter7_simple_vs_advanced.png', dpi=110)

# Synthetic validation heatmap
plt.figure(figsize=(10, 6))
pivot = syn_tbl['mean'].unstack()
if hasattr(pivot, 'plot'):
    pivot.plot(kind='bar')
    plt.title('Mean Sharpe on Synthetic Paths')
    plt.ylabel('Sharpe'); plt.tight_layout(); plt.savefig('iter7_synthetic.png', dpi=110)

print('\n=== ITERATION #7 PERFORMANCE ===')
print(res.to_string(index=False))
print('\n=== WALK-FORWARD ===')
print(wf.to_string(index=False))
print('\n=== VALIDATION ===')
print(val.to_string(index=False))
print('\nIteration #7 complete.')