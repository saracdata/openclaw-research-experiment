"""Iteration #1 — Comprehensive QuantStart-inspired experiments.
Incorporates ideas from QuantStart article archive:
- 60/40 benchmark portfolio (monthly rebalanced)
- All Weather Portfolio (Dalio risk-parity)
- Dual Momentum GEM (Antonacci exact rules)
- Simple vs Advanced strategy comparison
- Rebalance timing luck sensitivity
- Synthetic data stress tests
Outputs: iter1_*.csv, iter1_*.png, updated REPORT.md
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/root/quant')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import load, backtest, sharpe, max_dd, perf
from stats import nw_tstat
from scipy.stats import norm

COST = 10
TICKERS = ['SPY', 'AGG', 'TLT', 'IEI', 'GLD', 'GSG', 'VTI', 'EFA', 'EEM', 'IEF', 'VNQ', 'DBC', 'XLP', 'XLU', 'XLE', 'XLF', 'VIG', 'SCHD', 'MDY', 'XLK', 'XLV']
px = pd.DataFrame({t: load(t)['Close'] for t in TICKERS}).dropna()
rets = px.pct_change()

# ======================================================================
# E1. 60/40 Benchmark (monthly rebalanced) - QuantStart standard
# ======================================================================
def rebalanced_portfolio(weights_dict):
    """Monthly rebalanced portfolio — set weights at each month-end business day."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    total = sum(weights_dict.values())
    norm_w = {k: v/total for k, v in weights_dict.items()}
    # Get month-end dates that exist in the data
    # First, find the first date of each month
    first_of_month = px.groupby([px.index.year, px.index.month]).head(1).index
    # Last date of each month
    last_of_month = px.groupby([px.index.year, px.index.month]).tail(1).index
    for d in last_of_month:
        for k, v in norm_w.items():
            if k in w.columns:
                w.loc[d, k] = v
    # Forward fill from the FIRST date of the dataset
    first_date = px.index[0]
    for k, v in norm_w.items():
        if k in w.columns:
            w.loc[first_date, k] = v
    w = w.ffill().fillna(0)
    return w

# 60/40 US Equities/Bonds (SPY/AGG)
w_6040 = rebalanced_portfolio({'SPY': 0.6, 'AGG': 0.4})
r_6040 = backtest(w_6040, rets, COST)

# All Weather Portfolio (Dalio via QuantStart)
w_aw = rebalanced_portfolio({'VTI': 0.30, 'TLT': 0.40, 'IEI': 0.15, 'GLD': 0.075, 'GSG': 0.075})
r_aw = backtest(w_aw, rets, COST)

# Simple EW Risk Assets
w_ew = rebalanced_portfolio({t: 1.0 for t in ['SPY','EFA','EEM','VNQ','DBC','XLK']})
r_ew = backtest(w_ew, rets, COST)

# ======================================================================
# E2. Dual Momentum GEM — Antonacci's exact rules per QuantStart
# ======================================================================
# 126-day (6-month) momentum on: SPY (US equities), EFA (intl), AGG (bonds)
# Absolute momentum: if SPY mom > 0, hold top-1 of SPY/EFA/AGG; else hold AGG
mom = px[['SPY','EFA','AGG']] / px[['SPY','EFA','AGG']].shift(126) - 1
w_gem = pd.DataFrame(0.0, index=px.index, columns=px.columns)
valid = mom.dropna().index
for t in valid:
    spy_mom = mom.loc[t, 'SPY']
    if spy_mom > 0:
        best = mom.loc[t, ['SPY','EFA']].idxmax()  # relative strength
        w_gem.loc[t, best] = 1.0
    else:
        w_gem.loc[t, 'AGG'] = 1.0
r_gem = backtest(w_gem, rets, COST)

# ======================================================================
# E3. Simple vs Advanced comparison
# ======================================================================
# Simple: SMA200 on SPY only
w_sma = (px[['SPY']] > px[['SPY']].rolling(200).mean()).astype(float)
w_sma = w_sma.reindex(columns=px.columns).fillna(0)
r_sma = backtest(w_sma, rets, COST)

# Simple: RSI(2) on SPY
from strategies import rsi2_meanrev
w_rsi = rsi2_meanrev(px['SPY']).to_frame('SPY').reindex(columns=px.columns).fillna(0)
r_rsi = backtest(w_rsi, rets, COST)

# Advanced: Multi-asset TSMOM + Risk Parity (equal vol contribution)
mom12 = px / px.shift(252 + 21) - 1
signs = (mom12 > 0).astype(float) * 2 - 1
# Risk parity weights: inverse vol * signal
vol = rets.rolling(63).std() * np.sqrt(252)
inv_vol = 1.0 / vol.replace(0, np.nan)
w_rp = signs.mul(inv_vol).div(inv_vol.abs().sum(axis=1), axis=0).fillna(0)
r_rp = backtest(w_rp, rets, COST)

# Advanced: Dual Momentum + Vol Target (10%)
r_gem_vol = backtest(w_gem, rets, COST)
gem_vol = r_gem_vol.rolling(21).std() * np.sqrt(252)
lev = (0.10 / gem_vol).clip(upper=1.5).fillna(1.0).shift(1)
w_gem_vt = w_gem.mul(lev, axis=0)
r_gem_vt = backtest(w_gem_vt, rets, COST)

# ======================================================================
# E4. Rebalance Timing Luck (QuantStart: different month-end dates)
# ======================================================================
def rebalanced_portfolio_offset(weights_dict, day_of_month=-1):
    """Rebalance on specific day of month (1=1st, -1=last business day)."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    total = sum(weights_dict.values())
    norm_w = {k: v/total for k, v in weights_dict.items()}
    # Build rebalance dates
    rebal_dates = []
    for ym in pd.date_range(px.index[0], px.index[-1], freq='MS'):
        if day_of_month > 0:
            # day_of_month-th business day
            bd = pd.bdate_range(ym, periods=day_of_month)
            if len(bd) >= day_of_month:
                d = bd[day_of_month - 1]
                if d in px.index:
                    rebal_dates.append(d)
        else:
            # last business day of month
            eom = ym + pd.offsets.MonthEnd(0)
            bd = pd.bdate_range(eom - pd.offsets.BDay(2), eom)
            d = bd[-1]
            if d in px.index:
                rebal_dates.append(d)
    for d in rebal_dates:
        for k, v in norm_w.items():
            if k in w.columns:
                w.loc[d, k] = v
    w = w.ffill().fillna(0)
    return w

timing_results = []
for day in [1, 5, 10, 15, -1]:  # 1st, 5th, 10th, 15th, last business day
    w = rebalanced_portfolio_offset({'SPY': 0.6, 'AGG': 0.4}, day)
    r = backtest(w, rets, COST).dropna()
    timing_results.append({'RebalanceDay': day, 'Sharpe': round(sharpe(r), 2),
                           'AnnRet%': round(252*r.mean()*100, 2),
                           'MaxDD%': round(100*max_dd((1+r).cumprod()), 2)})
timing_tbl = pd.DataFrame(timing_results)
timing_tbl.to_csv('iter1_rebalance_timing.csv', index=False)
print('E4 Rebalance timing:', timing_tbl.to_string(index=False))

# ======================================================================
# E5. Synthetic data stress test (QuantStart: correlated time series)
# ======================================================================
def generate_correlated_returns(n_assets=5, n_days=2520, corr=0.3, seed=42):
    """Generate synthetic returns with specified correlation structure."""
    rng = np.random.default_rng(seed)
    # Factor model: 1 common factor + idiosyncratic
    factor = rng.normal(0, 0.01, n_days)
    idio = rng.normal(0, 0.01, (n_days, n_assets))
    R = np.sqrt(corr) * factor[:, None] + np.sqrt(1-corr) * idio
    return pd.DataFrame(R, columns=[f'SYN{i}' for i in range(n_assets)])

syn_rets = generate_correlated_returns(5, len(px), 0.3)
# Apply simple momentum on synthetic data
mom_s = syn_rets / syn_rets.shift(126) - 1
w_syn = pd.DataFrame(0.0, index=syn_rets.index, columns=syn_rets.columns)
valid = mom_s.dropna().index
best = mom_s.loc[valid].idxmax(axis=1)
for t in valid:
    w_syn.loc[t, best.loc[t]] = 1.0
r_syn = backtest(w_syn, syn_rets, COST).dropna()
syn_sharpe = sharpe(r_syn)
print(f'E5 Synthetic data Sharpe (momentum): {syn_sharpe:.2f}')

# ======================================================================
# Compile and validate all strategies
# ======================================================================
strategies = {
    '60/40 SPY/AGG (Monthly)': r_6040,
    'All Weather (Dalio)': r_aw,
    'EW Risk Assets': r_ew,
    'Dual Momentum GEM (Antonacci)': r_gem,
    'SMA200 Trend (SPY)': r_sma,
    'RSI(2) MeanRev (SPY)': r_rsi,
    'Multi-asset TSMOM + Risk Parity': r_rp,
    'GEM + Vol Target 10%': r_gem_vt,
}

results = []
all_returns = []
for name, r in strategies.items():
    r = r.dropna()
    all_returns.append(r)
    results.append(perf(r, name))

res = pd.DataFrame(results)
# Add Calmar
res['Calmar'] = res['AnnRet%'] / res['MaxDD%'].abs()

# Walk-forward folds
n = len(px)
fb = [(k * n // 4, (k + 1) * n // 4) for k in range(4)]
wf_rows = []
for name, r in strategies.items():
    r = r.dropna()
    shs = [round(sharpe(r.iloc[a:b]), 2) for a, b in fb]
    wf_rows.append({'Strategy': name, 'F1': shs[0], 'F2': shs[1], 'F3': shs[2], 'F4': shs[3]})
wf = pd.DataFrame(wf_rows)

# Statistical validation
sharpes = [sharpe(r) for r in all_returns]
sr_std = float(np.std(sharpes, ddof=1))
from stats import report_validation
vals = [report_validation(results[i]['name'], all_returns[i], len(strategies), sr_std)
        for i in range(len(strategies))]

# Save outputs
res.to_csv('iter1_comprehensive_perf.csv', index=False)
wf.to_csv('iter1_comprehensive_walkforward.csv', index=False)
pd.DataFrame(vals).to_csv('iter1_comprehensive_validation.csv', index=False)

# Plot equity curves
plt.figure(figsize=(11, 6))
for name, r in strategies.items():
    plt.plot((1+r.dropna()).cumprod(), label=name)
plt.legend(fontsize=8, ncol=2); plt.title('Iteration #1 — Equity Curves (net 10bp)')
plt.ylabel('Growth of $1'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('iter1_comprehensive_equity.png', dpi=110)

# Plot simple vs advanced Sharpe
simple = ['SMA200 Trend (SPY)', 'RSI(2) MeanRev (SPY)', '60/40 SPY/AGG (Monthly)']
advanced = ['Dual Momentum GEM (Antonacci)', 'Multi-asset TSMOM + Risk Parity', 'GEM + Vol Target 10%']
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].barh(simple, [res[res.name==s].Sharpe.values[0] for s in simple], color='steelblue')
ax[0].set_title('Simple Strategies'); ax[0].set_xlabel('Sharpe')
ax[1].barh(advanced, [res[res.name==s].Sharpe.values[0] for s in advanced], color='coral')
ax[1].set_title('Advanced Strategies'); ax[1].set_xlabel('Sharpe')
fig.suptitle('Simple vs Advanced (QuantStart theme)')
plt.tight_layout(); plt.savefig('iter1_simple_vs_advanced.png', dpi=110)

print('\n=== COMPREHENSIVE PERFORMANCE ===')
print(res.to_string(index=False))
print('\n=== WALK-FORWARD ===')
print(wf.to_string(index=False))
print('\n=== VALIDATION ===')
pd.DataFrame(vals).to_string(index=False)
print('\nIteration #1 comprehensive complete.')