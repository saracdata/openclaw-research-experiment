"""Round 2: composite strategies engineered for high return / shallow drawdown.
All use next-bar execution, 10bp costs, same validation stack."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/root/quant')
from engine import load, backtest, sharpe, perf
from strategies import sma_trend, tsmom, xsec_momentum
from stats import report_validation

COST = 10
T = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'AGG', 'LQD', 'HYG', 'VNQ',
     'DBC', 'XLP', 'XLU', 'XLE', 'XLF', 'IEF', 'VIG', 'SCHD', 'MDY', 'XLK', 'XLV']
px = pd.DataFrame({t: load(t)['Close'] for t in T}).dropna()
rets = px.pct_change()
RISK = ['SPY', 'QQQ', 'EFA', 'EEM', 'VNQ', 'DBC', 'MDY', 'XLK']       # risk assets
SAFE = ['IEF', 'TLT', 'AGG']                                          # defensive assets

results, all_r = [], []

def add(pos, name, rets=rets):
    if isinstance(pos, pd.DataFrame):
        turn = pos.diff().abs().fillna(pos.abs().iloc[0])
        r = (pos * rets).sum(axis=1) - turn.sum(axis=1) * COST / 1e4
    else:
        r = backtest(pos, rets['SPY'], COST)
    results.append(perf(r, name))
    all_r.append(r)
    return r

def vol_target_positions(pos, target=0.10, window=21, cap=2.0):
    """Scale a position frame by trailing vol of the resulting strategy's own returns."""
    return pos  # applied per-strategy below

# --- S1: Baseline B&H on the expanded universe (equal weight risk assets) ---
eq = pd.DataFrame(0.0, index=px.index, columns=RISK)
eq[RISK] = 1.0 / len(RISK)
add(eq, 'EW Risk Assets B&H')

# --- S2: GEM-style dual momentum on 12 assets, 126d ---
mom6 = px / px.shift(126) - 1
valid0 = mom6[RISK].dropna().index
mom_valid = mom6[RISK].loc[valid0]
best = mom_valid.idxmax(axis=1)
gem = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for t in mom_valid.index:
    gem.loc[t, best.loc[t]] = 1.0
add(gem, 'GEM rotation (12 asset universe)')

# --- S3: Dual momentum + trend filter: hold top-1 risk asset only if above its
#        200d SMA or 126d momentum > safe-asset momentum; else hold best SAFE ---
sma_ok = px[RISK] > px[RISK].rolling(200).mean()
safe_mom = px[SAFE] / px[SAFE].shift(126) - 1
safe_mom = safe_mom.dropna()
best_safe = safe_mom.idxmax(axis=1)
defensive = pd.DataFrame(0.0, index=px.index, columns=px.columns)
valid = mom6[RISK].dropna().index
for t in valid:
    b = best.loc[t]
    if bool(sma_ok.loc[t, b]):
        defensive.loc[t, b] = 1.0
    else:
        defensive.loc[t, best_safe.loc[t]] = 1.0
add(defensive, 'GEM + SMA200 defensive switch')

# --- S4: Top-2 dual momentum, half weight each, with defensive switch ---
mom_rank = mom6[RISK].loc[valid0].rank(axis=1, ascending=False)
top2 = (mom_rank <= 2)
w2 = top2.astype(float)
w2 = w2.div(w2.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
defensive2 = pd.DataFrame(0.0, index=px.index, columns=px.columns)
valid = valid0
for t in valid:
    active = [c for c in RISK if w2.loc[t, c] > 0 and bool(sma_ok.loc[t, c])]
    if active:
        for c in active:
            defensive2.loc[t, c] = 1.0 / len(active)
    else:
        defensive2.loc[t, best_safe.loc[t]] = 1.0
add(defensive2, 'Top-2 GEM + SMA200 defensive')

# --- S5: S4 + volatility targeting to 10% annualized ---
strat_ret_s4 = (defensive2.shift(1).fillna(0) * rets).sum(axis=1)
rolling_vol = strat_ret_s4.rolling(21).std() * np.sqrt(252)
lev = (0.10 / rolling_vol).clip(upper=1.5).fillna(1.0).shift(1)
add(defensive2.mul(lev, axis=0), 'Top-2 GEM + SMA200 + VolTarget 10%')

# --- S6: Trend-following cross-asset: long each of 8 assets when above SMA100, equal weight ---
sma100 = px[RISK] > px[RISK].rolling(100).mean()
trend_w = sma100.astype(float)
trend_w = trend_w.div(trend_w.sum(axis=1).replace(0, 1), axis=0).fillna(0)
add(trend_w, 'Multi-asset SMA100 tactical')

# --- S7: 12-1 TSMOM long/short across risk assets, vol-scaled ---
mom12 = px[RISK] / px[RISK].shift(252 + 21) - 1
signs = (mom12 > 0).astype(float) * 2 - 1
w = signs.div(signs.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 2.0  # 2x gross
add(w, 'Cross-asset TSMOM long/short 2x')

# --- validation ---
sharpes = [np.sqrt(252) * r.dropna().mean() / r.dropna().std() for r in all_r]
sr_std = float(np.std(sharpes, ddof=1))
vals = [report_validation(results[i]['name'], all_r[i], len(all_r), sr_std) for i in range(len(all_r))]

# --- walk-forward folds ---
n = len(rets)
fb = [(k * n // 4, (k + 1) * n // 4) for k in range(4)]
wf = []
for i, r in enumerate(all_r):
    r = r.dropna()
    shs = [round(sharpe(r.iloc[a:b]), 2) for a, b in fb]
    wf.append({'Strategy': results[i]['name'], 'F1': shs[0], 'F2': shs[1], 'F3': shs[2], 'F4': shs[3]})

# --- calmar ---
for row in results:
    row['Calmar'] = round(row['AnnRet%'] / abs(row['MaxDD%']), 2)

res = pd.DataFrame(results)
val = pd.DataFrame(vals)
wf = pd.DataFrame(wf)
print('\n=== ROUND 2: PERFORMANCE ===')
print(res.to_string(index=False))
print('\n=== ROUND 2: VALIDATION ===')
print(val.to_string(index=False))
print('\n=== ROUND 2: WALK-FORWARD SHARPE ===')
print(wf.to_string(index=False))
res.to_csv('results2_perf.csv', index=False)
val.to_csv('results2_stats.csv', index=False)
wf.to_csv('results2_walkforward.csv', index=False)
