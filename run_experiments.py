"""Run all strategy experiments and statistical validation."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/root/quant')
from engine import load, backtest, perf, sharpe
from strategies import (buy_and_hold, sma_trend, tsmom, xsec_momentum, rsi2_meanrev,
                        dual_momentum, pairs_zscore, vol_target, ma_crossover,
                        short_term_reversal)
from stats import report_validation

COST = 10  # bps per turnover unit
etf_px = {t: load(t)['Close'] for t in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']}
stock_px = {t: load(t)['Close'] for t in ['AAPL', 'MSFT', 'NVDA', 'JPM', 'XOM',
                                          'AMZN', 'GOOGL', 'META', 'BRK-B', 'UNH']}
etf = pd.DataFrame(etf_px).dropna()
stocks = pd.DataFrame(stock_px).dropna()
spy_ret = etf['SPY'].pct_change()

results, validations = [], []
N_TRIALS = 10  # number of strategy variants tested (used for DSR hurdle)

ALL_RETS = []

def add(position, name, rets=None):
    r = backtest(position, rets if rets is not None else spy_ret, COST)
    ALL_RETS.append(r)
    results.append(perf(r, name))
    return r

add(buy_and_hold(etf['SPY']), 'Buy&Hold SPY')
add(sma_trend(etf['SPY']), 'SMA200 Trend (SPY)')
add(tsmom(etf['SPY']), 'TSMOM 12-1 (SPY)')
add(ma_crossover(etf['SPY']), 'MA 50/200 Cross (SPY)')
add(rsi2_meanrev(etf['SPY']), 'RSI(2) MeanRev (SPY)')
add(vol_target(etf['SPY']), 'Vol-Target 10% (SPY)')
# Multi-asset dual momentum (own rets)
gem_pos = dual_momentum(etf)
gem_rets = etf.pct_change()
turn = gem_pos.diff().abs().fillna(gem_pos.abs().iloc[0])
r_gem = (gem_pos * gem_rets).sum(axis=1) - turn.sum(axis=1) * COST / 1e4
results.append(perf(r_gem, 'Dual Momentum GEM'))
ALL_RETS.append(r_gem)
# X-sectional momentum on stocks
mom_pos = xsec_momentum(stocks)
stock_rets = stocks.pct_change()
pos_d = mom_pos.shift(1).fillna(0)
turn = pos_d.diff().abs().fillna(pos_d.abs().iloc[0])
r_mom = (pos_d * stock_rets).sum(axis=1) - turn.sum(axis=1) * COST / 1e4
results.append(perf(r_mom, 'XSec Mom 12-1 Top3'))
# Short-term reversal
rev_pos = short_term_reversal(stocks)
pos_d = rev_pos.shift(1).fillna(0)
turn = pos_d.diff().abs().fillna(pos_d.abs().iloc[0])
r_rev = (pos_d * stock_rets).sum(axis=1) - turn.sum(axis=1) * COST / 1e4
results.append(perf(r_rev, 'Short-Term Reversal'))
# Pairs AAPL/MSFT
p = pairs_zscore(stocks['AAPL'], stocks['MSFT'])
hedge = p * 1.0
pr = stocks.pct_change()
pos_a = p.shift(1).fillna(0)
pos_b = -pos_a  # unit hedge approx
turn = (pos_a.diff().abs() + pos_b.diff().abs()).fillna(0)
r_pair = pos_a * pr['AAPL'] + pos_b * pr['MSFT'] - turn * COST / 1e4
results.append(perf(r_pair, 'Pairs AAPL/MSFT z-score'))

ALL_RETS += [r_mom, r_rev, r_pair]
all_rets = pd.concat(ALL_RETS, axis=1)
sharpes = [np.sqrt(252) * r.dropna().mean() / r.dropna().std() for r in ALL_RETS]
sr_std = float(np.std(sharpes, ddof=1))
validations = [report_validation(results[i]['name'], ALL_RETS[i], N_TRIALS, sr_std)
               for i in range(len(ALL_RETS))]

res = pd.DataFrame(results)
val = pd.DataFrame(validations)
print('\n=== PERFORMANCE (net of 10bp costs, next-bar execution) ===')
print(res.to_string(index=False))
print('\n=== STATISTICAL VALIDATION ===')
print(val.to_string(index=False))

# Walk-forward stability: 4 equal folds, Sharpe per fold per strategy
print('\n=== WALK-FORWARD STABILITY (Sharpe per fold) ===')
n = len(spy_ret)
fold_bounds = [(k * n // 5, (k + 1) * n // 5) for k in range(5)]
ret_map = {'SMA200 Trend (SPY)': ALL_RETS[1], 'TSMOM 12-1 (SPY)': ALL_RETS[2],
           'RSI(2) MeanRev (SPY)': ALL_RETS[4], 'Vol-Target 10% (SPY)': ALL_RETS[5],
           'Dual Momentum GEM': ALL_RETS[6], 'XSec Mom 12-1 Top3': r_mom}
wf_rows = []
for nm, r in ret_map.items():
    r = r.dropna()
    shs = []
    for k in range(4):
        seg = r.iloc[fold_bounds[k][1]:fold_bounds[k + 1][1]]
        shs.append(round(sharpe(seg), 2))
    wf_rows.append({'Strategy': nm, 'F1': shs[0], 'F2': shs[1], 'F3': shs[2], 'F4': shs[3]})
wf = pd.DataFrame(wf_rows)
print(wf.to_string(index=False))
res.to_csv('/root/quant/results_perf.csv')
val.to_csv('/root/quant/results_stats.csv', index=False)
wf.to_csv('/root/quant/results_walkforward.csv', index=False)
print('saved results_perf.csv, results_stats.csv, results_walkforward.csv')
