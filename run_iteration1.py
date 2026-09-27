"""Iteration #1 — guided by QuantStart "Beginner's Guide to Quantitative Trading".

Experiments addressing the article's core backtesting-bias themes:
  E1. Data cleanliness: spike filter on raw OHLC (accuracy concern)
  E2. Parameter sweep for SMA trend (window 50..400) — with IN-SAMPLE vs
      OUT-OF-SAMPLE split (train 2012-2018, test 2019-2026) to expose
      optimization/data-snooping bias.
  E3. Momentum lookback sweep (63..378d) with same train/test discipline.
  E4. Transaction-cost sensitivity (0 / 10 / 30 bps) on the best strategies
      (execution-cost realism).
Outputs: iter1_*.csv, iter1_sma_heatmap.png, iter1_momentum_oos.png
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/root/quant')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import load, backtest, sharpe, max_dd
from stats import nw_tstat

COST = 10
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'AGG']
px = pd.DataFrame({t: load(t)['Close'] for t in TICKERS}).dropna()
rets = px.pct_change()

# ---------------------------------------------------------------------------
# E1. Spike filter (data accuracy check)
# ---------------------------------------------------------------------------
def spike_filter(s, z=8.0, window=50):
    """Flag returns beyond `z` rolling std (potential bad ticks)."""
    r = s.pct_change()
    mu = r.rolling(window, min_periods=20).mean()
    sd = r.rolling(window, min_periods=20).std()
    flags = ((r - mu).abs() > z * sd) & (r.abs() > 0.05)
    return flags

spikes = {t: int(spike_filter(px[t]).sum()) for t in TICKERS}
pd.DataFrame([{'Ticker': t, 'SpikeFlags': n} for t, n in spikes.items()]) \
    .to_csv('iter1_spike_check.csv', index=False)
print('E1 spike check:', spikes)

# Train/test split (no peeking: parameters chosen on train, judged on test)
train_end = '2019-01-01'
is_mask = px.index < train_end
oos_mask = px.index >= train_end

# ---------------------------------------------------------------------------
# E2. SMA window sweep with IS/OOS discipline
# ---------------------------------------------------------------------------
rows = []
for window in [50, 100, 150, 200, 250, 300, 350, 400]:
    ok = px > px.rolling(window).mean()
    w = ok.astype(float)
    w = w.div(w.sum(axis=1).replace(0, 1), axis=0).fillna(0)
    r = backtest(w, rets, COST)
    for label, mask in [('IS', is_mask), ('OOS', oos_mask)]:
        seg = r[mask].dropna()
        rows.append({'SMA_window': window, 'split': label,
                     'Sharpe': round(sharpe(seg), 2),
                     'AnnRet%': round(252 * seg.mean() * 100, 2),
                     'MaxDD%': round(100 * max_dd((1 + seg).cumprod()), 2),
                     'NW_t': round(nw_tstat(seg)[0], 2)})
sma_tbl = pd.DataFrame(rows)
sma_tbl.to_csv('iter1_sma_sweep.csv', index=False)
print('\nE2 SMA sweep:\n', sma_tbl.pivot(index='SMA_window', columns='split', values='Sharpe'))

# Heatmap: IS vs OOS Sharpe by window
piv = sma_tbl.pivot(index='SMA_window', columns='split', values='Sharpe')
plt.figure(figsize=(7, 5))
plt.imshow(piv.values, cmap='RdYlGn', aspect='auto')
plt.xticks([0, 1], ['In-sample (2012-18)', 'Out-of-sample (2019-26)'])
plt.yticks(range(len(piv)), piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        plt.text(j, i, piv.values[i, j], ha='center', va='center', fontsize=12)
plt.title('SMA Trend Strategy — Sharpe by window & split\n(optimization-bias check)')
plt.colorbar(label='Sharpe')
plt.tight_layout(); plt.savefig('iter1_sma_heatmap.png', dpi=110)

# ---------------------------------------------------------------------------
# E3. Momentum lookback sweep with IS/OOS discipline (dual-momentum rotation)
# ---------------------------------------------------------------------------
rows = []
for lb in [42, 63, 126, 189, 252, 315, 378]:
    mom = px / px.shift(lb) - 1
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    valid = mom.dropna().index
    best = mom.loc[valid].idxmax(axis=1)
    for t in valid:
        w.loc[t, best.loc[t]] = 1.0
    r = backtest(w, rets, COST)
    for label, mask in [('IS', is_mask), ('OOS', oos_mask)]:
        seg = r[mask].dropna()
        rows.append({'lookback': lb, 'split': label,
                     'Sharpe': round(sharpe(seg), 2),
                     'AnnRet%': round(252 * seg.mean() * 100, 2),
                     'MaxDD%': round(100 * max_dd((1 + seg).cumprod()), 2),
                     'NW_t': round(nw_tstat(seg)[0], 2)})
mom_tbl = pd.DataFrame(rows)
mom_tbl.to_csv('iter1_momentum_sweep.csv', index=False)
print('\nE3 Momentum sweep:\n', mom_tbl.pivot(index='lookback', columns='split', values='Sharpe'))

piv2 = mom_tbl.pivot(index='lookback', columns='split', values='Sharpe')
plt.figure(figsize=(7, 5))
plt.plot(piv2['IS'], 'o-', label='In-sample (2012-18)')
plt.plot(piv2['OOS'], 's-', label='Out-of-sample (2019-26)')
plt.axhline(0, color='gray', lw=.5)
plt.xlabel('Momentum lookback (days)'); plt.ylabel('Sharpe')
plt.title('Dual Momentum — IS vs OOS Sharpe by lookback')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('iter1_momentum_oos.png', dpi=110)

# ---------------------------------------------------------------------------
# E4. Transaction-cost sensitivity
# ---------------------------------------------------------------------------
def sma_trend_w(window=200):
    ok = px > px.rolling(window).mean()
    w = ok.astype(float)
    return w.div(w.sum(axis=1).replace(0, 1), axis=0).fillna(0)

def dual_mom_w(lb=126):
    mom = px / px.shift(lb) - 1
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    valid = mom.dropna().index
    best = mom.loc[valid].idxmax(axis=1)
    for t in valid:
        w.loc[t, best.loc[t]] = 1.0
    return w

rows = []
for name, w in [('SMA200 Trend', sma_trend_w()), ('Dual Momentum', dual_mom_w())]:
    for c in [0, 10, 30]:
        r = backtest(w, rets, c).dropna()
        rows.append({'Strategy': name, 'cost_bps': c,
                     'Sharpe': round(sharpe(r), 2),
                     'AnnRet%': round(252 * r.mean() * 100, 2),
                     'turnover/yr': round(w.diff().abs().sum().sum() / len(w) * 252, 1)})
cost_tbl = pd.DataFrame(rows)
cost_tbl.to_csv('iter1_cost_sensitivity.csv', index=False)
print('\nE4 Cost sensitivity:\n', cost_tbl.to_string(index=False))

print('\nIteration #1 complete: 4 CSVs + 2 PNGs written.')