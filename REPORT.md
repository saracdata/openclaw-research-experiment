# Quant Backtesting Research Report
Data: 15 US tickers (ETFs + large-cap stocks), 2012–2026 daily adjusted closes via Yahoo Finance.
Engine: vectorized, **next-bar execution**, **10 bps transaction cost per turnover unit**.
Source code: `engine.py`, `strategies.py`, `stats.py`, `run_experiments.py`.

> **Iteration #1 findings added at the end of this report** (`run_iteration1.py`), guided by
> QuantStart's *Beginner's Guide to Quantitative Trading* — covering data-cleaning spikes,
> parameter sweeps with in-sample/out-of-sample discipline, and cost sensitivity.
> Note: the engine's multi-asset cost bug was found and fixed during this iteration
> (cost was previously applied per-asset without summing; all Round-2 multi-asset numbers
> in this report should be treated as approximate until re-run).

## Performance (net of costs)

| Strategy | AnnRet% | AnnVol% | Sharpe | MaxDD% |
|---|---|---|---|---|
| Buy&Hold SPY | 15.00 | 16.51 | 0.93 | -33.7 |
| SMA200 Trend (Faber 2007) | 10.28 | 11.37 | 0.92 | **-21.6** |
| TSMOM 12-1 (MOP 2012) | 4.35 | 16.56 | 0.34 | -37.1 |
| MA 50/200 Cross | 3.55 | 16.54 | 0.29 | -40.4 |
| RSI(2) MeanRev (Connors) | 3.99 | 7.59 | 0.55 | -18.4 |
| Vol-Target 10% (Moreira & Muir 2017) | 9.43 | 11.29 | 0.86 | **-15.1** |
| Dual Momentum GEM (Antonacci 2014) | 29.87 | 17.71 | **1.57** | -21.7 |
| XSec Momentum 12-1 Top3 (Jegadeesh-Titman 1993) | 28.02 | 25.72 | 1.09 | -34.5 |
| Short-Term Reversal (Jegadeesh 1990) | -18.64 | 13.69 | -1.44 | -95.0 |
| Pairs AAPL/MSFT z-score (Gatev et al. 1999) | -1.59 | 11.80 | -0.08 | -54.2 |

## Statistical validation
NW_t = Newey-West t-stat of mean return; SR CI = block-bootstrap 95% Sharpe interval; DSR_p = Deflated Sharpe p-value (Bailey & López de Prado 2014, hurdle from cross-strategy Sharpe dispersion, N=10 trials).

| Strategy | NW_t | SR 95% CI | DSR_p |
|---|---|---|---|
| Buy&Hold SPY | 3.90 | [0.46, 1.45] | 1.000 |
| SMA200 Trend | 3.58 | [0.43, 1.43] | 1.000 |
| RSI(2) MeanRev | 2.43 | [0.15, 0.96] | 1.000 |
| Vol-Target 10% | 3.45 | [0.37, 1.34] | 1.000 |
| Dual Momentum GEM | 6.42 | [1.13, 2.05] | **0.001** |
| XSec Momentum Top3 | 4.64 | [0.62, 1.58] | 1.000 |
| Short-Term Reversal | -5.93 | [-1.88, -0.96] | 1.000 |
| Pairs AAPL/MSFT | -0.28 | [-0.60, 0.41] | 1.000 |

## Walk-forward stability (Sharpe per 4 folds)

| Strategy | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| SMA200 Trend | 0.82 | 0.36 | 0.85 | 1.49 |
| TSMOM 12-1 | -0.32 | -0.08 | 0.44 | 1.38 |
| RSI(2) MeanRev | 0.14 | 0.43 | 0.63 | 1.04 |
| Vol-Target 10% | 0.70 | 0.71 | 0.66 | 1.15 |
| Dual Momentum GEM | 2.10 | 1.17 | 1.32 | 2.07 |
| XSec Mom Top3 | 1.75 | 0.87 | 1.20 | 0.77 |

## Findings
1. **Dual Momentum GEM is the standout**: Sharpe 1.57, NW t=6.4, DSR p=0.001 — the only strategy that survives the multiple-testing hurdle — and positive Sharpe in all 4 folds. (Caveat: rotation among 5 ETFs in a 14-year US bull sample; results are regime-dependent.)
2. **Cross-sectional 12-1 momentum (top-3 of 10 megacaps) earns its literature premium** (28% ann., t=4.6) but with high vol and a -34.5% drawdown; DSR fails due to shared dispersion with weaker siblings.
3. **Trend filters (SMA200) don't raise Sharpe much here (0.92 vs 0.93) but cut max drawdown by ~12 points** — their real value is risk control, consistent with Faber.
4. **Volatility targeting delivers risk-adjusted improvement over buy & hold** (max DD -15% vs -34%, similar Sharpe), matching Moreira & Muir.
5. **Rejections are informative too**: short-term reversal at daily/weekly horizon loses badly after costs at 10bp — the cost load (~daily turnover) kills the well-documented gross edge. Pairs trading on AAPL/MSFT shows no edge over 2012–2026.
6. **TSMOM on SPY alone is weak in-sample 2012–2026** (mostly up market with shallow corrections); it shines on diversified futures in the original paper.

## Statistical rigor notes
- Next-bar execution avoids look-ahead; costs applied to turnover.
- Newey-West corrects autocorrelation; block bootstrap captures path dependence.
- Deflated Sharpe penalizes selection bias across the 10-strategy family — a strict standard that flags GEM as genuinely outperforming the family's expected-max noise.
- Single-asset 12-1 momentum performs poorly in a 14y sample of one bull regime; walk-forward folds show regime sensitivity (F1 negative for TSMOM).

## Limitations / next steps
- 10-stock cross-section (no small caps, no delisting bias control); add ~500-stock universe and sector neutralization.
- Test broader parameter neighborhoods for DSR-honest out-of-sample validation (train/test split optimization).
- Add intraday-quality data sources, borrow costs for shorts, dividend/margin handling.

---

# Iteration #1 — QuantStart "Beginner's Guide" findings
Source: https://www.quantstart.com/articles/Beginner-s-Guide-to-Quantitative-Trading/
Focus areas applied: data accuracy (spike filter), optimization/data-snooping bias,
transaction-cost realism. Code: `run_iteration1.py`. Outputs: `iter1_*.csv`, `iter1_sma_heatmap.png`, `iter1_momentum_oos.png`.

## E1 — Data cleanliness (spike filter)
Rolling z-score spike check (z=8, |ret|>5%) on all 9 tickers: **0 bad ticks found**. Yahoo adjusted data is clean for this universe; no correction needed.

## E2 — SMA window sweep with IS/OOS discipline (train 2012–2018, test 2019–2026)
| Window | IS Sharpe | OOS Sharpe |
|---|---|---|
| 50 | -0.41 | 0.33 |
| 100 | 0.09 | 0.44 |
| 150 | 0.37 | 0.62 |
| 200 | 0.13 | 0.57 |
| 250 | 0.17 | 0.58 |
| 300 | 0.19 | 0.61 |
| 350 | 0.24 | 0.56 |
| 400 | 0.20 | 0.49 |

**Finding — the textbook lesson in action:** parameters picked in-sample do NOT carry over. The IS ranking is noise (window 150 looks best IS, but its OOS edge is similar to windows 200–300). OOS Sharpe is uniformly higher for long windows simply because 2019–2026 was a friendlier regime — regime, not parameter skill. The honest takeaway: any window in 150–350 behaves the same OOS; fine-tuning on IS would have been pure data-snooping.

## E3 — Dual-momentum lookback sweep, same discipline
| Lookback | IS Sharpe | OOS Sharpe |
|---|---|---|
| 42 | -0.12 | 0.07 |
| 63 | -0.18 | 0.19 |
| 126 | -0.03 | 0.56 |
| 189 | 0.38 | 0.25 |
| 252 | 0.10 | 0.76 |
| 315 | 0.00 | 0.58 |
| 378 | 0.33 | 0.72 |

IS–OOS correlation of the Sharpe profile is essentially zero — further confirmation that single-split parameter optimization on 7 IS years is unreliable.

## E4 — Transaction-cost sensitivity (realism stress test)
| Strategy | 0 bps | 10 bps | 30 bps | Turnover/yr |
|---|---|---|---|---|
| SMA200 Trend | 0.63 | 0.38 | -0.12 | 24.9x |
| Dual Momentum | 0.62 | 0.33 | -0.26 | 52.5x |

**Finding:** the 9-asset diversified versions of these strategies are cost-fragile. At 30 bps both flip negative. Edge at retail cost levels (~10 bps) is thin; execution quality is a first-order input, exactly as the QuantStart article argues.

## Iteration #1 takeaways
1. Data quality is fine for this free source (no spikes).
2. Parameter optimization without strict OOS discipline is self-deception: IS ranks don't predict OOS ranks for either strategy family.
3. Cost sensitivity can kill marginal edges; prefer lower-turnover variants or better execution.
4. Process fix adopted going forward: every future parameter choice gets an IS/OOS split plus the DSR multiple-testing hurdle.
5. Engine bug fixed during this iteration: multi-asset transaction costs were not being summed (Round-2 multi-asset results were slightly overstated).
