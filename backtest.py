#!/usr/bin/env python3
"""
backtest.py — Complete quantitative backtesting script with scipy.optimize.

Implements:
  1. Data collection (Yahoo Finance via yfinance, CSV cache in ./data)
  2. Vectorized backtest engine (next-bar execution, transaction costs)
  3. Strategy set: buy & hold, SMA trend, time-series momentum, dual momentum
  4. scipy.optimize portfolio optimization: maximize Sharpe (SLSQP) with
     walk-forward re-optimization (no look-ahead: uses only trailing data)
  5. Statistical validation: Sharpe, max drawdown, Calmar, Newey-West t-stat,
     block-bootstrap confidence intervals, walk-forward fold Sharpes

Usage:
    python3 backtest.py                 # run full pipeline
    python3 backtest.py --tickers SPY QQQ TLT GLD
    python3 backtest.py --start 2015-01-01

Requires: pandas, numpy, scipy, yfinance
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

try:
    import yfinance as yf
except ImportError:
    yf = None

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_DIR = "data"
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EFA", "EEM", "IEF", "AGG"]
DEFAULT_START = "2012-01-01"
COST_BPS = 10          # transaction cost per unit turnover (basis points)
WF_FOLDS = 4           # walk-forward folds
RISK_FREE = 0.0        # annualized risk-free rate used in Sharpe
TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# 1. Data collection
# ----------------------------------------------------------------------------
def fetch(ticker: str, start: str) -> pd.DataFrame:
    """Download adjusted OHLCV for one ticker; cache to DATA_DIR/<ticker>.csv."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, skiprows=[1, 2], parse_dates=True)
        df.index = pd.to_datetime(df.index, format="mixed")
        df = df.astype(float)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    if yf is None:
        raise RuntimeError(f"No cache for {ticker} and yfinance is not installed.")
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if len(df) > 100:
                df.to_csv(path)
                return fetch(ticker, start)  # re-read through cache path
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Failed to download {ticker}")


def load_prices(tickers, start=DEFAULT_START) -> pd.DataFrame:
    """Close-price panel, columns=tickers, inner-joined dates, NaNs dropped."""
    px = pd.DataFrame({t: fetch(t, start)["Close"] for t in tickers})
    return px.dropna()


# ----------------------------------------------------------------------------
# 2. Backtest engine
# ----------------------------------------------------------------------------
def backtest(positions: pd.DataFrame, returns: pd.DataFrame,
             cost_bps: float = COST_BPS, lag: int = 1) -> pd.Series:
    """
    Vectorized backtest.

    positions : target weights per asset (rows=dates). Executed `lag` bars after
                the signal to avoid look-ahead bias.
    returns   : simple returns per asset.
    cost_bps  : proportional cost paid on each unit of turnover.

    Returns net daily strategy return series.
    """
    pos = positions.shift(lag).fillna(0.0)
    turnover = pos.diff().abs()
    turnover.iloc[0] = pos.iloc[0].abs()
    cost = turnover.sum(axis=1) * cost_bps / 1e4
    return (pos * returns).sum(axis=1) - cost


# ----------------------------------------------------------------------------
# 3. Performance metrics
# ----------------------------------------------------------------------------
def sharpe_ratio(returns: pd.Series, freq: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    excess = r.mean() - RISK_FREE / freq
    return float(np.sqrt(freq) * excess / r.std()) if r.std() > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline (negative fraction)."""
    cum = (1 + returns.fillna(0)).cumprod()
    return float(((cum - cum.cummax()) / cum.cummax()).min())


def performance(returns: pd.Series, label: str = "") -> dict:
    r = returns.dropna()
    cum = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    ann_ret = cum.iloc[-1] ** (1 / years) - 1 if years > 0 and cum.iloc[-1] > 0 else np.nan
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sr = sharpe_ratio(r)
    dd = max_drawdown(r)
    return {"Strategy": label, "AnnRet%": round(100 * ann_ret, 2),
            "AnnVol%": round(100 * ann_vol, 2), "Sharpe": round(sr, 2),
            "MaxDD%": round(100 * dd, 2),
            "Calmar": round(ann_ret / abs(dd), 2) if dd < 0 else np.nan}


def newey_west_tstat(returns: pd.Series, lags: int = 5):
    """Newey-West t-statistic that mean daily return is zero."""
    r = returns.dropna().values
    n = len(r)
    if n < 50 or np.std(r) == 0:
        return 0.0
    e = r - r.mean()
    s = np.sum(e * e) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * np.sum(e[l:] * e[:-l]) / n
    se = np.sqrt(s / n)
    return float(r.mean() / se) if se > 0 else 0.0


def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, block: int = 21,
                        seed: int = 42):
    """Circular block bootstrap 95% CI for annualized Sharpe ratio."""
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    n = len(r)
    n_blocks = n // block + 1
    starts = rng.integers(0, n, n_blocks * n_boot)
    out = []
    for b in range(n_boot):
        idx = []
        for s in starts[b * n_blocks:(b + 1) * n_blocks]:
            idx.extend(range(s, min(s + block, n)))
        sample = r[idx[:n]]
        if sample.std() > 0:
            out.append(np.sqrt(TRADING_DAYS) * sample.mean() / sample.std())
    return tuple(np.percentile(out, [2.5, 97.5]))


# ----------------------------------------------------------------------------
# 4. Rules-based strategies (each returns a weights DataFrame)
# ----------------------------------------------------------------------------
def w_buy_and_hold(px: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(1.0 / px.shape[1], index=px.index, columns=px.columns)
    return w


def w_sma_trend(px: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Equal-weight assets whose close is above their SMA; cash otherwise."""
    ok = px > px.rolling(window).mean()
    w = ok.astype(float)
    return w.div(w.sum(axis=1).replace(0, 1), axis=0).fillna(0)


def w_tsmom(px: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Long/short time-series momentum (Moskowitz, Ooi & Pedersen 2012)."""
    mom = px.shift(skip) / px.shift(lookback + skip) - 1
    signs = (mom > 0).astype(float) * 2 - 1
    w = signs.div(signs.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0)


def w_dual_momentum(px: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """Rotate 100% into the single highest-momentum asset (Antonacci 2014)."""
    mom = px / px.shift(lookback) - 1
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    valid = mom.dropna().index
    best = mom.loc[valid].idxmax(axis=1)
    for t in valid:
        w.loc[t, best.loc[t]] = 1.0
    return w


# ----------------------------------------------------------------------------
# 5. scipy.optimize: maximum-Sharpe weights with walk-forward re-optimization
# ----------------------------------------------------------------------------
def max_sharpe_weights(mean_vec: np.ndarray, cov: np.ndarray,
                       long_only: bool = True) -> np.ndarray:
    """
    Maximize (w'mu - rf) / sqrt(w'Cov w) subject to sum(w)=1 and bounds.
    Solved with scipy.optimize.minimize (SLSQP) on the negative Sharpe.
    """
    n = len(mean_vec)

    def neg_sharpe(w):
        ret = w @ mean_vec - RISK_FREE / TRADING_DAYS
        vol = np.sqrt(max(w @ cov @ w, 1e-12))
        return -ret / vol * np.sqrt(TRADING_DAYS)

    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.repeat(1.0 / n, n)
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                   constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, 1) if long_only else res.x
    s = w.sum()
    return w / s if s != 0 else w0


def w_optimized(px: pd.DataFrame, lookback: int = 252, window: int = 126,
                reopt_every: int = 21) -> pd.DataFrame:
    """
    Walk-forward optimized portfolio: re-fit max-Sharpe weights every
    `reopt_every` trading days using only the trailing `lookback` days
    of data (no look-ahead). Held constant between re-optimizations.
    """
    rets = px.pct_change()
    cols = list(px.columns)
    w = pd.DataFrame(0.0, index=px.index, columns=cols)
    start_idx = px.index[min(lookback, len(px) - 1)]
    dates = px.index[px.index >= start_idx]
    next_reopt = None
    current = np.repeat(1.0 / len(cols), len(cols))
    for t in dates:
        if next_reopt is None or t >= next_reopt:
            hist = rets.loc[:t].tail(lookback).dropna()
            if len(hist) >= 60 and hist.std().min() > 0:
                mu = hist.mean().values * TRADING_DAYS          # annualized
                cov = hist.cov().values * TRADING_DAYS
                # shrink covariance toward diagonal for numerical stability
                cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))
                current = max_sharpe_weights(mu, cov)
            next_reopt = t + pd.Timedelta(days=int(reopt_every * 1.4))
        w.loc[t] = current
    return w


# ----------------------------------------------------------------------------
# 6. Experiment runner
# ----------------------------------------------------------------------------
def walk_folds(index, n_folds=WF_FOLDS):
    n = len(index)
    edges = [k * n // n_folds for k in range(n_folds + 1)]
    return [(edges[k], edges[k + 1]) for k in range(n_folds)]


def run(tickers, start):
    os.makedirs(DATA_DIR, exist_ok=True)
    px = load_prices(tickers, start)
    rets = px.pct_change()
    print(f"Data: {px.shape[0]} days x {px.shape[1]} assets "
          f"({px.index[0].date()} - {px.index[-1].date()})\n")

    strategies = {
        "Buy&Hold Equal Weight": w_buy_and_hold(px),
        "SMA200 Trend": w_sma_trend(px),
        "TSMOM 12-1 L/S": w_tsmom(px),
        "Dual Momentum 6m": w_dual_momentum(px),
        "Optimized MaxSharpe (walk-forward)": w_optimized(px),
    }

    results, ret_series = [], {}
    for name, weights in strategies.items():
        r = backtest(weights, rets)
        ret_series[name] = r
        results.append(performance(r, name))
    table = pd.DataFrame(results)

    # Walk-forward fold Sharpes
    folds = walk_folds(px.index)
    wf = pd.DataFrame({
        name: [round(sharpe_ratio(r.dropna().iloc[a:b]), 2)
               for a, b in folds]
        for name, r in ret_series.items()
    }, index=[f"F{i+1}" for i in range(len(folds))]).T

    # Statistical validation
    val_rows = []
    for name, r in ret_series.items():
        lo, hi = bootstrap_sharpe_ci(r)
        val_rows.append({"Strategy": name,
                         "NW_t": round(newey_west_tstat(r), 2),
                         "SR_CI_lo": round(lo, 2), "SR_CI_hi": round(hi, 2),
                         "skew": round(float(stats.skew(r.dropna())), 2),
                         "kurt": round(float(stats.kurtosis(r.dropna(), fisher=False)), 2)})
    val = pd.DataFrame(val_rows).set_index("Strategy")

    print("=== PERFORMANCE (net of 10bp costs, next-bar execution) ===")
    print(table.to_string(index=False))
    print("\n=== WALK-FORWARD SHARPE PER FOLD ===")
    print(wf.to_string())
    print("\n=== STATISTICAL VALIDATION ===")
    print(val.to_string())

    # Save outputs
    table.to_csv("backtest_results.csv", index=False)
    val.to_csv("backtest_validation.csv")
    wf.to_csv("backtest_walkforward.csv")
    pd.DataFrame(ret_series).to_csv("backtest_returns.csv")
    print("\nSaved: backtest_results.csv, backtest_validation.csv, "
          "backtest_walkforward.csv, backtest_returns.csv")
    return table, val, wf, ret_series


def main():
    ap = argparse.ArgumentParser(description="Quant backtester with scipy.optimize")
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default=DEFAULT_START)
    args = ap.parse_args()
    run(args.tickers, args.start)


if __name__ == "__main__":
    main()
