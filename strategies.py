"""Literature-backed strategies. Each returns a target daily position weight."""
import numpy as np
import pandas as pd
from engine import load


# 1) Buy & hold baseline
def buy_and_hold(px):
    return pd.Series(1.0, index=px.index)


# 2) SMA 200 trend filter on price (Faber 2007 "A Quantitative Approach to Tactical Asset Allocation")
def sma_trend(px, window=200):
    sma = px.rolling(window).mean()
    return (px > sma).astype(float)


# 3) Time-series momentum / 12-1 month (Moskowitz, Ooi, Pedersen 2012)
def tsmom(px, lookback=252, skip=21):
    mom = px.shift(skip) / px.shift(lookback + skip) - 1
    return (mom > 0).astype(float) * 2 - 1


# 4) Cross-sectional momentum, 12-1, top-3 of 8 (Jegadeesh & Titman 1993)
def xsec_momentum(price_df, lookback=252, skip=21, top_n=3):
    mom = price_df.shift(skip) / price_df.shift(lookback + skip) - 1
    ranks = mom.rank(axis=1, ascending=False)
    pos = (ranks <= top_n).astype(float)
    weights = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return weights


# 5) RSI(2) mean reversion on SPY (Connors style: buy RSI<10, exit RSI>60 or 5d)
def rsi2_meanrev(px, period=2, buy=10, exit_=60):
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    pos = pd.Series(0.0, index=px.index)
    pos[rsi < buy] = 1.0
    pos[rsi > exit_] = 0.0
    return pos.ffill().fillna(0)


# 6) Dual momentum GEM-style rotation (Antonacci 2014): SPY vs GLD vs TLT momentum
def dual_momentum(price_df, lookback=126):
    mom = price_df / price_df.shift(lookback) - 1
    valid = mom.dropna()
    best = valid.idxmax(axis=1)
    pos = pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
    pos.loc[valid.index, :] = 0.0
    for t in valid.index:
        pos.loc[t, best.loc[t]] = 1.0
    return pos


# 7) Pairs trading: cointegration z-score on AAPL/MSFT (Gatev, Goetzmann, Rouwenhorst 1999 style)
def pairs_zscore(px_a, px_b, window=60, entry=2.0):
    log_a, log_b = np.log(px_a), np.log(px_b)
    hedge = log_a.rolling(window).cov(log_b) / log_b.rolling(window).var()
    spread = log_a - hedge * log_b
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    pos = pd.Series(0.0, index=px_a.index)
    pos[z < -entry] = 1.0    # spread low: long A, short B
    pos[z > entry] = -1.0
    pos[z.abs() < 0.5] = 0.0  # unwind near equilibrium
    pos = pos.ffill().fillna(0)
    return pos  # weight on A; B gets -pos*hedge


# 8) Volatility targeting on SPY (Moreira & Muir 2017 "Volatility-Managed Portfolios")
def vol_target(px, target=0.10, window=21, lev_cap=2.0):
    ret = px.pct_change()
    vol = ret.rolling(window).std() * np.sqrt(252)
    lev = (target / vol).clip(upper=lev_cap).fillna(1.0)
    return lev


# 9) Moving-average crossover 50/200 (classic trend)
def ma_crossover(px, fast=50, slow=200):
    return (px.rolling(fast).mean() > px.rolling(slow).mean()).astype(float) * 2 - 1


# 10) Short-term reversal, weekly (Jegadeesh 1990): short last-week winners, hold 5d
def short_term_reversal(price_df, lookback=5):
    ret = price_df / price_df.shift(lookback) - 1
    mean = ret.mean(axis=1)
    pos = ret.apply(lambda col: -(col - mean), axis=0)  # demeaned negative momentum
    pos = pos.div(pos.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return pos
