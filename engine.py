"""Vectorized daily backtest engine with transaction costs and risk metrics."""
import numpy as np
import pandas as pd


def load(ticker):
    df = pd.read_csv(f'data/{ticker}.csv', index_col=0, skiprows=[1, 2], parse_dates=True)
    df.index = pd.to_datetime(df.index, format='mixed')
    df = df.astype(float)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def sharpe(ret, freq=252):
    r = ret.dropna()
    if r.std() == 0: return 0.0
    return np.sqrt(freq) * r.mean() / r.std()


def max_dd(cum):
    peak = cum.cummax()
    return ((cum - peak) / peak).min()


def backtest(position, returns, cost_bps=10, lag=1):
    """position: target weight series (may be +/-). Executed next bar after signal.
    Returns daily strategy return series net of turnover costs."""
    pos = position.shift(lag).fillna(0.0)
    turnover = pos.diff().abs()
    turnover.iloc[0] = pos.iloc[0].abs()
    if isinstance(turnover, pd.DataFrame):
        cost = turnover.sum(axis=1) * cost_bps / 1e4
    else:
        cost = turnover * cost_bps / 1e4
    strat = pos * returns
    if isinstance(strat, pd.DataFrame):
        strat = strat.sum(axis=1)
    return strat - cost


def perf(strat_ret, label=''):
    r = strat_ret.dropna()
    cum = (1 + r).cumprod()
    yrs = len(r) / 252
    ann_ret = cum.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0
    ann_vol = r.std() * np.sqrt(252)
    sh = sharpe(r)
    dd = max_dd(cum)
    return {'name': label, 'AnnRet%': round(100 * ann_ret, 2),
            'AnnVol%': round(100 * ann_vol, 2), 'Sharpe': round(sh, 2),
            'MaxDD%': round(100 * dd, 2), 'days': len(r)}


def drawdown_series(strat_ret):
    return (1 + strat_ret.fillna(0)).cumprod()
