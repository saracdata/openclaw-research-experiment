"""Statistical validation: bootstrap CIs, deflated Sharpe ratio, walk-forward, 
Newey-West t-stats, and multiple-testing awareness (Bailey & Lopez de Prado 2014)."""
import numpy as np
import pandas as pd
from scipy import stats


def nw_tstat(ret, lags=5):
    """Newey-West t-stat of mean return != 0."""
    r = ret.dropna().values
    n = len(r)
    if n < 50 or np.std(r) == 0: return 0.0, n
    e = r - r.mean()
    g0 = np.sum(e * e) / n
    s = g0
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        s += 2 * w * np.sum(e[l:] * e[:-l]) / n
    se = np.sqrt(s / n)
    return (r.mean() / se if se > 0 else 0.0), n


def circular_bootstrap_ci(ret, n_boot=2000, seed=42):
    """Stationary-ish circular block bootstrap CI for annualized Sharpe."""
    rng = np.random.default_rng(seed)
    r = ret.dropna().values
    n = len(r)
    block = 21
    n_blocks = n // block + 1
    starts = rng.integers(0, n, n_blocks * n_boot)
    sharpes = []
    for b in range(n_boot):
        idx = []
        for s in starts[b * n_blocks:(b + 1) * n_blocks]:
            idx += list(range(s, min(s + block, n)))
        sample = r[idx[:n]]
        if sample.std() > 0:
            sharpes.append(np.sqrt(252) * sample.mean() / sample.std())
    return np.percentile(sharpes, [2.5, 97.5])


def dsr_test(sr, n_trials, T, sr_std, skew=None, kurt=None):
    """Deflated Sharpe Ratio p-value (Bailey & Lopez de Prado 2014).
    sr_std: cross-trial std of Sharpe estimates (empirical dispersion of the
    strategy family's Sharpes)."""
    if T < 30 or sr_std <= 0: return np.nan
    emc = 0.577215665
    N = max(n_trials, 2)
    e_max = np.sqrt(2 * np.log(N)) - (np.log(np.pi) + emc) / (2 * np.sqrt(2 * np.log(N)))
    sr0 = sr_std * e_max  # expected max Sharpe under null (annualized)
    g3 = skew if skew is not None else 0.0
    g4 = kurt if kurt is not None else 3.0
    denom = np.sqrt(max(1e-9, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    t = (sr - sr0) * np.sqrt(T) / denom
    return float(1 - stats.norm.cdf(t))


def walk_forward_split(index, n_splits=4):
    """Yield (train_end, test_start, test_end) expanding-window splits."""
    n = len(index)
    fold = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        yield k * fold, k * fold, min((k + 1) * fold, n)


def report_validation(name, ret, n_trials, sr_std):
    r = ret.dropna()
    t, n = nw_tstat(r)
    lo, hi = circular_bootstrap_ci(r)
    sr = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0.0
    dsr = dsr_test(sr, n_trials, n, sr_std,
                   skew=float(stats.skew(r)), kurt=float(stats.kurtosis(r, fisher=False)))
    yrs = n / 252
    return {'Strategy': name, 'NW_t': round(t, 2), 'SR_2.5%': round(lo, 2),
            'SR_97.5%': round(hi, 2), 'DSR_p': round(dsr, 4), 'years': round(yrs, 1)}
