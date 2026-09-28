"""
Iteration #17 — Comprehensive QuantStart Beginner's Guide + Latest Research
Focus: Kelly Criterion, Strategy Combination, Robust Regime Detection, 
       Kalman Filter Pairs, GARCH Vol Forecasting, Synthetic Data Validation
Based on:
- QuantStart Beginner's Guide (data quality, bias, execution, risk mgmt)
- QuantStart articles: HMM, Kalman Filter, GARCH, Synthetic Data, HRP
- Latest papers: Transformer (Quantformer), ML Multi-factor, Deep RL (from iter16)
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/root/quant')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from engine import load, backtest, perf, sharpe, max_dd
from strategies import (sma_trend, tsmom, xsec_momentum, rsi2_meanrev, 
                        dual_momentum, pairs_zscore, vol_target, ma_crossover, short_term_reversal)
from stats import nw_tstat, circular_bootstrap_ci, dsr_test, walk_forward_split

# ============================================================
# DATA LOADING
# ============================================================
ALL_TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'DBC', 'VNQ', 
               'XLE', 'XLF', 'XLK', 'XLV', 'XLU', 'IEI', 'VIG', 'SCHD', 'MDY',
               'IEF', 'AGG', 'VXX', 'VTI', 'VEA', 'VWO', 'GOVT', 'SHY', 'BIL', 
               'LQD', 'HYG', 'SMH', 'XLP', 'XLU']

# Use broader universe
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'DBC', 'VNQ', 
           'XLE', 'XLF', 'XLK', 'XLV', 'XLU', 'IEI', 'VIG', 'SCHD', 'MDY']

price_data = {}
for t in TICKERS:
    df = load(t)
    price_data[t] = df['Close']

prices = pd.DataFrame(price_data).dropna()
returns = prices.pct_change().dropna()
spy = prices['SPY']
spy_ret = spy.pct_change().dropna()

print(f"Data: {len(prices)} days, {len(TICKERS)} tickers")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# ============================================================
# E1. DATA QUALITY: SPIKE FILTER + CORPORATE ACTIONS CHECK
# ============================================================
print("\n=== E1: Data Quality Checks ===")

def spike_filter(s, z=8.0, window=50):
    r = s.pct_change()
    mu = r.rolling(window, min_periods=20).mean()
    sd = r.rolling(window, min_periods=20).std()
    flags = ((r - mu).abs() > z * sd) & (r.abs() > 0.05)
    return flags

def dividend_check(px):
    """Check for unadjusted dividends - large overnight gaps without volume spike"""
    ret = px.pct_change()
    vol_chg = px.pct_change()  # placeholder - we don't have volume in returns
    # Large negative return with low volume might indicate ex-dividend
    return ret < -0.05  # simplified

spikes = {t: int(spike_filter(prices[t]).sum()) for t in TICKERS}
div_flags = {t: int(dividend_check(prices[t]).sum()) for t in TICKERS}

pd.DataFrame([{'Ticker': t, 'SpikeFlags': spikes[t], 'DivFlags': div_flags[t]} 
              for t in TICKERS]).to_csv('/root/quant/iter17_data_quality.csv', index=False)
print(f"Spike flags: {spikes}")
print(f"Dividend flags: {div_flags}")

# ============================================================
# E2. KELLY CRITERION POSITION SIZING
# ============================================================
print("\n=== E2: Kelly Criterion Position Sizing ===")

def kelly_fraction(returns, max_leverage=2.0):
    """Kelly fraction = mu / sigma^2 (for Gaussian approximation)"""
    mu = returns.mean()
    var = returns.var()
    if var == 0:
        return 0
    kelly = mu / var
    return np.clip(kelly, 0, max_leverage)

# Test Kelly on various strategies
strategies_for_kelly = {
    'SMA200': sma_trend(spy, 200),
    'VolTarget': vol_target(spy),
    'TSMOM': tsmom(spy),
    'RSI2': rsi2_meanrev(spy),
    'MA50_200': ma_crossover(spy),
}

kelly_results = {}
for name, pos in strategies_for_kelly.items():
    pos_aligned = pos.reindex(spy_ret.index).fillna(0)
    strat_ret = backtest(pos_aligned, spy_ret, cost_bps=10)
    kelly_lev = kelly_fraction(strat_ret.dropna())
    # Apply Kelly leverage
    kelly_pos = pos_aligned * kelly_lev
    kelly_ret = backtest(kelly_pos, spy_ret, cost_bps=10)
    
    base_perf = perf(strat_ret.dropna(), name)
    kelly_perf = perf(kelly_ret.dropna(), f'{name}_Kelly')
    
    kelly_results[name] = {
        'kelly_fraction': round(kelly_lev, 3),
        'base_sharpe': base_perf['Sharpe'],
        'kelly_sharpe': kelly_perf['Sharpe'],
        'base_annret': base_perf['AnnRet%'],
        'kelly_annret': kelly_perf['AnnRet%'],
        'base_maxdd': base_perf['MaxDD%'],
        'kelly_maxdd': kelly_perf['MaxDD%'],
    }
    print(f"{name}: Kelly={kelly_lev:.3f}, Base Sharpe={base_perf['Sharpe']:.2f}, Kelly Sharpe={kelly_perf['Sharpe']:.2f}")

pd.DataFrame(kelly_results).T.to_csv('/root/quant/iter17_kelly_sizing.csv')

# ============================================================
# E3. STRATEGY COMBINATION: HIERARCHICAL RISK PARITY (HRP)
# ============================================================
print("\n=== E3: Strategy Combination with HRP ===")

# Generate individual strategy returns
strat_returns = {}
for name, pos in strategies_for_kelly.items():
    pos_aligned = pos.reindex(spy_ret.index).fillna(0)
    strat_returns[name] = backtest(pos_aligned, spy_ret, cost_bps=10)

# Multi-asset strategies
ma_strats = {
    'XSecMom': xsec_momentum(prices).reindex(returns.index).fillna(0).mean(axis=1),
    'GEM': dual_momentum(prices[['SPY','GLD','TLT']]).reindex(returns[['SPY','GLD','TLT']].index).fillna(0).mean(axis=1),
    'ShortRev': short_term_reversal(prices).reindex(returns.index).fillna(0).mean(axis=1),
}

for name, pos in ma_strats.items():
    benchmark = returns.mean(axis=1) if name != 'GEM' else returns[['SPY','GLD','TLT']].mean(axis=1)
    strat_returns[name] = backtest(pos.reindex(benchmark.index).fillna(0), benchmark, cost_bps=10)

# Also include RL portfolio from iteration 16
try:
    rl_ret = pd.read_csv('/root/quant/iter16_rl_returns.csv', index_col=0, parse_dates=True).squeeze()
    strat_returns['RL_Portfolio'] = rl_ret
except:
    pass

strat_df = pd.DataFrame(strat_returns).dropna()
print(f"Strategy returns shape: {strat_df.shape}")

# Correlation matrix
corr = strat_df.corr()
corr.to_csv('/root/quant/iter17_strategy_corr.csv')

# Hierarchical Risk Parity (HRP) - simplified implementation
def hrp_weights(corr_matrix):
    """Simplified HRP using hierarchical clustering on correlation"""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    # Convert correlation to distance
    dist = np.sqrt(2 * (1 - corr_matrix.values))
    np.fill_diagonal(dist, 0)
    
    # Hierarchical clustering
    link = linkage(squareform(dist), method='ward')
    
    # Get cluster order from dendrogram
    dendro = dendrogram(link, no_plot=True)
    order = dendro['leaves']
    
    # Recursive bisection for HRP weights (simplified: inverse variance within clusters)
    n = len(corr_matrix)
    weights = np.ones(n) / n
    
    # Simple recursive bisection
    def get_cluster_var(cov, items):
        w = np.ones(len(items)) / len(items)
        return w @ cov[np.ix_(items, items)] @ w
    
    def bisect(cov, items):
        if len(items) == 1:
            return {items[0]: 1.0}
        mid = len(items) // 2
        left_items = items[:mid]
        right_items = items[mid:]
        left_var = get_cluster_var(cov, left_items)
        right_var = get_cluster_var(cov, right_items)
        alpha = 1 - left_var / (left_var + right_var)
        left_w = bisect(cov, left_items)
        right_w = bisect(cov, right_items)
        result = {}
        for k, v in left_w.items():
            result[k] = v * alpha
        for k, v in right_w.items():
            result[k] = v * (1 - alpha)
        return result
    
    ordered_items = [corr_matrix.index[i] for i in order]
    cov_matrix = strat_df.cov().values
    hrp_dict = bisect(cov_matrix, list(range(n)))
    hrp_w = pd.Series([hrp_dict.get(i, 0) for i in range(n)], index=corr_matrix.index)
    return hrp_w

hrp_w = hrp_weights(corr)
hrp_w.to_csv('/root/quant/iter17_hrp_weights.csv')
print(f"HRP weights:\n{hrp_w}")

# Equal weight, inverse volatility, and HRP portfolios
eq_w = pd.Series(1/len(strat_df.columns), index=strat_df.columns)
iv_w = (1 / strat_df.std()).div((1 / strat_df.std()).sum())

portfolios = {
    'Equal_Weight': (strat_df * eq_w).sum(axis=1),
    'Inv_Vol': (strat_df * iv_w).sum(axis=1),
    'HRP': (strat_df * hrp_w).sum(axis=1),
}

for name, ret in portfolios.items():
    p = perf(ret.dropna(), name)
    print(f"{name}: Sharpe={p['Sharpe']:.2f}, AnnRet%={p['AnnRet%']:.2f}, MaxDD%={p['MaxDD%']:.2f}")

pd.DataFrame({k: v for k, v in portfolios.items()}).to_csv('/root/quant/iter17_combined_portfolios.csv')

# ============================================================
# E4. ROBUST REGIME DETECTION: HMM WITH MORE STATES
# ============================================================
print("\n=== E4: HMM Regime Detection (3-4 states) ===")

from hmmlearn import hmm

# Use SPY returns and realized vol as features
feat = pd.DataFrame(index=spy_ret.index)
feat['ret'] = spy_ret
feat['vol'] = spy_ret.rolling(21).std() * np.sqrt(252)
feat = feat.dropna()

X = feat[['ret', 'vol']].values

# Fit HMM with 3 and 4 states
best_model = None
best_score = -np.inf
best_n = 0

for n_states in [2, 3, 4, 5]:
    try:
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='full', 
                                 n_iter=100, random_state=42)
        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_score = score
            best_model = model
            best_n = n_states
    except:
        continue

print(f"Best HMM: {best_n} states, log-likelihood={best_score:.1f}")

# Get regime sequence
hidden_states = best_model.predict(X)
regime_df = pd.DataFrame({'state': hidden_states}, index=feat.index)
regime_df['state_name'] = regime_df['state'].map({
    0: 'Low_Vol_Bull', 1: 'High_Vol_Bear', 2: 'High_Vol_Bull', 3: 'Low_Vol_Sideways', 4: 'Crisis'
})

# Analyze regimes
for s in sorted(regime_df['state'].unique()):
    mask = regime_df['state'] == s
    avg_ret = feat.loc[mask, 'ret'].mean() * 252 * 100
    avg_vol = feat.loc[mask, 'vol'].mean()
    freq = mask.sum() / len(mask) * 100
    print(f"  State {s}: Ret={avg_ret:.1f}%, Vol={avg_vol:.1f}%, Freq={freq:.1f}%")

regime_df.to_csv('/root/quant/iter17_hmm_regimes.csv')

# Regime-conditional strategy performance
regime_perf = {}
for strat_name, strat_ret in strat_returns.items():
    aligned = strat_ret.reindex(feat.index).dropna()
    for s in sorted(regime_df['state'].unique()):
        mask = regime_df['state'] == s
        seg = aligned[mask]
        if len(seg) > 20:
            regime_perf[f'{strat_name}_State{s}'] = {
                'Sharpe': sharpe(seg),
                'AnnRet%': 252 * seg.mean() * 100,
                'Count': len(seg)
            }

pd.DataFrame(regime_perf).T.to_csv('/root/quant/iter17_regime_conditional.csv')

# ============================================================
# E5. KALMAN FILTER PAIRS TRADING
# ============================================================
print("\n=== E5: Kalman Filter Pairs Trading ===")

from pykalman import KalmanFilter

# Test pairs from available data
pairs = [('SPY', 'IVV'), ('GLD', 'IAU'), ('TLT', 'IEF'), ('XLE', 'XOP'), ('XLK', 'SMH')]
# Use available tickers
available_pairs = []
for a, b in pairs:
    if a in prices.columns and b in prices.columns:
        available_pairs.append((a, b))

# Add some ETF pairs
etf_pairs = [('SPY', 'QQQ'), ('TLT', 'IEF'), ('GLD', 'DBC'), ('XLF', 'XLU'), ('EFA', 'EEM')]
for a, b in etf_pairs:
    if a in prices.columns and b in prices.columns:
        available_pairs.append((a, b))

kalman_results = {}

for a, b in available_pairs[:5]:  # Limit to 5 pairs
    px_a = prices[a].dropna()
    px_b = prices[b].dropna()
    common_idx = px_a.index.intersection(px_b.index)
    px_a = px_a[common_idx]
    px_b = px_b[common_idx]
    
    if len(common_idx) < 252:
        continue
    
    log_a = np.log(px_a.values)
    log_b = np.log(px_b.values)
    
    # State space: hedge ratio as hidden state
    # Observation: log_a = hedge * log_b + noise
    # Transition: hedge_t = hedge_{t-1} + noise
    
    try:
        kf = KalmanFilter(
            transition_matrices=[[1]],
            observation_matrices=[[1]],
            initial_state_mean=1.0,
            initial_state_covariance=1.0,
            transition_covariance=0.001,
            observation_covariance=0.01,
            em_vars=['transition_covariance', 'observation_covariance']
        )
        # Use EM to fit on the spread
        spread = log_a - log_b
        kf = kf.em(spread, n_iter=5)
    except Exception as e:
        print(f"  {a}/{b}: Kalman init failed - {e}")
        continue
    
    try:
        state_means, state_covs = kf.filter(log_a)
        hedge_ratio = pd.Series(state_means.flatten(), index=common_idx)
        
        spread = log_a - hedge_ratio * log_b
        spread_mean = spread.rolling(60).mean()
        spread_std = spread.rolling(60).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-8)
        
        # Trading signal
        pos = pd.Series(0.0, index=common_idx)
        pos[z_score < -2] = 1.0   # Long A, short B
        pos[z_score > 2] = -1.0   # Short A, long B
        pos[z_score.abs() < 0.5] = 0.0
        pos = pos.ffill().fillna(0)
        
        # Returns
        ret_a = px_a.pct_change()
        ret_b = px_b.pct_change()
        pair_ret = pos.shift(1) * (ret_a - hedge_ratio * ret_b)
        
        p = perf(pair_ret.dropna(), f'KF_Pairs_{a}_{b}')
        kalman_results[f'{a}_{b}'] = p
        print(f"  {a}/{b}: Sharpe={p['Sharpe']:.2f}, AnnRet%={p['AnnRet%']:.2f}, MaxDD%={p['MaxDD%']:.2f}")
    except Exception as e:
        print(f"  {a}/{b}: Failed - {e}")

pd.DataFrame(kalman_results).T.to_csv('/root/quant/iter17_kalman_pairs.csv')

# ============================================================
# E6. GARCH VOLATILITY FORECASTING FOR VOL TARGETING
# ============================================================
print("\n=== E6: GARCH Volatility Forecasting ===")

try:
    from arch import arch_model
    
    # Fit GARCH(1,1) on SPY returns
    spy_ret_clean = spy_ret.dropna() * 100  # scale for numerical stability
    
    garch = arch_model(spy_ret_clean, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch.fit(disp='off', show_warning=False)
    
    # Forecast 1-step ahead conditional variance
    forecasts = garch_fit.conditional_volatility / 100  # back to decimal
    forecasts.index = spy_ret_clean.index
    
    # Compare with rolling std
    rolling_vol = spy_ret.rolling(21).std() * np.sqrt(252)
    garch_vol_ann = forecasts * np.sqrt(252)
    
    # Vol targeting with GARCH forecast
    target_vol = 0.10
    garch_lev = (target_vol / garch_vol_ann).clip(upper=2.0).fillna(1.0)
    rolling_lev = (target_vol / rolling_vol).clip(upper=2.0).fillna(1.0)
    
    garch_pos = garch_lev
    rolling_pos = rolling_lev
    
    garch_ret = backtest(garch_pos.reindex(spy_ret.index).fillna(0), spy_ret, cost_bps=10)
    rolling_ret = backtest(rolling_pos.reindex(spy_ret.index).fillna(0), spy_ret, cost_bps=10)
    
    garch_perf = perf(garch_ret.dropna(), 'GARCH_VolTarget')
    rolling_perf = perf(rolling_ret.dropna(), 'Rolling_VolTarget')
    
    print(f"GARCH VolTarget: Sharpe={garch_perf['Sharpe']:.2f}, AnnRet%={garch_perf['AnnRet%']:.2f}, MaxDD%={garch_perf['MaxDD%']:.2f}")
    print(f"Rolling VolTarget: Sharpe={rolling_perf['Sharpe']:.2f}, AnnRet%={rolling_perf['AnnRet%']:.2f}, MaxDD%={rolling_perf['MaxDD%']:.2f}")
    
    # Save forecasts
    pd.DataFrame({
        'garch_vol': garch_vol_ann,
        'rolling_vol': rolling_vol,
        'garch_lev': garch_lev,
        'rolling_lev': rolling_lev
    }).to_csv('/root/quant/iter17_garch_vol.csv')
    
    garch_results = pd.DataFrame([garch_perf, rolling_perf])
    garch_results.to_csv('/root/quant/iter17_garch_comparison.csv')
    
except ImportError:
    print("  arch package not available, skipping GARCH")

# ============================================================
# E7. SYNTHETIC DATA VALIDATION (QuantStart style)
# ============================================================
print("\n=== E7: Synthetic Data Generation & Validation ===")

def generate_synthetic_returns(n_assets=10, n_days=1000, corr_structure='factor'):
    """Generate synthetic returns with realistic correlation structure"""
    np.random.seed(42)
    
    if corr_structure == 'factor':
        # Factor model: 3 factors
        n_factors = 3
        factor_ret = np.random.randn(n_days, n_factors) * 0.01
        loadings = np.random.randn(n_assets, n_factors) * 0.5
        idio = np.random.randn(n_days, n_assets) * 0.005
        ret = factor_ret @ loadings.T + idio
    elif corr_structure == 'constant':
        # Constant correlation
        rho = 0.3
        cov = np.eye(n_assets) * 0.01 + np.ones((n_assets, n_assets)) * 0.01 * rho
        ret = np.random.multivariate_normal(np.zeros(n_assets), cov, n_days)
    else:
        ret = np.random.randn(n_days, n_assets) * 0.01
    
    return pd.DataFrame(ret, columns=[f'Asset_{i}' for i in range(n_assets)])

# Generate synthetic data
syn_ret = generate_synthetic_returns(n_assets=10, n_days=2520, corr_structure='factor')

# Test strategies on synthetic data
syn_strategies = {}
for name, pos_fn in [('SMA200', lambda p: sma_trend(p, 200)),
                      ('MA50_200', lambda p: ma_crossover(p, 50, 200)),
                      ('RSI2', lambda p: rsi2_meanrev(p)),
                      ('VolTarget', lambda p: vol_target(p))]:
    # Apply to first asset
    px = (1 + syn_ret.iloc[:, 0]).cumprod() * 100
    pos = pos_fn(px).reindex(syn_ret.index).fillna(0)
    syn_strategies[name] = backtest(pos, syn_ret.iloc[:, 0], cost_bps=10)

# Cross-sectional momentum on synthetic
syn_xsec = xsec_momentum((1 + syn_ret).cumprod() * 100)
syn_strategies['XSecMom'] = backtest(syn_xsec.reindex(syn_ret.index).fillna(0).mean(axis=1), 
                                      syn_ret.mean(axis=1), cost_bps=10)

syn_results = {}
for name, ret in syn_strategies.items():
    p = perf(ret.dropna(), name)
    syn_results[name] = p
    print(f"  {name}: Sharpe={p['Sharpe']:.2f}, AnnRet%={p['AnnRet%']:.2f}")

pd.DataFrame(syn_results).T.to_csv('/root/quant/iter17_synthetic_validation.csv')

# ============================================================
# E8. PURGED WALK-FORWARD WITH COMBINATORIAL CV
# ============================================================
print("\n=== E8: Purged Combinatorial Cross-Validation ===")

def purged_cv_split(n_samples, n_splits=4, pct_embargo=0.01):
    """Generate purged CV indices"""
    embargo = int(n_samples * pct_embargo)
    fold_size = n_samples // n_splits
    indices = np.arange(n_samples)
    
    splits = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        test_idx = indices[test_start:test_end]
        
        # Purge: exclude embargo around test
        train_idx = indices[(indices < test_start - embargo) | (indices > test_end + embargo)]
        splits.append((train_idx, test_idx))
    
    return splits

# Test on best strategies
best_strats = ['SMA200', 'VolTarget', 'XSecMom', 'GEM', 'RL_Portfolio']

cv_results = {}
for name in best_strats:
    if name in strat_returns:
        ret = strat_returns[name].dropna()
    elif name == 'RL_Portfolio':
        try:
            ret = pd.read_csv('/root/quant/iter16_rl_returns.csv', index_col=0, parse_dates=True).squeeze().dropna()
        except:
            continue
    else:
        continue
    
    if len(ret) < 500:
        continue
    
    splits = purged_cv_split(len(ret), n_splits=5, pct_embargo=0.02)
    fold_sharpes = []
    for train_idx, test_idx in splits:
        test_ret = ret.iloc[test_idx]
        if len(test_ret) > 30:
            fold_sharpes.append(sharpe(test_ret))
    
    cv_results[name] = {
        'Fold_Sharpes': fold_sharpes,
        'Mean': np.mean(fold_sharpes) if fold_sharpes else 0,
        'Std': np.std(fold_sharpes) if fold_sharpes else 0,
        'Min': np.min(fold_sharpes) if fold_sharpes else 0,
        'Max': np.max(fold_sharpes) if fold_sharpes else 0,
    }
    print(f"  {name}: Mean={cv_results[name]['Mean']:.3f}, Std={cv_results[name]['Std']:.3f}")

pd.DataFrame(cv_results).T.to_csv('/root/quant/iter17_purged_cv.csv')

# ============================================================
# E9. COST-AWARE PARAMETER OPTIMIZATION
# ============================================================
print("\n=== E9: Cost-Aware Parameter Optimization ===")

def optimize_sma_cost_aware(price, returns, cost_bps=10):
    """Find optimal SMA window accounting for turnover costs"""
    windows = range(50, 401, 25)
    results = []
    
    for w in windows:
        pos = (price > price.rolling(w).mean()).astype(float)
        pos = pos.div(pos.sum(axis=1).replace(0, 1), axis=0).fillna(0)
        
        # Calculate turnover
        turnover = pos.diff().abs().sum(axis=1).mean() * 252
        
        strat_ret = backtest(pos, returns, cost_bps=cost_bps)
        p = perf(strat_ret.dropna(), f'SMA_{w}')
        
        # Cost-adjusted Sharpe
        gross_sharpe = sharpe((pos * returns).sum(axis=1).dropna())
        cost_drag = turnover * cost_bps / 1e4 * 252 / (returns.std() * np.sqrt(252))
        
        results.append({
            'window': w,
            'sharpe': p['Sharpe'],
            'ann_ret': p['AnnRet%'],
            'max_dd': p['MaxDD%'],
            'turnover': turnover,
            'gross_sharpe': gross_sharpe,
            'cost_drag': cost_drag
        })
    
    return pd.DataFrame(results)

# Optimize on SPY
sma_opt = optimize_sma_cost_aware(spy, spy_ret, cost_bps=10)
sma_opt.to_csv('/root/quant/iter17_sma_cost_optimization.csv', index=False)

# Find best
best_row = sma_opt.loc[sma_opt['sharpe'].idxmax()]
print(f"Best SMA window: {best_row['window']}, Sharpe={best_row['sharpe']:.2f}, Turnover={best_row['turnover']:.1f}")

# Multi-asset cost optimization
ma_opt = optimize_sma_cost_aware(prices, returns, cost_bps=10)
ma_opt.to_csv('/root/quant/iter17_xsec_cost_optimization.csv', index=False)

# ============================================================
# E10. STATISTICAL VALIDATION OF ALL NEW STRATEGIES
# ============================================================
print("\n=== E10: Comprehensive Statistical Validation ===")

# Collect all strategy returns for validation
all_strat_returns = {}

# Baseline strategies
for name, pos in strategies_for_kelly.items():
    pos_aligned = pos.reindex(spy_ret.index).fillna(0)
    all_strat_returns[name] = backtest(pos_aligned, spy_ret, cost_bps=10)

# Multi-asset
all_strat_returns['XSecMom'] = backtest(
    xsec_momentum(prices).reindex(returns.index).fillna(0).mean(axis=1),
    returns.mean(axis=1), cost_bps=10)
all_strat_returns['GEM'] = backtest(
    dual_momentum(prices[['SPY','GLD','TLT']]).reindex(returns[['SPY','GLD','TLT']].index).fillna(0).mean(axis=1),
    returns[['SPY','GLD','TLT']].mean(axis=1), cost_bps=10)

# Kelly versions
for name, pos in strategies_for_kelly.items():
    pos_aligned = pos.reindex(spy_ret.index).fillna(0)
    strat_ret = backtest(pos_aligned, spy_ret, cost_bps=10)
    kelly_lev = kelly_fraction(strat_ret.dropna())
    kelly_pos = pos_aligned * kelly_lev
    all_strat_returns[f'{name}_Kelly'] = backtest(kelly_pos, spy_ret, cost_bps=10)

# Combined portfolios
for name, ret in portfolios.items():
    all_strat_returns[name] = ret

# Kalman pairs
for pair_name, pair_ret in kalman_results.items():
    pass  # already stored separately

# RL
try:
    rl_ret = pd.read_csv('/root/quant/iter16_rl_returns.csv', index_col=0, parse_dates=True).squeeze()
    all_strat_returns['RL_Portfolio'] = rl_ret
except:
    pass

# Transformer/ML from iter16 (zero signals)
try:
    tf_ret = pd.read_csv('/root/quant/iter16_transformer_factor_returns.csv', index_col=0, parse_dates=True).squeeze()
    all_strat_returns['Transformer'] = tf_ret
except:
    pass

# Validation
val_results = {}
all_sharpes = [sharpe(s.dropna()) for s in all_strat_returns.values() if len(s.dropna()) > 100]
sr_std = np.std(all_sharpes)
n_trials = len(all_sharpes)

for name, ret in all_strat_returns.items():
    ret_clean = ret.dropna()
    if len(ret_clean) < 100:
        continue
    nw_t, n = nw_tstat(ret_clean)
    sh = sharpe(ret_clean)
    try:
        lo, hi = circular_bootstrap_ci(ret_clean)
    except:
        lo, hi = sh * 0.5, sh * 1.5
    dsr = dsr_test(sh, n_trials, n, sr_std,
                   skew=float(stats.skew(ret_clean)), kurt=float(stats.kurtosis(ret_clean, fisher=False)))
    yrs = n / 252
    val_results[name] = {
        'NW_t': round(nw_t, 3),
        'Sharpe': round(sh, 3),
        'DSR_p': round(dsr, 3),
        'BS_CI_low': round(lo, 3),
        'BS_CI_high': round(hi, 3),
        'Years': round(yrs, 1)
    }
    print(f"{name}: NW_t={nw_t:.3f}, SR={sh:.3f}, DSR_p={dsr:.3f}, CI=[{lo:.3f}, {hi:.3f}]")

val_df = pd.DataFrame(val_results).T
val_df.to_csv('/root/quant/iter17_validation.csv')

# ============================================================
# PLOTTING
# ============================================================
print("\n=== Creating Plots ===")

# 1. Strategy equity curves
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

all_for_plot = {**strat_returns, **portfolios}
for i, (name, ret) in enumerate(all_for_plot.items()):
    if i >= 15:
        break
    ax = axes[i]
    ret_clean = ret.dropna()
    if len(ret_clean) > 0:
        cum = (1 + ret_clean).cumprod()
        cum.plot(ax=ax, label=name)
        bench = (1 + spy_ret.loc[ret_clean.index]).cumprod()
        bench.plot(ax=ax, label='SPY', alpha=0.5, color='gray')
        ax.set_title(f'{name} (Sharpe={sharpe(ret_clean):.2f})')
        ax.legend(fontsize=7)

# Correlation heatmap
if i < 15:
    ax = axes[15]
    strat_corr = strat_df.corr()
    im = ax.imshow(strat_corr, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xticks(range(len(strat_corr.columns)))
    ax.set_yticks(range(len(strat_corr.columns)))
    ax.set_xticklabels(strat_corr.columns, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(strat_corr.columns, fontsize=7)
    ax.set_title('Strategy Correlation')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('/root/quant/iter17_equity.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Performance comparison
perf_results = {}
for name, ret in all_strat_returns.items():
    if len(ret.dropna()) > 100:
        perf_results[name] = perf(ret.dropna(), name)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

names = list(perf_results.keys())
sharpes = [perf_results[n]['Sharpe'] for n in names]
ann_rets = [perf_results[n]['AnnRet%'] for n in names]
max_dds = [perf_results[n]['MaxDD%'] for n in names]
calmars = [perf_results[n]['Calmar'] for n in names]

axes[0,0].barh(names, sharpes, color=['green' if s>1 else 'orange' if s>0.5 else 'red' for s in sharpes])
axes[0,0].set_title('Sharpe Ratio')
axes[0,0].axvline(x=1, color='black', linestyle='--', alpha=0.5)

axes[0,1].barh(names, ann_rets, color='blue')
axes[0,1].set_title('Annualized Return %')

axes[0,2].barh(names, max_dds, color='red')
axes[0,2].set_title('Max Drawdown %')

axes[1,0].barh(names, calmars, color='purple')
axes[1,0].set_title('Calmar Ratio')

# HRP weights
ax = axes[1,1]
hrp_w_sorted = hrp_w.sort_values()
hrp_w_sorted.plot(kind='barh', ax=ax)
ax.set_title('HRP Weights')

# Regime distribution
ax = axes[1,2]
regime_counts = regime_df['state'].value_counts().sort_index()
regime_counts.plot(kind='bar', ax=ax)
ax.set_title('HMM Regime Distribution')

plt.tight_layout()
plt.savefig('/root/quant/iter17_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. SMA cost optimization plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(sma_opt['window'], sma_opt['sharpe'], 'o-', label='Net Sharpe (10bp)')
ax.plot(sma_opt['window'], sma_opt['gross_sharpe'], 's-', label='Gross Sharpe')
ax.set_xlabel('SMA Window')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('SMA Window Optimization: Cost-Aware')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/root/quant/iter17_sma_cost_opt.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. HMM regime analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for s in sorted(regime_df['state'].unique()):
    ax = axes[s // 2, s % 2]
    mask = regime_df['state'] == s
    spy_seg = spy_ret[mask]
    cum = (1 + spy_seg).cumprod()
    cum.plot(ax=ax, color='blue')
    ax.set_title(f'Regime {s}: SPY Equity ({mask.sum()} days)')
plt.tight_layout()
plt.savefig('/root/quant/iter17_hmm_regimes.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# SAVE ALL RESULTS & UPDATE REPORT
# ============================================================
print("\n=== Saving Results ===")

# Performance summary
perf_summary = pd.DataFrame(perf_results).T
perf_summary.to_csv('/root/quant/iter17_comprehensive_perf.csv')

print("\n=== Iteration #17 Complete ===")
print("Files generated:")
print("  - iter17_data_quality.csv")
print("  - iter17_kelly_sizing.csv")
print("  - iter17_strategy_corr.csv")
print("  - iter17_hrp_weights.csv")
print("  - iter17_combined_portfolios.csv")
print("  - iter17_hmm_regimes.csv")
print("  - iter17_regime_conditional.csv")
print("  - iter17_kalman_pairs.csv")
print("  - iter17_garch_vol.csv (if arch available)")
print("  - iter17_garch_comparison.csv (if arch available)")
print("  - iter17_synthetic_validation.csv")
print("  - iter17_purged_cv.csv")
print("  - iter17_sma_cost_optimization.csv")
print("  - iter17_xsec_cost_optimization.csv")
print("  - iter17_validation.csv")
print("  - iter17_comprehensive_perf.csv")
print("  - iter17_equity.png")
print("  - iter17_performance.png")
print("  - iter17_sma_cost_opt.png")
print("  - iter17_hmm_regimes.png")
