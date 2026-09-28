"""Iteration #25: QuantStart-Inspired Novel Research - Fixed & Enhanced
- Synthetic correlated data generation (QuantStart: "Generating Synthetic Histories")
- Vasicek/OU calibrated mean-reversion strategies (QuantStart: "Vasicek Model Simulation", "Ornstein-Uhlenbeck")
- Signature-based regime features (QuantStart: "Rough Path Theory and Signatures" Parts 1-4)
- Multi-asset regime-conditional allocation with costs (QuantStart: "Systematic Tactical Asset Allocation")
- Purged CV + Bootstrap validation (QuantStart: "Backtesting Systematic Trading Strategies")
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from engine import load, backtest, perf, sharpe, max_dd
from strategies import *

np.random.seed(42)

# ============================================================
# 1. DATA LOADING
# ============================================================
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'TLT', 'GLD', 'DBC', 'VNQ',
           'XLE', 'XLF', 'XLK', 'XLP', 'XLU', 'XLV', 'VIG', 'SCHD', 'SHY']

available = []
for t in tickers:
    try:
        df = load(t)
        available.append(t)
    except:
        pass

print(f"Loaded {len(available)} tickers: {available}")

price_df = pd.DataFrame()
for t in available:
    df = load(t)
    price_df[t] = df['Close']

returns = price_df.pct_change().dropna()
returns = returns.loc['2015-01-01':'2024-12-31']

# ============================================================
# 2. SYNTHETIC CORRELATED DATA GENERATION (QuantStart style)
# ============================================================
def generate_synthetic_correlated(returns, n_samples=2000):
    """Generate synthetic returns preserving correlation structure using Cholesky"""
    mu = returns.mean().values
    lw = LedoitWolf().fit(returns)
    cov = lw.covariance_
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(len(mu)))
    z = np.random.randn(n_samples, len(mu))
    synth = z @ L.T + mu
    synth_df = pd.DataFrame(synth, columns=returns.columns, index=pd.date_range(returns.index[0], periods=n_samples, freq='B'))
    return synth_df

synth_returns = generate_synthetic_correlated(returns, n_samples=2520)  # ~10 years

# ============================================================
# 3. VASICEK / OU CALIBRATION (QuantStart articles)
# ============================================================
def calibrate_vasicek(series, dt=1/252):
    """Calibrate Vasicek (OU) process: dx = kappa*(theta - x)*dt + sigma*dW"""
    x = series.values
    x_lag = np.roll(x, 1)[1:]
    x = x[1:]
    dx = x - x_lag
    X = np.column_stack([np.ones_like(x_lag), x_lag])
    beta = np.linalg.lstsq(X, dx, rcond=None)[0]
    kappa = -beta[1] / dt
    theta = beta[0] / (kappa * dt)
    resid = dx - X @ beta
    sigma = resid.std() / np.sqrt(dt)
    return max(kappa, 0.01), theta, max(sigma, 0.001)

def vasicek_signal(returns, window=60):
    """Generate mean-reversion signals using rolling Vasicek calibration"""
    signals = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        col_signals = np.zeros(len(returns))
        for i in range(window, len(returns)):
            window_data = returns[col].iloc[i-window:i]
            kappa, theta, sigma = calibrate_vasicek(window_data)
            current = returns[col].iloc[i]
            z = (current - theta) / (sigma / np.sqrt(2 * kappa))
            col_signals[i] = -np.clip(z / 3, -1, 1)
        signals.loc[:, col] = col_signals
    return signals

# ============================================================
# 4. SIGNATURE-BASED REGIME FEATURES (Rough Path Theory - QuantStart)
# ============================================================
def log_signature(path, level=2):
    """Compute log-signature of a path (truncated tensor algebra)"""
    incs = np.diff(path)
    if len(incs) == 0:
        return np.zeros(level * len(path))
    sig = []
    sig.append(incs.mean())  # Level 1: mean increment
    if level >= 2 and len(incs) > 1:
        area = np.sum([incs[i] * incs[j] for i in range(len(incs)) for j in range(i)])
        sig.append(area / len(incs))
    return np.array(sig)

def signature_regime_features(returns, window=60, level=2):
    """Extract signature features for regime detection"""
    n_assets = len(returns.columns)
    features = []
    dates = []
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        sig_feats = []
        for col in returns.columns:
            path = window_data[col].cumsum().values
            sig_feats.extend(log_signature(path, level))
        if level >= 2:
            for j in range(min(5, n_assets)):
                for k in range(j+1, min(5, n_assets)):
                    p1 = window_data.iloc[:, j].cumsum().values
                    p2 = window_data.iloc[:, k].cumsum().values
                    inc1 = np.diff(p1)
                    inc2 = np.diff(p2)
                    area = np.sum([inc1[i] * inc2[j] for i in range(len(inc1)) for j in range(i)])
                    sig_feats.append(area / len(inc1))
        features.append(sig_feats)
        dates.append(returns.index[i])
    return pd.DataFrame(features, index=dates)

# ============================================================
# 5. HMM REGIME DETECTION (QuantStart: "Market Regime Detection using HMM")
# ============================================================
def fit_hmm_regimes(features, n_states=3):
    """Fit HMM on signature features for regime detection"""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.fillna(0))
    
    model = hmm.GaussianHMM(n_components=n_states, covariance_type='full', 
                            n_iter=100, random_state=42)
    model.fit(X)
    regimes = model.predict(X)
    probs = model.predict_proba(X)
    return regimes, probs, model, scaler

# ============================================================
# 6. REGIME-CONDITIONAL PORTFOLIO OPTIMIZATION
# ============================================================
def regime_conditional_portfolio(returns, regimes, probs, lookback=60):
    """Optimize portfolio weights conditional on regime"""
    n_states = probs.shape[1]
    aligned_returns = returns.iloc[-len(probs):]
    weights = pd.DataFrame(0.0, index=aligned_returns.index, columns=returns.columns)
    
    for state in range(n_states):
        state_mask = probs[:, state] > 0.5
        if state_mask.sum() < lookback:
            continue
        state_returns = aligned_returns.iloc[state_mask]
        if len(state_returns) < lookback:
            continue
        
        mu = state_returns.mean().values
        lw = LedoitWolf().fit(state_returns)
        cov = lw.covariance_
        
        inv_cov = np.linalg.pinv(cov + 1e-4 * np.eye(len(mu)))
        w = inv_cov @ mu
        w = np.maximum(w, 0)  # Long only
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(len(mu)) / len(mu)
        
        state_dates = aligned_returns.index[state_mask]
        for d in state_dates:
            weights.loc[d] = w
    
    weights = weights.ffill().fillna(1/len(returns.columns))
    weights = weights.reindex(returns.index).ffill().fillna(1/len(returns.columns))
    return weights

# ============================================================
# 7. COMBINED STRATEGY: VASICEK MR + SIGNATURE REGIMES
# ============================================================
print("Computing Vasicek mean-reversion signals...")
mr_signals = vasicek_signal(returns, window=60)

print("Computing signature features...")
sig_features = signature_regime_features(returns, window=60, level=2)

print("Fitting HMM regimes...")
regimes, probs, hmm_model, scaler = fit_hmm_regimes(sig_features, n_states=3)

print("Computing regime-conditional portfolio...")
regime_weights = regime_conditional_portfolio(returns, regimes, probs, lookback=60)

# Combine: Use MR signals as "tilt" on top of regime portfolio
# Regime weights give long-only allocation, MR gives directional tilt per asset
combined_pos = regime_weights * (1 + mr_signals)  # Tilt regime weights by MR signal
combined_pos = combined_pos.clip(-1, 1)

# Also test simple combination: MR signal * regime weight (directional)
combined_pos_v2 = mr_signals * regime_weights.abs()
combined_pos_v2 = combined_pos_v2.clip(-1, 1)

# ============================================================
# 8. BACKTESTING WITH TRANSACTION COSTS
# ============================================================
cost_bps = 10
aligned_returns = returns.iloc[-len(combined_pos):]

strategies = {
    'Vasicek_MR': mr_signals,
    'Regime_Conditional': regime_weights,
    'Combined_Tilt': combined_pos,
    'Combined_Directional': combined_pos_v2,
    'Equal_Weight': pd.DataFrame(1/len(returns.columns), index=returns.index, columns=returns.columns),
}

results = {}
for name, pos in strategies.items():
    if isinstance(pos, pd.DataFrame):
        strat_ret = backtest(pos, aligned_returns, cost_bps=cost_bps)
    else:
        strat_ret = backtest(pos, aligned_returns['SPY'], cost_bps=cost_bps)
    results[name] = perf(strat_ret, name)
    print(f"{name}: {results[name]}")

# ============================================================
# 9. PURGED CROSS-VALIDATION + BOOTSTRAP
# ============================================================
def purged_kfold_indices(n, n_splits=5, embargo_pct=0.01):
    fold_size = n // n_splits
    embargo = int(n * embargo_pct)
    indices = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n
        test_idx = np.arange(start, end)
        train_idx = np.concatenate([
            np.arange(0, max(0, start - embargo)),
            np.arange(min(n, end + embargo), n)
        ])
        indices.append((train_idx, test_idx))
    return indices

def bootstrap_sharpe(returns, n_boot=1000):
    n = len(returns)
    boots = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_ret = returns.iloc[idx]
        boots.append(sharpe(boot_ret))
    return np.array(boots)

# Evaluate best combined strategy
best_combined_name = 'Combined_Tilt'
best_combined_pos = strategies[best_combined_name]
best_combined_ret = backtest(best_combined_pos, aligned_returns, cost_bps=cost_bps)

n = len(best_combined_pos)
folds = purged_kfold_indices(n, n_splits=5, embargo_pct=0.01)

cv_sharpes = []
for train_idx, test_idx in folds:
    train_returns = aligned_returns.iloc[train_idx]
    test_returns = aligned_returns.iloc[test_idx]
    train_pos = best_combined_pos.iloc[train_idx]
    test_pos = best_combined_pos.iloc[test_idx]
    
    strat_train = backtest(train_pos, train_returns, cost_bps=cost_bps)
    strat_test = backtest(test_pos, test_returns, cost_bps=cost_bps)
    cv_sharpes.append(sharpe(strat_test))

cv_sharpes = np.array(cv_sharpes)
print(f"CV Sharpe ({best_combined_name}): mean={cv_sharpes.mean():.4f}, std={cv_sharpes.std():.4f}")

boot_sharpes = bootstrap_sharpe(best_combined_ret, n_boot=1000)
print(f"Bootstrap Sharpe: mean={boot_sharpes.mean():.4f}, std={boot_sharpes.std():.4f}")

# ============================================================
# 10. PROBABILISTIC SHARPE RATIO (PSR)
# ============================================================
def probabilistic_sharpe(sharpe_obs, sharpe_bench=0, n_obs=None, skew=0, kurt=3):
    if n_obs is None:
        n_obs = len(best_combined_ret)
    if sharpe_obs == sharpe_bench:
        return 0.5
    z = (sharpe_obs - sharpe_bench) * np.sqrt(n_obs - 1) / np.sqrt(1 - skew*sharpe_obs + (kurt-1)/4*sharpe_obs**2)
    return norm.cdf(z)

psr = probabilistic_sharpe(sharpe(best_combined_ret), sharpe_bench=0, n_obs=len(best_combined_ret),
                           skew=best_combined_ret.skew(), kurt=best_combined_ret.kurtosis())
print(f"PSR: {psr:.4f}")

# ============================================================
# 11. SYNTHETIC VALIDATION (QuantStart: "Generating Synthetic Histories")
# ============================================================
print("\n=== Synthetic Validation ===")
synth_strategies = {}
for name, pos_fn in [('Vasicek_MR', vasicek_signal), ('Regime_Conditional', regime_conditional_portfolio)]:
    if name == 'Vasicek_MR':
        synth_pos = pos_fn(synth_returns, window=60)
        synth_ret = backtest(synth_pos, synth_returns, cost_bps=cost_bps)
    else:
        synth_sig = signature_regime_features(synth_returns, window=60, level=2)
        synth_regimes, synth_probs, _, _ = fit_hmm_regimes(synth_sig, n_states=3)
        synth_pos = pos_fn(synth_returns, synth_regimes, synth_probs, lookback=60)
        synth_ret = backtest(synth_pos, synth_returns, cost_bps=cost_bps)
    synth_strategies[name] = perf(synth_ret, f"{name}_Synthetic")
    print(f"{name}_Synthetic: {synth_strategies[name]}")

# ============================================================
# 12. REGIME CHARACTERIZATION
# ============================================================
regime_chars = []
aligned_returns_reg = returns.iloc[-len(regimes):]
for s in range(3):
    mask = regimes == s
    if mask.sum() > 0:
        regime_ret = aligned_returns_reg[mask].mean(axis=1)
        regime_chars.append({
            'regime': s,
            'frequency': mask.mean(),
            'mean_ret': regime_ret.mean() * 252,
            'vol': regime_ret.std() * np.sqrt(252),
            'sharpe': sharpe(regime_ret)
        })
regime_df = pd.DataFrame(regime_chars)
print("\nRegime Characteristics:")
print(regime_df)

# ============================================================
# 13. SAVE ALL OUTPUTS
# ============================================================
# Performance summary
perf_df = pd.DataFrame(results).T
perf_df.to_csv('iter25_comprehensive_perf.csv')

cv_df = pd.DataFrame({'fold': range(len(cv_sharpes)), 'sharpe': cv_sharpes})
cv_df.to_csv('iter25_comprehensive_walkforward.csv')

val_df = pd.DataFrame({
    'metric': ['PSR', 'Bootstrap_Mean', 'Bootstrap_Std', 'CV_Mean', 'CV_Std'],
    'value': [psr, boot_sharpes.mean(), boot_sharpes.std(), cv_sharpes.mean(), cv_sharpes.std()]
})
val_df.to_csv('iter25_comprehensive_validation.csv')

# Synthetic validation
synth_perf_df = pd.DataFrame(synth_strategies).T
synth_perf_df.to_csv('iter25_synthetic_validation.csv')

# Regime info
regime_df.to_csv('iter25_signature_regimes.csv')

# Returns for plotting
best_combined_ret.to_csv('iter25_combined_returns.csv')
aligned_returns.to_csv('iter25_aligned_returns.csv')

# Vasicek params
vasicek_params = []
for col in returns.columns:
    kappa, theta, sigma = calibrate_vasicek(returns[col])
    vasicek_params.append({'asset': col, 'kappa': kappa, 'theta': theta, 'sigma': sigma})
pd.DataFrame(vasicek_params).to_csv('iter25_vasicek_calibration.csv')

# ============================================================
# 14. PLOTTING
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Equity curves
for name, pos in strategies.items():
    if isinstance(pos, pd.DataFrame):
        ret = backtest(pos, aligned_returns, cost_bps=cost_bps)
    else:
        ret = backtest(pos, aligned_returns['SPY'], cost_bps=cost_bps)
    cum = (1 + ret).cumprod()
    axes[0, 0].plot(cum.index, cum.values, label=name, linewidth=1)
axes[0, 0].set_title('Equity Curves (Net of Costs)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Regime probabilities
sig_dates = sig_features.index
for s in range(3):
    axes[0, 1].plot(sig_dates, probs[:, s], label=f'Regime {s}', alpha=0.7)
axes[0, 1].set_title('HMM Regime Probabilities (Signature Features)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Bootstrap Sharpe distribution
axes[1, 0].hist(boot_sharpes, bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(sharpe(best_combined_ret), color='red', linestyle='--', label=f'Observed: {sharpe(best_combined_ret):.3f}')
axes[1, 0].axvline(boot_sharpes.mean(), color='green', linestyle='--', label=f'Bootstrap Mean: {boot_sharpes.mean():.3f}')
axes[1, 0].set_title('Bootstrap Sharpe Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# CV Sharpe
axes[1, 1].bar(range(len(cv_sharpes)), cv_sharpes, alpha=0.7, edgecolor='black')
axes[1, 1].axhline(cv_sharpes.mean(), color='red', linestyle='--', label=f'Mean: {cv_sharpes.mean():.3f}')
axes[1, 1].set_title('Purged CV Sharpe by Fold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iter25_equity.png', dpi=150)
plt.close()

# Regime characterization plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['green', 'gray', 'red']
for _, row in regime_df.iterrows():
    ax.scatter(row['vol']*100, row['mean_ret']*100, s=200, c=colors[int(row['regime'])], 
               label=f"Regime {int(row['regime'])} (freq={row['frequency']:.1%})", alpha=0.7)
ax.set_xlabel('Annualized Volatility (%)')
ax.set_ylabel('Annualized Return (%)')
ax.set_title('Regime Risk-Return Profile (Signature+HMM)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('iter25_regime_profile.png', dpi=150)
plt.close()

print("\n=== Iteration #25 Complete ===")
print(f"Files saved: iter25_*.csv, iter25_*.png")