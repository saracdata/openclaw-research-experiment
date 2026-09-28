"""Iteration #26: Regime-Conditioned VRP + Tail Hedging (Best of Iter #14 + #25)
- Combines Iteration #14's Tail_Hedged (Sharpe 2.99) with Iteration #25's Regime Detection
- VRP strategy conditioned on signature+HMM regimes
- Tail hedging overlay on regime-conditional portfolio
- Synthetic validation (QuantStart: "Generating Synthetic Histories")
- Purged CV + Bootstrap validation
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
# Use Iteration #14's 30 ETFs for VRP (needs options proxies) + Iteration #25's 19 ETFs
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'TLT', 'GLD', 'DBC', 'VNQ',
           'XLE', 'XLF', 'XLK', 'XLP', 'XLU', 'XLV', 'VIG', 'SCHD', 'SHY',
           'BIL', 'GOVT', 'HYG', 'IEF', 'IEI', 'LQD', 'MDY', 'SMH', 'VEA', 'VTI', 'VWO']

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
# 2. SYNTHETIC CORRELATED DATA GENERATION
# ============================================================
def generate_synthetic_correlated(returns, n_samples=2000):
    mu = returns.mean().values
    lw = LedoitWolf().fit(returns)
    cov = lw.covariance_
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(len(mu)))
    z = np.random.randn(n_samples, len(mu))
    synth = z @ L.T + mu
    synth_df = pd.DataFrame(synth, columns=returns.columns, index=pd.date_range(returns.index[0], periods=n_samples, freq='B'))
    return synth_df

synth_returns = generate_synthetic_correlated(returns, n_samples=2520)

# ============================================================
# 3. VRP STRATEGY (Variance Risk Premium - Iteration #14 style)
# ============================================================
def vrp_signal(returns, window=60):
    """
    VRP = Implied Vol - Realized Vol (proxy using options-implied vs historical)
    Proxy: Use rolling volatility term structure slope as VRP signal
    """
    signals = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    for col in returns.columns:
        col_signals = np.zeros(len(returns))
        for i in range(window, len(returns)):
            # Short-term realized vol (21d)
            short_vol = returns[col].iloc[i-21:i].std() * np.sqrt(252)
            # Medium-term realized vol (63d)
            med_vol = returns[col].iloc[i-63:i].std() * np.sqrt(252)
            # Long-term realized vol (126d)
            long_vol = returns[col].iloc[i-126:i].std() * np.sqrt(252) if i >= 126 else med_vol
            
            # VRP proxy: term structure slope (short - long) / long
            # Positive = vol term structure upward sloping = VRP positive = short vol
            if long_vol > 0:
                vrp = (short_vol - long_vol) / long_vol
            else:
                vrp = 0
            
            # Signal: negative VRP (backwardation) -> long vol / long asset
            # Positive VRP (contango) -> short vol / underweight asset
            col_signals[i] = -np.clip(vrp * 2, -1, 1)  # Scale and clip
        
        signals.loc[:, col] = col_signals
    
    return signals

# ============================================================
# 4. SIGNATURE-BASED REGIME FEATURES
# ============================================================
def log_signature(path, level=3):  # Level 3 for richer features
    incs = np.diff(path)
    if len(incs) == 0:
        return np.zeros(level * len(path))
    sig = []
    sig.append(incs.mean())  # Level 1
    if level >= 2 and len(incs) > 1:
        area = np.sum([incs[i] * incs[j] for i in range(len(incs)) for j in range(i)])
        sig.append(area / len(incs))
    if level >= 3 and len(incs) > 2:
        # Level 3: cubic term
        cubic = np.sum([incs[i] * incs[j] * incs[k] 
                       for i in range(len(incs)) 
                       for j in range(i) 
                       for k in range(j)])
        sig.append(cubic / len(incs))
    return np.array(sig)

def signature_regime_features(returns, window=60, level=3):
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
            for j in range(min(8, n_assets)):  # More cross-asset pairs
                for k in range(j+1, min(8, n_assets)):
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
# 5. HMM REGIME DETECTION
# ============================================================
def fit_hmm_regimes(features, n_states=4):  # 4 states for more granularity
    scaler = StandardScaler()
    X = scaler.fit_transform(features.fillna(0))
    
    model = hmm.GaussianHMM(n_components=n_states, covariance_type='full', 
                            n_iter=200, random_state=42)
    model.fit(X)
    regimes = model.predict(X)
    probs = model.predict_proba(X)
    return regimes, probs, model, scaler

# ============================================================
# 6. REGIME-CONDITIONAL PORTFOLIO OPTIMIZATION
# ============================================================
def regime_conditional_portfolio(returns, regimes, probs, lookback=60):
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
# 7. TAIL HEDGING (Iteration #14 style)
# ============================================================
def tail_hedge_overlay(base_weights, returns, hedge_ratio=0.2, tail_threshold=-0.02):
    """
    Add tail hedge: when portfolio returns breach threshold, 
    shift to defensive assets (TLT, SHY, GLD, BIL)
    """
    defensive = ['TLT', 'SHY', 'GLD', 'BIL', 'GOVT', 'IEF', 'IEI']
    defensive = [d for d in defensive if d in returns.columns]
    
    hedged_weights = base_weights.copy()
    portfolio_ret = (base_weights.shift(1) * returns).sum(axis=1).fillna(0)
    
    # Rolling portfolio vol for scaling
    port_vol = portfolio_ret.rolling(21).std() * np.sqrt(252)
    
    for i in range(1, len(hedged_weights)):
        if portfolio_ret.iloc[i-1] < tail_threshold and port_vol.iloc[i-1] > 0:
            # Activate tail hedge: move hedge_ratio to defensive assets
            current_w = hedged_weights.iloc[i].values
            hedge_amount = hedge_ratio
            
            # Reduce risky assets proportionally
            risky_mask = ~hedged_weights.columns.isin(defensive)
            risky_sum = current_w[risky_mask].sum()
            if risky_sum > 0:
                current_w[risky_mask] *= (1 - hedge_amount / risky_sum)
            
            # Add to defensive
            def_mask = hedged_weights.columns.isin(defensive)
            if def_mask.sum() > 0:
                current_w[def_mask] += hedge_amount / def_mask.sum()
            
            hedged_weights.iloc[i] = np.maximum(current_w, 0)
            hedged_weights.iloc[i] /= hedged_weights.iloc[i].sum()
    
    return hedged_weights

# ============================================================
# 8. COMBINED STRATEGY: REGIME VRP + TAIL HEDGE
# ============================================================
print("Computing VRP signals...")
vrp_signals = vrp_signal(returns, window=60)

print("Computing signature features...")
sig_features = signature_regime_features(returns, window=60, level=3)

print("Fitting HMM regimes...")
regimes, probs, hmm_model, scaler = fit_hmm_regimes(sig_features, n_states=4)

print("Computing regime-conditional portfolio...")
regime_weights = regime_conditional_portfolio(returns, regimes, probs, lookback=60)

# Combine: VRP as tilt on regime weights
vrp_tilted = regime_weights * (1 + vrp_signals * 0.5)  # 50% tilt strength
vrp_tilted = vrp_tilted.clip(-1, 1)

# Apply tail hedging
tail_hedged = tail_hedge_overlay(vrp_tilted, returns, hedge_ratio=0.3, tail_threshold=-0.015)

# Also test: pure regime + tail hedge (no VRP)
regime_tail_hedged = tail_hedge_overlay(regime_weights, returns, hedge_ratio=0.3, tail_threshold=-0.015)

# Also test: VRP only + tail hedge
vrp_only = vrp_signals.copy()
vrp_tail_hedged = tail_hedge_overlay(vrp_only, returns, hedge_ratio=0.3, tail_threshold=-0.015)

# ============================================================
# 9. BACKTESTING
# ============================================================
cost_bps = 10
aligned_returns = returns.iloc[-len(vrp_tilted):]

strategies = {
    'Regime_Conditional': regime_weights,
    'VRP_Tilted': vrp_tilted,
    'Regime_TailHedged': regime_tail_hedged,
    'VRP_TailHedged': vrp_tail_hedged,
    'Full_Combo': tail_hedged,
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
# 10. PURGED CV + BOOTSTRAP
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

def probabilistic_sharpe(sharpe_obs, sharpe_bench=0, n_obs=None, skew=0, kurt=3):
    if n_obs is None:
        n_obs = len(returns)
    if sharpe_obs == sharpe_bench:
        return 0.5
    z = (sharpe_obs - sharpe_bench) * np.sqrt(n_obs - 1) / np.sqrt(1 - skew*sharpe_obs + (kurt-1)/4*sharpe_obs**2)
    return norm.cdf(z)

# Evaluate all strategies
validation_results = {}
for name, pos in strategies.items():
    if name == 'Equal_Weight':
        continue
    strat_ret = backtest(pos, aligned_returns, cost_bps=cost_bps)
    
    # Purged CV
    n = len(pos)
    folds = purged_kfold_indices(n, n_splits=5, embargo_pct=0.01)
    cv_sharpes = []
    for train_idx, test_idx in folds:
        train_returns = aligned_returns.iloc[train_idx]
        test_returns = aligned_returns.iloc[test_idx]
        train_pos = pos.iloc[train_idx]
        test_pos = pos.iloc[test_idx]
        strat_test = backtest(test_pos, test_returns, cost_bps=cost_bps)
        cv_sharpes.append(sharpe(strat_test))
    cv_sharpes = np.array(cv_sharpes)
    
    # Bootstrap
    boot_sharpes = bootstrap_sharpe(strat_ret, n_boot=500)
    
    # PSR
    psr = probabilistic_sharpe(sharpe(strat_ret), sharpe_bench=0, n_obs=len(strat_ret),
                               skew=strat_ret.skew(), kurt=strat_ret.kurtosis())
    
    validation_results[name] = {
        'sharpe': sharpe(strat_ret),
        'psr': psr,
        'cv_mean': cv_sharpes.mean(),
        'cv_std': cv_sharpes.std(),
        'boot_mean': boot_sharpes.mean(),
        'boot_std': boot_sharpes.std(),
    }
    print(f"\n{name} Validation:")
    print(f"  Sharpe: {sharpe(strat_ret):.4f}, PSR: {psr:.4f}")
    print(f"  CV: {cv_sharpes.mean():.4f} ± {cv_sharpes.std():.4f}")
    print(f"  Bootstrap: {boot_sharpes.mean():.4f} ± {boot_sharpes.std():.4f}")

# ============================================================
# 11. SYNTHETIC VALIDATION
# ============================================================
print("\n=== Synthetic Validation ===")
synth_strategies = {}

# Regime conditional on synthetic
synth_sig = signature_regime_features(synth_returns, window=60, level=3)
synth_regimes, synth_probs, _, _ = fit_hmm_regimes(synth_sig, n_states=4)
synth_regime_w = regime_conditional_portfolio(synth_returns, synth_regimes, synth_probs, lookback=60)
synth_regime_ret = backtest(synth_regime_w, synth_returns, cost_bps=cost_bps)
synth_strategies['Regime_Conditional'] = perf(synth_regime_ret, 'Regime_Conditional_Synthetic')

# VRP on synthetic
synth_vrp = vrp_signal(synth_returns, window=60)
synth_vrp_ret = backtest(synth_vrp, synth_returns, cost_bps=cost_bps)
synth_strategies['VRP_Only'] = perf(synth_vrp_ret, 'VRP_Only_Synthetic')

# Full combo on synthetic
synth_vrp_tilted = synth_regime_w * (1 + synth_vrp * 0.5)
synth_vrp_tilted = synth_vrp_tilted.clip(-1, 1)
synth_tail = tail_hedge_overlay(synth_vrp_tilted, synth_returns, hedge_ratio=0.3, tail_threshold=-0.015)
synth_full_ret = backtest(synth_tail, synth_returns, cost_bps=cost_bps)
synth_strategies['Full_Combo'] = perf(synth_full_ret, 'Full_Combo_Synthetic')

for name, res in synth_strategies.items():
    print(f"{name}: Sharpe={res['Sharpe']:.2f}, Ret={res['AnnRet%']:.2f}%, DD={res['MaxDD%']:.2f}%")

# ============================================================
# 12. REGIME CHARACTERIZATION
# ============================================================
regime_chars = []
aligned_returns_reg = returns.iloc[-len(regimes):]
for s in range(4):
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
# 13. SAVE OUTPUTS
# ============================================================
perf_df = pd.DataFrame(results).T
perf_df.to_csv('iter26_comprehensive_perf.csv')

val_df = pd.DataFrame(validation_results).T
val_df.to_csv('iter26_comprehensive_validation.csv')

synth_perf_df = pd.DataFrame(synth_strategies).T
synth_perf_df.to_csv('iter26_synthetic_validation.csv')

regime_df.to_csv('iter26_signature_regimes.csv')

# Best strategy returns
best_name = max(validation_results, key=lambda k: validation_results[k]['sharpe'])
best_pos = strategies[best_name]
best_ret = backtest(best_pos, aligned_returns, cost_bps=cost_bps)
best_ret.to_csv('iter26_best_returns.csv')
aligned_returns.to_csv('iter26_aligned_returns.csv')

# ============================================================
# 14. PLOTTING
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Equity curves
for name, pos in strategies.items():
    if isinstance(pos, pd.DataFrame):
        ret = backtest(pos, aligned_returns, cost_bps=cost_bps)
    else:
        ret = backtest(pos, aligned_returns['SPY'], cost_bps=cost_bps)
    cum = (1 + ret).cumprod()
    axes[0, 0].plot(cum.index, cum.values, label=name, linewidth=1)
axes[0, 0].set_title('Equity Curves (Net of Costs)')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Regime probabilities
sig_dates = sig_features.index
for s in range(4):
    axes[0, 1].plot(sig_dates, probs[:, s], label=f'Regime {s}', alpha=0.7)
axes[0, 1].set_title('HMM Regime Probabilities (Level-3 Signatures)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Validation comparison
val_names = list(validation_results.keys())
val_sharpes = [validation_results[n]['sharpe'] for n in val_names]
val_psrs = [validation_results[n]['psr'] for n in val_names]
axes[0, 2].bar(val_names, val_sharpes, alpha=0.7, label='Sharpe')
ax2 = axes[0, 2].twinx()
ax2.plot(val_names, val_psrs, 'ro-', label='PSR')
axes[0, 2].set_title('Strategy Validation: Sharpe & PSR')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(True, alpha=0.3)

# CV Sharpe distributions
for i, name in enumerate(val_names):
    # Recompute for plotting
    pos = strategies[name]
    n = len(pos)
    folds = purged_kfold_indices(n, n_splits=5, embargo_pct=0.01)
    cv_sharpes = []
    for train_idx, test_idx in folds:
        test_pos = pos.iloc[test_idx]
        test_returns = aligned_returns.iloc[test_idx]
        strat_test = backtest(test_pos, test_returns, cost_bps=cost_bps)
        cv_sharpes.append(sharpe(strat_test))
    axes[1, 0].boxplot(cv_sharpes, positions=[i], widths=0.6, labels=[name])
axes[1, 0].set_title('Purged CV Sharpe Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Bootstrap distributions (best strategy)
best_pos = strategies[best_name]
best_ret = backtest(best_pos, aligned_returns, cost_bps=cost_bps)
boot_sharpes = bootstrap_sharpe(best_ret, n_boot=1000)
axes[1, 1].hist(boot_sharpes, bins=50, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(sharpe(best_ret), color='red', linestyle='--', label=f'Observed: {sharpe(best_ret):.3f}')
axes[1, 1].axvline(boot_sharpes.mean(), color='green', linestyle='--', label=f'Mean: {boot_sharpes.mean():.3f}')
axes[1, 1].set_title(f'Bootstrap Sharpe: {best_name}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Regime profile
colors = ['green', 'blue', 'gray', 'red']
for _, row in regime_df.iterrows():
    axes[1, 2].scatter(row['vol']*100, row['mean_ret']*100, s=200, 
                       c=colors[int(row['regime'])], 
                       label=f"Regime {int(row['regime'])} ({row['frequency']:.1%})", alpha=0.7)
axes[1, 2].set_xlabel('Annualized Volatility (%)')
axes[1, 2].set_ylabel('Annualized Return (%)')
axes[1, 2].set_title('Regime Risk-Return Profile')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iter26_equity.png', dpi=150)
plt.close()

# Additional: Drawdown comparison
fig, ax = plt.subplots(figsize=(12, 6))
for name, pos in strategies.items():
    if isinstance(pos, pd.DataFrame):
        ret = backtest(pos, aligned_returns, cost_bps=cost_bps)
    else:
        ret = backtest(pos, aligned_returns['SPY'], cost_bps=cost_bps)
    cum = (1 + ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    ax.plot(dd.index, dd.values * 100, label=name, linewidth=1)
ax.set_title('Drawdown Comparison')
ax.set_ylabel('Drawdown (%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('iter26_drawdown.png', dpi=150)
plt.close()

print("\n=== Iteration #26 Complete ===")
print(f"Best Strategy: {best_name} (Sharpe: {validation_results[best_name]['sharpe']:.4f})")
print("Files saved: iter26_*.csv, iter26_*.png")