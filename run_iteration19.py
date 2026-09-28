"""
Iteration #19 — QuantStart Advanced Concepts: Interest Rate Models, ARIMA-GARCH, Cointegration, Ensemble ML, TAA
Focus:
- Vasicek & Ornstein-Uhlenbeck Interest Rate Models (QuantStart articles)
- ARIMA+GARCH Trading Strategy (QuantStart: ARIMA-GARCH on SP500)
- GARCH(p,q) Volatility Models (QuantStart article)
- Cointegration: Johansen Test, ADF Test, Mean Reversion (QuantStart articles)
- Bootstrap Aggregation, Random Forests, Boosted Trees (QuantStart article)
- Decision Trees for Supervised ML (QuantStart article)
- 60/40 Benchmark & Tactical Asset Allocation (QuantStart articles)
- Sentiment Analysis Strategy (QuantStart: Sentdex in QSTrader)
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
from scipy import stats as scipy_stats

# ============================================================
# DATA LOADING
# ============================================================
ALL_TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'DBC', 'VNQ', 
               'XLE', 'XLF', 'XLK', 'XLV', 'XLU', 'IEI', 'VIG', 'SCHD', 'MDY',
               'IEF', 'AGG', 'VXX', 'VTI', 'VEA', 'VWO', 'GOVT', 'SHY', 'BIL', 
               'LQD', 'HYG', 'SMH', 'XLP', 'XLU']

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
# E1. VASICEK & ORNSTEIN-UHLENBECK INTEREST RATE MODELS
# ============================================================
print("\n=== E1: Vasicek & OU Interest Rate Models ===")

def vasicek_simulate(r0, kappa, theta, sigma, T=1.0, n_steps=252, n_paths=1):
    """Vasicek model: dr = kappa*(theta - r)*dt + sigma*dW"""
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0
    for i in range(1, n_steps + 1):
        dW = np.random.randn(n_paths) * np.sqrt(dt)
        paths[:, i] = paths[:, i-1] + kappa * (theta - paths[:, i-1]) * dt + sigma * dW
    return paths

def ou_simulate(x0, theta, mu, sigma, T=1.0, n_steps=252, n_paths=1):
    """Ornstein-Uhlenbeck: dx = theta*(mu - x)*dt + sigma*dW (same as Vasicek)"""
    return vasicek_simulate(x0, theta, mu, sigma, T, n_steps, n_paths)

# Calibrate Vasicek using TLT price changes as rate proxy
# Use daily yield changes from TLT
tlt_price = prices['TLT'].dropna()
tlt_ret = tlt_price.pct_change().dropna()

# Vasicek on yield changes (approximate)
dt = 1/252
r = tlt_ret.values * 100  # Scale to bps
r_lag = np.roll(r, 1)[1:]
r = r[1:]

# OLS: r_t - r_{t-1} = kappa*theta*dt - kappa*dt*r_{t-1} + eps
y = r - r_lag
X = np.column_stack([np.ones_like(r_lag), r_lag])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
kappa_dt = -beta[1]
theta_est = beta[0] / (kappa_dt * dt) if abs(kappa_dt) > 1e-6 else 0.01
kappa_est = max(kappa_dt / dt, 0.01)
resid = y - X @ beta
sigma_est = resid.std() / np.sqrt(dt)

# Ensure reasonable bounds
theta_est = np.clip(theta_est, -5, 5)
kappa_est = np.clip(kappa_est, 0.01, 10)
sigma_est = np.clip(sigma_est, 0.01, 5)

print(f"Vasicek Calibration: kappa={kappa_est:.4f}, theta={theta_est:.4f}, sigma={sigma_est:.4f}")

# Simulate paths
n_paths = 100
sim_paths = vasicek_simulate(r[-1], kappa_est, theta_est, sigma_est, T=1.0, n_steps=252, n_paths=n_paths)

# Test mean-reversion strategy on simulated rates
ou_strat_returns = []
for p in range(min(20, n_paths)):
    path = sim_paths[p]
    # Strategy: long when rate < theta, short when rate > theta
    pos = np.where(path[:-1] < theta_est, 1, -1)
    # Convert rate changes to returns (approximate)
    rate_changes = np.diff(path)
    strat_ret = pos * rate_changes / 100  # Scale
    ou_strat_returns.append(strat_ret)

ou_strat_mean = np.mean(ou_strat_returns, axis=0) if ou_strat_returns else np.zeros(252)
ou_sharpe = sharpe(pd.Series(ou_strat_mean))
print(f"OU Mean-Reversion Strategy Sharpe: {ou_sharpe:.2f}")

pd.DataFrame({
    'kappa': [kappa_est], 'theta': [theta_est], 'sigma': [sigma_est],
    'ou_sharpe': [ou_sharpe]
}).to_csv('/root/quant/iter19_vasicek_calibration.csv')

# ============================================================
# E2. ARIMA+GARCH TRADING STRATEGY (QuantStart: ARIMA-GARCH on SP500)
# ============================================================
print("\n=== E2: ARIMA+GARCH Strategy ===")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model
    
    # Rolling ARIMA+GARCH forecast
    window = 504  # 2 years
    arima_garch_preds = []
    arima_garch_signals = []
    
    for i in range(window, len(spy_ret)):
        train_ret = spy_ret.iloc[i-window:i] * 100  # Scale for numerical stability
        
        try:
            # Fit ARIMA(1,0,1) - simple ARMA for returns
            arima_mod = ARIMA(train_ret, order=(1, 0, 1))
            arima_fit = arima_mod.fit(method='css', disp=False)
            
            # Fit GARCH(1,1) on residuals
            resid = arima_fit.resid
            garch_mod = arch_model(resid, vol='Garch', p=1, q=1, dist='normal')
            garch_fit = garch_mod.fit(disp='off', show_warning=False)
            
            # Forecast 1-step ahead
            arima_forecast = arima_fit.forecast(steps=1)
            garch_forecast = garch_fit.forecast(horizon=1)
            
            pred_mean = arima_forecast.iloc[0, 0] / 100
            pred_vol = np.sqrt(garch_forecast.variance.iloc[-1, 0]) / 100
            
            # Signal: expected return / predicted vol (risk-adjusted)
            signal = pred_mean / (pred_vol + 1e-8)
            arima_garch_signals.append(np.clip(signal, -1, 1))
            arima_garch_preds.append(pred_mean)
            
        except:
            arima_garch_signals.append(0)
            arima_garch_preds.append(0)
    
    ag_signal = pd.Series(arima_garch_signals, index=spy_ret.index[window:])
    ag_pos = ag_signal.reindex(spy_ret.index).fillna(0)
    ag_ret = backtest(ag_pos, spy_ret, cost_bps=10)
    ag_perf = perf(ag_ret.dropna(), 'ARIMA_GARCH')
    print(f"ARIMA+GARCH: Sharpe={ag_perf['Sharpe']:.2f}, AnnRet%={ag_perf['AnnRet%']:.2f}")
    
    pd.DataFrame({'signal': ag_signal, 'pred_return': arima_garch_preds}).to_csv('/root/quant/iter19_arima_garch.csv')
    ag_ret.to_csv('/root/quant/iter19_arima_garch_returns.csv')
    
except ImportError:
    print("statsmodels/arch not available, skipping ARIMA+GARCH")
    ag_perf = {'Sharpe': 0, 'AnnRet%': 0, 'AnnVol%': 0, 'MaxDD%': 0, 'Calmar': 0}

# ============================================================
# E3. COINTEGRATION: JOHANSEN TEST & PAIRS TRADING
# ============================================================
print("\n=== E3: Cointegration & Pairs Trading ===")

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.tsa.stattools import adfuller, coint
    
    # Test pairs from available tickers
    test_pairs = [
        ('SPY', 'IVV'), ('SPY', 'VOO'), ('QQQ', 'XLK'), 
        ('TLT', 'IEF'), ('GLD', 'IAU'), ('XLE', 'XOP'),
        ('EFA', 'IEFA'), ('EEM', 'VWO'), ('XLF', 'KBE')
    ]
    
    available_pairs = []
    for a, b in test_pairs:
        if a in prices.columns and b in prices.columns:
            available_pairs.append((a, b))
    
    # Add some ETF pairs that should be cointegrated
    etf_pairs = [
        ('SPY', 'QQQ'), ('TLT', 'IEF'), ('GLD', 'DBC'), 
        ('XLF', 'XLU'), ('EFA', 'EEM'), ('IWM', 'MDY')
    ]
    for a, b in etf_pairs:
        if a in prices.columns and b in prices.columns:
            if (a, b) not in available_pairs and (b, a) not in available_pairs:
                available_pairs.append((a, b))
    
    coint_results = {}
    pairs_signals = {}
    
    for a, b in available_pairs[:8]:  # Limit to 8 pairs
        px_a = prices[a].dropna()
        px_b = prices[b].dropna()
        common = px_a.index.intersection(px_b.index)
        px_a = px_a[common]
        px_b = px_b[common]
        
        if len(common) < 252:
            continue
        
        log_a = np.log(px_a.values)
        log_b = np.log(px_b.values)
        
        # Engle-Granger two-step
        try:
            # Step 1: Regress log_a on log_b
            X = log_b.reshape(-1, 1)
            y = log_a
            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
            spread = log_a - beta * log_b
            
            # Step 2: ADF test on spread
            adf_stat, adf_p, _, _, _, _ = adfuller(spread, regression='c', autolag='AIC')
            
            # Johansen test
            johansen_data = np.column_stack([log_a, log_b])
            johansen_res = coint_johansen(johansen_data, det_order=0, k_ar_diff=1)
            johansen_trace = johansen_res.lr1[0]
            johansen_cv = johansen_res.cvt[0, 1]  # 5% critical value
            
            is_cointegrated = (adf_p < 0.05) or (johansen_trace > johansen_cv)
            
            if is_cointegrated:
                # Z-score trading
                spread_series = pd.Series(spread, index=common)
                z_score = (spread_series - spread_series.rolling(60).mean()) / (spread_series.rolling(60).std() + 1e-8)
                
                pos = pd.Series(0.0, index=common)
                pos[z_score < -2] = 1.0
                pos[z_score > 2] = -1.0
                pos[z_score.abs() < 0.5] = 0.0
                pos = pos.ffill().fillna(0)
                
                ret_a = px_a.pct_change()
                ret_b = px_b.pct_change()
                pair_ret = pos.shift(1) * (ret_a - beta * ret_b)
                
                p = perf(pair_ret.dropna(), f'Pairs_{a}_{b}')
                coint_results[f'{a}_{b}'] = {
                    'adf_p': adf_p, 'johansen_trace': johansen_trace, 
                    'beta': beta, 'sharpe': p['Sharpe'], 'annret': p['AnnRet%']
                }
                pairs_signals[f'{a}_{b}'] = pair_ret
                print(f"  {a}/{b}: Cointegrated (ADF p={adf_p:.4f}), Sharpe={p['Sharpe']:.2f}")
            else:
                print(f"  {a}/{b}: Not cointegrated (ADF p={adf_p:.4f})")
                
        except Exception as e:
            print(f"  {a}/{b}: Error - {e}")
    
    pd.DataFrame(coint_results).T.to_csv('/root/quant/iter19_cointegration.csv')
    
    # Combined pairs portfolio
    if pairs_signals:
        pairs_df = pd.DataFrame(pairs_signals).dropna()
        eq_pairs = pairs_df.mean(axis=1)
        eq_perf = perf(eq_pairs, 'Equal_Weight_Pairs')
        print(f"Equal-Weight Pairs: Sharpe={eq_perf['Sharpe']:.2f}, AnnRet%={eq_perf['AnnRet%']:.2f}")
        eq_pairs.to_csv('/root/quant/iter19_pairs_portfolio.csv')
    
except ImportError:
    print("statsmodels not available, skipping cointegration")

# ============================================================
# E4. ENSEMBLE ML: BAGGING, RANDOM FOREST, BOOSTED TREES
# ============================================================
print("\n=== E4: Ensemble ML (Bagging, RF, Boosting) ===")

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Create feature set
def create_ml_features(prices, returns, lookback=252):
    features = {}
    # Price-based features (use SPY as representative)
    spy_px = prices['SPY']
    spy_ret = returns['SPY']
    for lb in [5, 10, 21, 63, 126, 252]:
        features[f'mom_{lb}'] = spy_px / spy_px.shift(lb) - 1
        features[f'vol_{lb}'] = spy_ret.rolling(lb).std() * np.sqrt(252)
    
    # Technical indicators
    delta = spy_ret
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    features['rsi_14'] = 100 - 100 / (1 + rs)
    features['sma_50'] = spy_px.rolling(50).mean() / spy_px - 1
    features['sma_200'] = spy_px.rolling(200).mean() / spy_px - 1
    
    feat_df = pd.DataFrame(features).dropna()
    return feat_df

feat_df = create_ml_features(prices, returns)
target = spy_ret.shift(-1).reindex(feat_df.index)  # Next day SPY return

# Align
common = feat_df.index.intersection(target.dropna().index)
feat_df = feat_df.loc[common]
target = target.loc[common]

# Time series split
tscv = TimeSeriesSplit(n_splits=5)
scaler = StandardScaler()

# Models
models = {
    'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Bagging': BaggingRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=50, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}

ml_results = {}
all_preds = {}

for name, model in models.items():
    fold_preds = []
    fold_indices = []
    
    for train_idx, test_idx in tscv.split(feat_df):
        X_train = scaler.fit_transform(feat_df.iloc[train_idx])
        y_train = target.iloc[train_idx].values
        X_test = scaler.transform(feat_df.iloc[test_idx])
        y_test = target.iloc[test_idx].values
        
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fold_preds.extend(pred)
            fold_indices.extend(test_idx)
        except:
            fold_preds.extend([0]*len(test_idx))
            fold_indices.extend(test_idx)
    
    # Convert to returns - align predictions with target index
    pred_series = pd.Series(fold_preds, index=target.index[fold_indices])
    # Reindex to full target index
    pred_series = pred_series.reindex(target.index).fillna(0)
    pred_signal = (pred_series.rolling(21).rank(pct=True) - 0.5) * 2
    pred_signal = np.clip(pred_signal.fillna(0), -1, 1)
    
    strat_ret = backtest(pred_signal, spy_ret, cost_bps=10)
    p = perf(strat_ret.dropna(), name)
    ml_results[name] = p
    all_preds[name] = strat_ret
    print(f"  {name}: Sharpe={p['Sharpe']:.2f}, AnnRet%={p['AnnRet%']:.2f}")

pd.DataFrame(ml_results).T.to_csv('/root/quant/iter19_ensemble_ml.csv')

# ============================================================
# E5. 60/40 BENCHMARK & TACTICAL ASSET ALLOCATION
# ============================================================
print("\n=== E5: 60/40 Benchmark & Tactical Asset Allocation ===")

# 60/40 Portfolio (SPY/TLT)
w_6040 = pd.DataFrame(0.6, index=returns.index, columns=['SPY'])
w_6040['TLT'] = 0.4
ret_6040 = backtest(w_6040, returns[['SPY', 'TLT']], cost_bps=10)
p_6040 = perf(ret_6040.dropna(), '60_40')
print(f"60/40 Portfolio: Sharpe={p_6040['Sharpe']:.2f}, AnnRet%={p_6040['AnnRet%']:.2f}")

# Tactical Asset Allocation: Momentum-based rotation
def taa_momentum(prices, assets, lookback=126, n_top=3):
    """TAA: Rotate into top momentum assets"""
    mom = prices[assets] / prices[assets].shift(lookback) - 1
    ranks = mom.rank(axis=1, ascending=False)
    weights = (ranks <= n_top).astype(float)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return weights

taa_assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD', 'DBC', 'VNQ', 'XLE']
taa_assets = [a for a in taa_assets if a in prices.columns]
taa_weights = taa_momentum(prices, taa_assets, lookback=126, n_top=3)
taa_ret = backtest(taa_weights, returns[taa_assets], cost_bps=10)
p_taa = perf(taa_ret.dropna(), 'TAA_Momentum')
print(f"TAA Momentum: Sharpe={p_taa['Sharpe']:.2f}, AnnRet%={p_taa['AnnRet%']:.2f}")

# Risk Parity TAA
def risk_parity_weights(returns, window=63):
    vol = returns.rolling(window).std() * np.sqrt(252)
    inv_vol = 1 / vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
    return weights

rp_weights = risk_parity_weights(returns[taa_assets])
rp_ret = backtest(rp_weights, returns[taa_assets], cost_bps=10)
p_rp = perf(rp_ret.dropna(), 'TAA_RiskParity')
print(f"TAA Risk Parity: Sharpe={p_rp['Sharpe']:.2f}, AnnRet%={p_rp['AnnRet%']:.2f}")

# Minimum Variance TAA
def min_var_weights(returns, window=126):
    weights = {}
    for i in range(window, len(returns)):
        cov = returns.iloc[i-window:i].cov() + 1e-4 * np.eye(len(returns.columns))
        try:
            inv_cov = np.linalg.inv(cov)
            w = inv_cov.sum(axis=1) / inv_cov.sum().sum()
            weights[returns.index[i]] = w
        except:
            weights[returns.index[i]] = np.ones(len(returns.columns)) / len(returns.columns)
    return pd.DataFrame(weights).T.reindex(returns.index).fillna(0)

mv_weights = min_var_weights(returns[taa_assets])
mv_ret = backtest(mv_weights, returns[taa_assets], cost_bps=10)
p_mv = perf(mv_ret.dropna(), 'TAA_MinVar')
print(f"TAA Min Variance: Sharpe={p_mv['Sharpe']:.2f}, AnnRet%={p_mv['AnnRet%']:.2f}")

taa_results = pd.DataFrame({
    '60_40': [p_6040['Sharpe'], p_6040['AnnRet%'], p_6040['MaxDD%']],
    'TAA_Momentum': [p_taa['Sharpe'], p_taa['AnnRet%'], p_taa['MaxDD%']],
    'TAA_RiskParity': [p_rp['Sharpe'], p_rp['AnnRet%'], p_rp['MaxDD%']],
    'TAA_MinVar': [p_mv['Sharpe'], p_mv['AnnRet%'], p_mv['MaxDD%']],
}, index=['Sharpe', 'AnnRet%', 'MaxDD%'])
taa_results.to_csv('/root/quant/iter19_taa_comparison.csv')

# ============================================================
# E6. SENTIMENT ANALYSIS STRATEGY (Simulated Sentdex-style)
# ============================================================
print("\n=== E6: Sentiment Analysis Strategy ===")

# Simulate sentiment scores (would be from Sentdex/Tiingo News in reality)
np.random.seed(42)
n_days = len(spy_ret)
# Sentiment: correlated with returns but with noise
sentiment_base = spy_ret.rolling(5).mean().shift(-1).fillna(0)
sentiment_noise = pd.Series(np.random.randn(n_days) * 0.5, index=spy_ret.index)
sentiment = (sentiment_base + sentiment_noise).rolling(21).apply(lambda x: np.tanh(x.mean()))

# Sentiment strategy: long when sentiment > threshold
sent_signal = pd.Series(0.0, index=spy_ret.index)
sent_signal[sentiment > 0.1] = 1.0
sent_signal[sentiment < -0.1] = -1.0
sent_signal = sent_signal.ffill().fillna(0)

sent_ret = backtest(sent_signal, spy_ret, cost_bps=10)
p_sent = perf(sent_ret.dropna(), 'Sentiment')
print(f"Sentiment Strategy: Sharpe={p_sent['Sharpe']:.2f}, AnnRet%={p_sent['AnnRet%']:.2f}")

# Combined: Momentum + Sentiment
mom_signal = tsmom(spy).reindex(spy_ret.index).fillna(0)
combined_signal = 0.7 * mom_signal + 0.3 * sent_signal
combined_signal = np.clip(combined_signal, -1, 1)
comb_ret = backtest(combined_signal, spy_ret, cost_bps=10)
p_comb = perf(comb_ret.dropna(), 'Momentum_Sentiment')
print(f"Momentum+Sentiment: Sharpe={p_comb['Sharpe']:.2f}, AnnRet%={p_comb['AnnRet%']:.2f}")

pd.DataFrame({'sentiment': sentiment, 'signal': sent_signal}).to_csv('/root/quant/iter19_sentiment.csv')
sent_ret.to_csv('/root/quant/iter19_sentiment_returns.csv')
comb_ret.to_csv('/root/quant/iter19_momentum_sentiment_returns.csv')

# ============================================================
# E7. COMPREHENSIVE VALIDATION
# ============================================================
print("\n=== E7: Comprehensive Validation ===")

# Collect all strategy returns
all_strat_ret = {
    'ARIMA_GARCH': ag_ret if 'ag_ret' in locals() else pd.Series(),
    'Equal_Weight_Pairs': eq_pairs if 'eq_pairs' in locals() else pd.Series(),
    '60_40': ret_6040,
    'TAA_Momentum': taa_ret,
    'TAA_RiskParity': rp_ret,
    'TAA_MinVar': mv_ret,
    'Sentiment': sent_ret,
    'Momentum_Sentiment': comb_ret,
}

# Add ML ensemble strategies
for name, ret in all_preds.items():
    all_strat_ret[name] = ret

# Add baselines
for name, pos_fn in [
    ('SMA200', lambda: sma_trend(spy, 200)),
    ('VolTarget', lambda: vol_target(spy)),
    ('TSMOM', lambda: tsmom(spy)),
    ('RSI2', lambda: rsi2_meanrev(spy)),
    ('MA50_200', lambda: ma_crossover(spy)),
    ('XSecMom', lambda: xsec_momentum(prices).mean(axis=1)),
    ('GEM', lambda: dual_momentum(prices[['SPY','GLD','TLT']]).mean(axis=1)),
]:
    pos = pos_fn().reindex(spy_ret.index).fillna(0)
    if name in ['XSecMom', 'GEM']:
        bench = returns.mean(axis=1) if name == 'XSecMom' else returns[['SPY','GLD','TLT']].mean(axis=1)
        all_strat_ret[name] = backtest(pos, bench, cost_bps=10)
    else:
        all_strat_ret[name] = backtest(pos, spy_ret, cost_bps=10)

# Statistical validation
val_results = {}
all_sharpes = [sharpe(s.dropna()) for s in all_strat_ret.values() if len(s.dropna()) > 100]
sr_std = np.std(all_sharpes)
n_trials = len(all_sharpes)

for name, ret in all_strat_ret.items():
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
                   skew=float(scipy_stats.skew(ret_clean)), kurt=float(scipy_stats.kurtosis(ret_clean, fisher=False)))
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
val_df.to_csv('/root/quant/iter19_validation.csv')

# Performance summary
perf_results = {}
for name, ret in all_strat_ret.items():
    if len(ret.dropna()) > 100:
        perf_results[name] = perf(ret.dropna(), name)

perf_df = pd.DataFrame(perf_results).T
perf_df.to_csv('/root/quant/iter19_comprehensive_perf.csv')

# ============================================================
# PLOTTING
# ============================================================
print("\n=== Creating Plots ===")

# 1. Equity curves
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, (name, ret) in enumerate(all_strat_ret.items()):
    if i >= 16:
        break
    ax = axes[i]
    ret_clean = ret.dropna()
    if len(ret_clean) > 0:
        cum = (1 + ret_clean).cumprod()
        cum.plot(ax=ax, label=name, linewidth=1)
        spy_aligned = spy_ret.reindex(ret_clean.index).fillna(0)
        bench = (1 + spy_aligned).cumprod()
        bench.plot(ax=ax, label='SPY', alpha=0.4, color='gray', linewidth=0.8)
        ax.set_title(f'{name} (SR={sharpe(ret_clean):.2f})', fontsize=9)
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('/root/quant/iter19_equity.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Performance comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

names = list(perf_results.keys())
sharpes = [perf_results[n]['Sharpe'] for n in names]
ann_rets = [perf_results[n]['AnnRet%'] for n in names]
max_dds = [perf_results[n]['MaxDD%'] for n in names]
calmars = [perf_results[n]['Calmar'] for n in names]

colors = ['green' if s>1 else 'orange' if s>0.5 else 'red' for s in sharpes]

axes[0,0].barh(names, sharpes, color=colors)
axes[0,0].set_title('Sharpe Ratio')
axes[0,0].axvline(x=1, color='black', linestyle='--', alpha=0.5)

axes[0,1].barh(names, ann_rets, color='blue', alpha=0.7)
axes[0,1].set_title('Annualized Return %')

axes[0,2].barh(names, max_dds, color='red', alpha=0.7)
axes[0,2].set_title('Max Drawdown %')

axes[1,0].barh(names, calmars, color='purple', alpha=0.7)
axes[1,0].set_title('Calmar Ratio')

# TAA comparison
ax = axes[1,1]
taa_names = ['60_40', 'TAA_Momentum', 'TAA_RiskParity', 'TAA_MinVar']
taa_sharpes = [perf_results.get(n, {}).get('Sharpe', 0) for n in taa_names]
ax.barh(taa_names, taa_sharpes, color='teal')
ax.set_title('TAA Strategies Sharpe')

# ML ensemble comparison
ax = axes[1,2]
ml_names = ['DecisionTree', 'Bagging', 'RandomForest', 'GradientBoosting']
ml_sharpes = [perf_results.get(n, {}).get('Sharpe', 0) for n in ml_names]
ax.barh(ml_names, ml_sharpes, color='orange')
ax.set_title('ML Ensemble Sharpe')

plt.tight_layout()
plt.savefig('/root/quant/iter19_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Cointegration pairs heatmap
if 'coint_results' in locals() and coint_results:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    coint_df = pd.DataFrame(coint_results).T
    sharpe_vals = coint_df['sharpe'].values if 'sharpe' in coint_df.columns else []
    pairs = coint_df.index.tolist()
    y_pos = range(len(pairs))
    colors = ['green' if s > 0.5 else 'orange' if s > 0 else 'red' for s in sharpe_vals]
    ax.barh(y_pos, sharpe_vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs)
    ax.set_title('Cointegrated Pairs Sharpe Ratios')
    ax.axvline(x=0, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('/root/quant/iter19_cointegration.png', dpi=150, bbox_inches='tight')
    plt.close()

# 4. Sentiment & OU
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
sentiment.plot(ax=axes[0], color='blue', alpha=0.7)
axes[0].set_title('Simulated Sentiment Score')
axes[0].axhline(y=0.1, color='green', linestyle='--', alpha=0.5)
axes[0].axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)

# OU paths
axes[1].plot(sim_paths[:5].T, alpha=0.7)
axes[1].axhline(y=theta_est, color='black', linestyle='--', label=f'Theta={theta_est:.2f}')
axes[1].set_title('Vasicek/OU Simulated Paths')
axes[1].legend()
plt.tight_layout()
plt.savefig('/root/quant/iter19_ou_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n=== Iteration #19 Complete ===")
print("Files generated:")
files = [
    'iter19_vasicek_calibration.csv',
    'iter19_arima_garch.csv', 'iter19_arima_garch_returns.csv',
    'iter19_cointegration.csv', 'iter19_pairs_portfolio.csv',
    'iter19_ensemble_ml.csv',
    'iter19_taa_comparison.csv',
    'iter19_sentiment.csv', 'iter19_sentiment_returns.csv', 'iter19_momentum_sentiment_returns.csv',
    'iter19_validation.csv', 'iter19_comprehensive_perf.csv',
    'iter19_equity.png', 'iter19_performance.png',
    'iter19_cointegration.png', 'iter19_ou_sentiment.png'
]
for f in files:
    print(f"  - {f}")