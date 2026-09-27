"""Iteration #8 — QuantStart: Deep Learning, Bias-Variance Tradeoff, Static Benchmarks, Purged CV for ML
Explores: Neural nets for regime/return prediction, Bias-variance in param selection,
Comprehensive static benchmarks (Permanent Portfolio, etc.), Purged CV for time-series ML.
Outputs: iter8_*.csv, iter8_*.png, updated REPORT.md
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/root/quant')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import load, backtest, sharpe, max_dd, perf
from stats import nw_tstat, report_validation
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

COST = 10
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'AGG',
           'VNQ', 'DBC', 'XLP', 'XLU', 'XLE', 'XLF', 'IEI', 'VIG', 'SCHD',
           'MDY', 'XLK', 'XLV', 'VTI', 'VEA', 'VWO', 'GOVT', 'SHY', 'BIL',
           'LQD', 'HYG', 'SMH']
px = pd.DataFrame({t: load(t)['Close'] for t in TICKERS}).dropna()
rets = px.pct_change().dropna()

# ======================================================================
# Helper: rebalanced portfolio
# ======================================================================
def rebalanced_portfolio(weights_dict):
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    total = sum(weights_dict.values())
    norm_w = {k: v/total for k, v in weights_dict.items()}
    months = px.resample('ME').last().index
    for d in months:
        if d in px.index:
            idx = px.index.get_loc(d)
            if idx + 1 < len(px.index):
                for k, v in norm_w.items():
                    if k in w.columns:
                        w.loc[px.index[idx+1], k] = v
    w = w.ffill().fillna(0)
    return w

# ======================================================================
# Base strategies
# ======================================================================
# SMA200
w_sma = (px[['SPY']] > px[['SPY']].rolling(200).mean()).astype(float).reindex(columns=px.columns).fillna(0)

# GEM
mom = px[['SPY','EFA','AGG']] / px[['SPY','EFA','AGG']].shift(126) - 1
w_gem = pd.DataFrame(0.0, index=px.index, columns=px.columns)
valid = mom.dropna().index
for t in valid:
    if mom.loc[t, 'SPY'] > 0:
        w_gem.loc[t, mom.loc[t, ['SPY','EFA']].idxmax()] = 1.0
    else:
        w_gem.loc[t, 'AGG'] = 1.0

# XSec Momentum
RISK = ['SPY','QQQ','IWM','EFA','EEM','VNQ','DBC','XLK','XLF','XLE','XLV','MDY','SMH','VIG']
RISK = [t for t in RISK if t in px.columns]
mom_xs = px[RISK] / px[RISK].shift(252) - 1
ranks = mom_xs.rank(axis=1, ascending=False)
w_xs = (ranks <= 5).astype(float)
w_xs = w_xs.div(w_xs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).reindex(columns=px.columns).fillna(0)

# ======================================================================
# E1. Static Allocation Benchmarks (from QuantStart static_backtest article)
# ======================================================================
# 60/40
w_6040 = rebalanced_portfolio({'SPY': 0.6, 'AGG': 0.4})

# All Weather (Dalio)
w_aw = rebalanced_portfolio({'VTI': 0.3, 'TLT': 0.4, 'IEI': 0.15, 'GLD': 0.075, 'DBC': 0.075})
# Map to our tickers
aw_map = {'VTI': 'SPY', 'TLT': 'TLT', 'IEI': 'IEI', 'GLD': 'GLD', 'DBC': 'DBC'}
w_aw_mapped = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for k, v in {'SPY': 0.3, 'TLT': 0.4, 'IEI': 0.15, 'GLD': 0.075, 'DBC': 0.075}.items():
    if k in px.columns:
        months = px.resample('ME').last().index
        for d in months:
            if d in px.index:
                idx = px.index.get_loc(d)
                if idx + 1 < len(px.index):
                    w_aw_mapped.loc[px.index[idx+1], k] = v
w_aw_mapped = w_aw_mapped.ffill().fillna(0)

# Permanent Portfolio (Browne)
w_pp = rebalanced_portfolio({'SPY': 0.25, 'TLT': 0.25, 'IEF': 0.25, 'GLD': 0.25})

# Golden Butterfly
w_gb = rebalanced_portfolio({'SPY': 0.2, 'IEF': 0.2, 'TLT': 0.2, 'GLD': 0.2, 'DBC': 0.2})

# Risk Parity (equal vol) on core assets
core_assets = ['SPY', 'TLT', 'IEF', 'GLD', 'DBC']
core_assets = [c for c in core_assets if c in px.columns]
core_rets = rets[core_assets]
core_vol = core_rets.rolling(63).std() * np.sqrt(252)
inv_vol = 1.0 / core_vol.replace(0, np.nan)
w_rp_static = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
w_rp_static = w_rp_static.reindex(columns=px.columns).fillna(0).shift(1).fillna(0)

# Global Market Portfolio (approximate)
w_gmp = rebalanced_portfolio({'VTI': 0.42, 'VEA': 0.18, 'VWO': 0.10, 'AGG': 0.18, 'VNQ': 0.04, 'DBC': 0.04, 'GLD': 0.04})
gmp_map = {'VTI': 'SPY', 'VEA': 'EFA', 'VWO': 'EEM', 'AGG': 'AGG', 'VNQ': 'VNQ', 'DBC': 'DBC', 'GLD': 'GLD'}
w_gmp_mapped = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for k, v in {'SPY': 0.42, 'EFA': 0.18, 'EEM': 0.10, 'AGG': 0.18, 'VNQ': 0.04, 'DBC': 0.04, 'GLD': 0.04}.items():
    if k in px.columns:
        months = px.resample('ME').last().index
        for d in months:
            if d in px.index:
                idx = px.index.get_loc(d)
                if idx + 1 < len(px.index):
                    w_gmp_mapped.loc[px.index[idx+1], k] = v
w_gmp_mapped = w_gmp_mapped.ffill().fillna(0)

# Test all static benchmarks
static_strats = {
    '60/40': w_6040,
    'All Weather': w_aw_mapped,
    'Permanent Portfolio': w_pp,
    'Golden Butterfly': w_gb,
    'Risk Parity Static': w_rp_static,
    'Global Market Portfolio': w_gmp_mapped,
}

static_results = {}
for name, w in static_strats.items():
    r = backtest(w, rets, COST).dropna()
    static_results[name] = {
        'Sharpe': round(sharpe(r), 2),
        'AnnRet%': round(252*r.mean()*100, 2),
        'MaxDD%': round(100*max_dd((1+r).cumprod()), 2),
        'Calmar': round(252*r.mean() / abs(max_dd((1+r).cumprod())), 2) if max_dd((1+r).cumprod()) < 0 else 0
    }

static_tbl = pd.DataFrame(static_results).T
static_tbl.to_csv('iter8_static_benchmarks.csv')
print('E1 Static Benchmarks:\n', static_tbl.to_string())

# ======================================================================
# E2. Deep Learning for Regime Prediction (from "What is Deep Learning?")
# ======================================================================
# Features: technical indicators, macro proxies
def create_ml_features(px, rets, lookback=63):
    """Create features for ML models."""
    feat = pd.DataFrame(index=px.index)
    
    # Price-based features
    for col in ['SPY', 'TLT', 'GLD', 'EFA']:
        if col in px.columns:
            # Returns at multiple horizons
            for h in [1, 5, 21, 63, 126, 252]:
                feat[f'{col}_ret_{h}'] = px[col].pct_change(h)
            # Moving averages
            for w in [20, 50, 100, 200]:
                feat[f'{col}_ma_{w}'] = px[col] / px[col].rolling(w).mean() - 1
            # Volatility
            for w in [21, 63, 126]:
                feat[f'{col}_vol_{w}'] = rets[col].rolling(w).std() * np.sqrt(252)
            # RSI
            delta = px[col].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            feat[f'{col}_rsi'] = 100 - 100/(1+rs)
    
    # Cross-asset features
    if 'SPY' in px.columns and 'TLT' in px.columns:
        feat['spy_tlt_ratio'] = px['SPY'] / px['TLT']
        feat['spy_gld_ratio'] = px['SPY'] / px['GLD'] if 'GLD' in px.columns else 0
    
    # VIX proxy (SPY 21d vol)
    feat['vix_proxy'] = rets['SPY'].rolling(21).std() * np.sqrt(252) * 100
    
    # Yield curve proxy (TLT vs IEF)
    if 'TLT' in px.columns and 'IEF' in px.columns:
        feat['term_spread'] = px['TLT'].pct_change(21) - px['IEF'].pct_change(21)
    
    return feat.dropna()

print("E2 Creating ML features...")
features = create_ml_features(px, rets)

# Target: next 21-day SPY return (regression) or direction (classification)
target_ret = px['SPY'].pct_change(21).shift(-21)
target_dir = (target_ret > 0).astype(int)

# Align
common_idx = features.index.intersection(target_ret.dropna().index)
X = features.loc[common_idx]
y_ret = target_ret.loc[common_idx]
y_dir = target_dir.loc[common_idx]

# Split: 70% train, 30% test (time-ordered)
split = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_ret_train, y_ret_test = y_ret.iloc[:split], y_ret.iloc[split:]
y_dir_train, y_dir_test = y_dir.iloc[:split], y_dir.iloc[split:]

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Models to compare (bias-variance tradeoff)
models = {
    'Linear': LinearRegression(),
    'Ridge(1)': Ridge(alpha=1.0),
    'Ridge(10)': Ridge(alpha=10.0),
    'Lasso(0.1)': Lasso(alpha=0.1),
    'RF(100)': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    'GBM(100)': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    'MLP(32)': MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42, early_stopping=True),
    'MLP(64,32)': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42, early_stopping=True),
    'MLP(128,64,32)': MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=500, random_state=42, early_stopping=True),
}

ml_results = {}
for name, model in models.items():
    try:
        model.fit(X_train_s, y_ret_train)
        pred = model.predict(X_test_s)
        # MSE
        mse = np.mean((pred - y_ret_test)**2)
        # Directional accuracy
        dir_acc = np.mean((pred > 0) == (y_ret_test > 0))
        # Correlation
        corr = np.corrcoef(pred, y_ret_test)[0,1]
        
        # Build strategy: go long when predicted return > 0
        w_ml = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        pred_series = pd.Series(pred, index=X_test.index)
        for t in pred_series[pred_series > 0].index:
            w_ml.loc[t, 'SPY'] = 1.0
        w_ml = w_ml.ffill().fillna(0)
        
        r_ml = backtest(w_ml, rets, COST).dropna()
        ml_results[name] = {
            'Test MSE': round(mse * 1e6, 2),
            'Dir Acc': round(dir_acc, 3),
            'Correlation': round(corr, 3),
            'Strategy Sharpe': round(sharpe(r_ml), 2),
            'Strategy AnnRet%': round(252*r_ml.mean()*100, 2),
            'Strategy MaxDD%': round(100*max_dd((1+r_ml).cumprod()), 2),
        }
    except Exception as e:
        ml_results[name] = {'Error': str(e)}

ml_tbl = pd.DataFrame(ml_results).T
ml_tbl.to_csv('iter8_ml_regime.csv')
print('E2 ML Regime Prediction:\n', ml_tbl.to_string())

# ======================================================================
# E3. Bias-Variance Tradeoff in Parameter Selection
# ======================================================================
# Test: how does parameter flexibility affect OOS performance?
# SMA window sweep with different training windows
param_results = {}

for train_window in [252, 504, 756, 1008, 1260]:  # 1-5 years
    # Split data
    train_end = train_window
    test_start = train_window
    
    # Optimize SMA window on training data
    best_window = 200
    best_sharpe = -np.inf
    
    for window in [50, 100, 150, 200, 250, 300, 350, 400]:
        w = (px[['SPY']][:train_end] > px[['SPY']][:train_end].rolling(window).mean()).astype(float).reindex(columns=px.columns).fillna(0)
        r = backtest(w, rets[:train_end], COST).dropna()
        if len(r) > 50:
            s = sharpe(r)
            if s > best_sharpe:
                best_sharpe = s
                best_window = window
    
    # Test on OOS
    w_oos = (px[['SPY']][test_start:] > px[['SPY']][test_start:].rolling(best_window).mean()).astype(float).reindex(columns=px.columns).fillna(0)
    r_oos = backtest(w_oos, rets[test_start:], COST).dropna()
    oos_sharpe = sharpe(r_oos) if len(r_oos) > 50 else 0
    
    param_results[train_window] = {
        'Best_Window': best_window,
        'IS_Sharpe': round(best_sharpe, 2),
        'OOS_Sharpe': round(oos_sharpe, 2),
        'Degradation': round(oos_sharpe - best_sharpe, 2)
    }

bv_tbl = pd.DataFrame(param_results).T
bv_tbl.to_csv('iter8_bias_variance.csv')
print('E3 Bias-Variance Tradeoff:\n', bv_tbl.to_string())

# More formal: model complexity vs performance
# Ridge alpha sweep
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_results = {}
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_s, y_ret_train)
    pred = model.predict(X_test_s)
    mse = np.mean((pred - y_ret_test)**2)
    dir_acc = np.mean((pred > 0) == (y_ret_test > 0))
    ridge_results[alpha] = {'MSE': round(mse*1e6, 2), 'DirAcc': round(dir_acc, 3)}

ridge_tbl = pd.DataFrame(ridge_results).T
ridge_tbl.to_csv('iter8_ridge_complexity.csv')
print('E3 Ridge Complexity:\n', ridge_tbl.to_string())

# ======================================================================
# E4. Purged Cross-Validation for Time-Series ML
# ======================================================================
# Implement Purged K-Fold (López de Prado) for ML model selection
def purged_kfold_split(n_samples, n_splits=5, embargo_pct=0.01):
    """Generate purged K-fold indices for time series."""
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits
    embargo = int(n_samples * embargo_pct)
    
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        
        # Train: everything before test_start - embargo, and after test_end + embargo
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[max(0, test_start - embargo):min(n_samples, test_end + embargo)] = False
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.arange(test_start, test_end)
        
        if len(train_idx) > 100 and len(test_idx) > 20:
            yield train_idx, test_idx

# Test ML models with purged CV
purge_results = {}
for name, model in models.items():
    if 'Error' not in ml_results.get(name, {}):
        cv_scores = []
        for train_idx, test_idx in purged_kfold_split(len(X), n_splits=5, embargo_pct=0.02):
            try:
                X_cv_train, X_cv_test = X.iloc[train_idx], X.iloc[test_idx]
                y_cv_train, y_cv_test = y_ret.iloc[train_idx], y_ret.iloc[test_idx]
                
                X_cv_train_s = scaler.fit_transform(X_cv_train)
                X_cv_test_s = scaler.transform(X_cv_test)
                
                model.fit(X_cv_train_s, y_cv_train)
                pred = model.predict(X_cv_test_s)
                corr = np.corrcoef(pred, y_cv_test)[0,1]
                cv_scores.append(corr)
            except:
                pass
        if cv_scores:
            purge_results[name] = {
                'Mean_Corr': round(np.mean(cv_scores), 3),
                'Std_Corr': round(np.std(cv_scores), 3),
                'Min_Corr': round(np.min(cv_scores), 3),
                'Max_Corr': round(np.max(cv_scores), 3),
            }

purge_tbl = pd.DataFrame(purge_results).T
purge_tbl.to_csv('iter8_purged_cv_ml.csv')
print('E4 Purged CV for ML:\n', purge_tbl.to_string())

# ======================================================================
# E5. Deep Learning: Classification for Regime (Bull/Bear/Sideways)
# ======================================================================
# Define regimes
spy_ret_21 = px['SPY'].pct_change(21)
spy_vol_63 = rets['SPY'].rolling(63).std() * np.sqrt(252)

regime = pd.Series(1, index=px.index)  # 0=bull, 1=sideways, 2=bear
regime[(spy_ret_21 > 0.02) & (spy_vol_63 < spy_vol_63.rolling(252).quantile(0.5))] = 0
regime[(spy_ret_21 < -0.02) | (spy_vol_63 > spy_vol_63.rolling(252).quantile(0.8))] = 2

# Align
common_idx = features.index.intersection(regime.dropna().index)
X_clf = features.loc[common_idx]
y_clf = regime.loc[common_idx]

split = int(len(X_clf) * 0.7)
X_clf_train, X_clf_test = X_clf.iloc[:split], X_clf.iloc[split:]
y_clf_train, y_clf_test = y_clf.iloc[:split], y_clf.iloc[split:]

X_clf_train_s = scaler.fit_transform(X_clf_train)
X_clf_test_s = scaler.transform(X_clf_test)

# Classification models
clf_models = {
    'Logistic': MLPClassifier(hidden_layer_sizes=(), max_iter=500, random_state=42),
    'MLP(32)': MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42),
    'MLP(64,32)': MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
    'MLP(128,64,32)': MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=500, random_state=42),
    'RF(100)': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),  # use regressor for proba
}

clf_results = {}
for name, model in clf_models.items():
    try:
        if name == 'RF(100)':
            # Use regressor to get probabilities
            model.fit(X_clf_train_s, y_clf_train)
            pred_proba = model.predict(X_clf_test_s)  # not proba
            pred = np.round(pred_proba).astype(int)
        else:
            model.fit(X_clf_train_s, y_clf_train)
            pred = model.predict(X_clf_test_s)
        
        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(y_clf_test, pred)
        
        # Strategy: long SPY in bull (0), cash in bear (2), half in sideways (1)
        w_dl = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        pred_series = pd.Series(pred, index=X_clf_test.index)
        for t, p in pred_series.items():
            if p == 0:  # bull
                w_dl.loc[t, 'SPY'] = 1.0
            elif p == 1:  # sideways
                w_dl.loc[t, 'SPY'] = 0.5
            else:  # bear
                w_dl.loc[t, 'SPY'] = 0.0
        w_dl = w_dl.ffill().fillna(0)
        
        r_dl = backtest(w_dl, rets, COST).dropna()
        clf_results[name] = {
            'Accuracy': round(acc, 3),
            'Strategy Sharpe': round(sharpe(r_dl), 2),
            'Strategy AnnRet%': round(252*r_dl.mean()*100, 2),
            'Strategy MaxDD%': round(100*max_dd((1+r_dl).cumprod()), 2),
        }
    except Exception as e:
        clf_results[name] = {'Error': str(e)}

clf_tbl = pd.DataFrame(clf_results).T
clf_tbl.to_csv('iter8_dl_classification.csv')
print('E5 DL Classification:\n', clf_tbl.to_string())

# ======================================================================
# E6. Ensemble of ML Models (Model Averaging)
# ======================================================================
# Combine predictions from multiple models
ensemble_preds = {}
for name in ['Ridge(1)', 'RF(100)', 'GBM(100)', 'MLP(64,32)']:
    if name in models:
        model = models[name]
        model.fit(X_train_s, y_ret_train)
        ensemble_preds[name] = model.predict(X_test_s)

# Simple average
avg_pred = np.mean(list(ensemble_preds.values()), axis=0)
# Weighted by inverse MSE
weights = {name: 1/ml_results[name]['Test MSE'] for name in ensemble_preds if 'Test MSE' in ml_results[name]}
total_w = sum(weights.values())
weighted_pred = sum(ensemble_preds[name] * (weights[name]/total_w) for name in weights)

# Test ensemble strategies
w_avg = pd.DataFrame(0.0, index=px.index, columns=px.columns)
w_wgt = pd.DataFrame(0.0, index=px.index, columns=px.columns)

avg_series = pd.Series(avg_pred, index=X_test.index)
wgt_series = pd.Series(weighted_pred, index=X_test.index)

for t in avg_series[avg_series > 0].index:
    w_avg.loc[t, 'SPY'] = 1.0
for t in wgt_series[wgt_series > 0].index:
    w_wgt.loc[t, 'SPY'] = 1.0

w_avg = w_avg.ffill().fillna(0)
w_wgt = w_wgt.ffill().fillna(0)

r_avg = backtest(w_avg, rets, COST).dropna()
r_wgt = backtest(w_wgt, rets, COST).dropna()

ens_tbl = pd.DataFrame({
    'Strategy': ['MLP(64,32) Single', 'Simple Average', 'MSE-Weighted Average'],
    'Sharpe': [round(sharpe(backtest(pd.DataFrame({'SPY': (pd.Series(models['MLP(64,32)'].predict(X_test_s), index=X_test.index) > 0).astype(float)}, index=px.index).ffill().fillna(0), rets, COST).dropna()), 2),
               round(sharpe(r_avg), 2), round(sharpe(r_wgt), 2)],
    'AnnRet%': [round(252*backtest(pd.DataFrame({'SPY': (pd.Series(models['MLP(64,32)'].predict(X_test_s), index=X_test.index) > 0).astype(float)}, index=px.index).ffill().fillna(0), rets, COST).dropna().mean()*100, 2),
                round(252*r_avg.mean()*100, 2), round(252*r_wgt.mean()*100, 2)],
    'MaxDD%': [round(100*max_dd((1+backtest(pd.DataFrame({'SPY': (pd.Series(models['MLP(64,32)'].predict(X_test_s), index=X_test.index) > 0).astype(float)}, index=px.index).ffill().fillna(0), rets, COST).dropna()).cumprod()), 2),
               round(100*max_dd((1+r_avg).cumprod()), 2), round(100*max_dd((1+r_wgt).cumprod()), 2)],
})
ens_tbl.to_csv('iter8_ensemble_ml.csv', index=False)
print('E6 Ensemble ML:\n', ens_tbl.to_string(index=False))

# ======================================================================
# Compile Iteration #8
# ======================================================================
all_strats = {
    'SMA200': w_sma,
    'GEM': w_gem,
    'XSec Mom': w_xs,
    '60/40': w_6040,
    'All Weather': w_aw_mapped,
    'Permanent Portfolio': w_pp,
    'Golden Butterfly': w_gb,
    'Risk Parity Static': w_rp_static,
    'Global Market Portfolio': w_gmp_mapped,
}

# Add best ML strategy if exists
if 'MLP(64,32)' in ml_results and 'Strategy Sharpe' in ml_results['MLP(64,32)']:
    # Recreate the ML strategy weights
    model = models['MLP(64,32)']
    model.fit(X_train_s, y_ret_train)
    pred = model.predict(X_test_s)
    w_ml = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    pred_series = pd.Series(pred, index=X_test.index)
    for t in pred_series[pred_series > 0].index:
        w_ml.loc[t, 'SPY'] = 1.0
    w_ml = w_ml.ffill().fillna(0)
    all_strats['ML-Regime'] = w_ml

results = []
all_rets = {}
for name, obj in all_strats.items():
    if isinstance(obj, pd.Series):
        r = obj
    else:
        r = backtest(obj, rets, COST).dropna()
    all_rets[name] = r
    results.append(perf(r, name))

res = pd.DataFrame(results)
res['Calmar'] = res['AnnRet%'] / res['MaxDD%'].abs()

# Walk-forward folds
n = len(px)
fb = [(k * n // 4, (k + 1) * n // 4) for k in range(4)]
wf_rows = []
for name, r in all_rets.items():
    r = r.dropna()
    if len(r) > 100:
        shs = [round(sharpe(r.iloc[a:b]), 2) for a, b in fb]
    else:
        shs = [0, 0, 0, 0]
    wf_rows.append({'Strategy': name, 'F1': shs[0], 'F2': shs[1], 'F3': shs[2], 'F4': shs[3]})
wf = pd.DataFrame(wf_rows)

# Validation
sharpes = [sharpe(r) for r in all_rets.values() if len(r) > 100]
sr_std = float(np.std(sharpes, ddof=1))
vals = [report_validation(results[i]['name'], list(all_rets.values())[i], len(all_strats), sr_std)
        for i in range(len(results)) if len(list(all_rets.values())[i]) > 100]
val = pd.DataFrame(vals)

# Save
res.to_csv('iter8_comprehensive_perf.csv', index=False)
wf.to_csv('iter8_comprehensive_walkforward.csv', index=False)
val.to_csv('iter8_comprehensive_validation.csv', index=False)

# Plots
plt.figure(figsize=(12, 7))
for name, r in all_rets.items():
    if len(r) > 100:
        plt.plot((1+r).cumprod(), label=name, alpha=0.8, linewidth=0.8)
plt.legend(fontsize=7, ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Iteration #8 — All Strategies Equity Curves (net 10bp)')
plt.ylabel('Growth of $1'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('iter8_equity.png', dpi=110, bbox_inches='tight')

# Static benchmarks
plt.figure(figsize=(10, 6))
static_tbl['Sharpe'].sort_values().plot(kind='barh', color='steelblue')
plt.title('Static Benchmark Sharpe Ratios (net 10bp)')
plt.xlabel('Sharpe'); plt.tight_layout(); plt.savefig('iter8_static.png', dpi=110)

# ML model comparison
plt.figure(figsize=(10, 6))
ml_plot = ml_tbl[['Test MSE', 'Dir Acc']].dropna()
if len(ml_plot) > 0:
    ax1 = ml_plot['Test MSE'].plot(kind='bar', color='coral', position=0, width=0.4, label='MSE (x1e6)')
    ax2 = ml_plot['Dir Acc'].plot(kind='bar', color='steelblue', position=1, width=0.4, label='Dir Acc', secondary_y=True)
    plt.title('ML Model Comparison: MSE vs Directional Accuracy')
    plt.legend(); plt.tight_layout(); plt.savefig('iter8_ml.png', dpi=110)

# Bias-variance
plt.figure(figsize=(10, 6))
bv_tbl[['IS_Sharpe', 'OOS_Sharpe']].plot(marker='o')
plt.title('Bias-Variance: IS vs OOS Sharpe by Training Window')
plt.xlabel('Training Window (days)'); plt.ylabel('Sharpe'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('iter8_bias_variance.png', dpi=110)

print('\n=== ITERATION #8 PERFORMANCE ===')
print(res.to_string(index=False))
print('\n=== WALK-FORWARD ===')
print(wf.to_string(index=False))
print('\n=== VALIDATION ===')
print(val.to_string(index=False))
print('\nIteration #8 complete.')