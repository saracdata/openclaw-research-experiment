"""
Iteration #15 — QuantStart Advanced Frontiers:
- Rough Volatility / fBM (fractional Brownian motion)
- Rough Path Theory / Signatures
- Limit Order Book (LOB) Microstructure
- Jupyter/Plotly Prototyping
- Advanced Options Strategies
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/root/quant')
from engine import load, backtest, perf, sharpe, max_dd
from strategies import sma_trend, xsec_momentum, vol_target

# ============================================================
# DATA LOADING
# ============================================================
tickers = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM', 'EFA', 'EEM', 'DBC', 'VNQ', 'XLE', 'XLF', 'XLK', 'XLV', 'XLU']
price_data = {}
for t in tickers:
    df = load(t)
    price_data[t] = df['Close']

prices = pd.DataFrame(price_data).dropna()
returns = prices.pct_change().dropna()

# Common tickers for SPY-specific strategies
spy = prices['SPY']
spy_ret = spy.pct_change().dropna()

print("Data loaded:", len(prices), "days,", len(tickers), "tickers")
print("Date range:", prices.index[0].date(), "to", prices.index[-1].date())

# ============================================================
# E1: ROUGH VOLATILITY / fBM - RFSV SIMULATION & STRESS TESTING
# ============================================================
print("\n=== E1: Rough Volatility / RFSV Stress Testing ===")

def simulate_rfsv(n_paths=50, n_days=252*14, H=0.1, nu=0.5, rho=-0.7, v0=0.04):
    """
    Rough Fractional Stochastic Volatility (RFSV) model:
    dS/S = sqrt(V) dW
    V_t = V_0 * exp(nu * B^H_t - 0.5 * nu^2 * t^(2H))
    where B^H is fBM with Hurst H.
    """
    dt = 1/252
    t = np.arange(n_days) * dt
    
    # Generate fBM using Cholesky (exact but O(N^3) - use small n for demo)
    # For larger n, use circulant embedding or wavelet method
    def fbm_cholesky(n, H):
        """Generate fBM path using Cholesky decomposition."""
        t = np.arange(n) * dt
        # Covariance matrix of fBM
        C = 0.5 * (np.abs(t[:, None])**(2*H) + np.abs(t[None, :])**(2*H) - np.abs(t[:, None] - t[None, :])**(2*H))
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        return L @ np.random.randn(n)
    
    paths = []
    for _ in range(n_paths):
        # Generate correlated fBM and BM
        W1 = np.random.randn(n_days) * np.sqrt(dt)  # Standard BM for price
        BH = fbm_cholesky(n_days, H) * np.sqrt(dt)  # fBM for volatility
        
        # Correlated: dW2 = rho * dW1 + sqrt(1-rho^2) * dBH
        W2 = rho * W1 + np.sqrt(1 - rho**2) * BH
        
        # Volatility path
        vol_path = v0 * np.exp(nu * W2 - 0.5 * nu**2 * t**(2*H))
        vol_path = np.clip(vol_path, 0.001, 2.0)  # Prevent explosion
        
        # Price path
        log_ret = np.sqrt(vol_path) * W1 - 0.5 * vol_path * dt
        price_path = 100 * np.exp(np.cumsum(log_ret))
        paths.append(price_path)
    
    return np.array(paths)

def test_strategy_on_paths(strategy_func, paths, **kwargs):
    """Test a strategy on multiple price paths."""
    sharpes = []
    for path in paths:
        px = pd.Series(path, index=pd.date_range('2012-01-01', periods=len(path), freq='B'))
        try:
            pos = strategy_func(px, **kwargs)
            ret = px.pct_change().dropna()
            pos = pos.reindex(ret.index).fillna(0)
            strat_ret = backtest(pos, ret, cost_bps=10, lag=1)
            sh = sharpe(strat_ret)
            sharpes.append(sh)
        except:
            sharpes.append(np.nan)
    return np.array(sharpes)

# Simulate RFSV paths (using smaller sample for speed)
print("Simulating RFSV paths...")
rfsv_paths = simulate_rfsv(n_paths=30, n_days=252*14, H=0.1, nu=0.3)

# Test strategies on RFSV
strategies_rfsv = {
    'SMA200': lambda px: sma_trend(px, 200),
    'VolTarget': lambda px: vol_target(px, target=0.10, window=21, lev_cap=2.0),
    'XSecMom': lambda px: xsec_momentum(pd.DataFrame({'SPY': px}), top_n=1).iloc[:, 0],
}

rfsv_results = {}
for name, func in strategies_rfsv.items():
    sh = test_strategy_on_paths(func, rfsv_paths)
    rfsv_results[name] = sh
    print(f"  {name}: mean={np.nanmean(sh):.3f}, std={np.nanstd(sh):.3f}, min={np.nanmin(sh):.3f}, max={np.nanmax(sh):.3f}, %neg={np.mean(sh<0)*100:.1f}%")

# Save RFSV results
rfsv_df = pd.DataFrame(rfsv_results)
rfsv_df.to_csv('/root/quant/iter15_rfsv_stress.csv', index=False)

# ============================================================
# E2: ROUGH PATH THEORY / SIGNATURES - SIGNATURE FEATURES FOR REGIME
# ============================================================
print("\n=== E2: Rough Path Theory / Signatures ===")

def compute_signature(path, order=3):
    """
    Compute truncated signature of a path up to given order.
    Signature S = (1, S^1, S^2, ..., S^k) where S^i are iterated integrals.
    For discrete path, use Riemann sum approximation.
    """
    n = len(path)
    if n < 2:
        return np.array([1.0])
    
    # Increments
    dX = np.diff(path)
    
    # Level 1: sum of increments
    S1 = np.sum(dX)
    
    # Level 2: iterated integrals
    S2 = 0.0
    for i in range(n-1):
        for j in range(i+1, n-1):
            S2 += dX[i] * dX[j]
    
    # Level 3: triple iterated integrals (simplified)
    S3 = 0.0
    if order >= 3:
        for i in range(n-1):
            for j in range(i+1, n-1):
                for k in range(j+1, n-1):
                    S3 += dX[i] * dX[j] * dX[k]
    
    sig = [1.0, S1]
    if order >= 2:
        sig.append(S2)
    if order >= 3:
        sig.append(S3)
    
    return np.array(sig)

def compute_log_signature(path, order=3):
    """Compute log-signature (free Lie algebra basis) - simplified."""
    sig = compute_signature(path, order)
    # Log-signature for 1D: log(S) = (log(S^1), S^2 - 0.5*(S^1)^2, ...)
    if len(sig) >= 2:
        logsig = [sig[0], sig[1]]
        if len(sig) >= 3:
            logsig.append(sig[2] - 0.5 * sig[1]**2)
        if len(sig) >= 4:
            # Approximation for level 3
            logsig.append(sig[3] - sig[1]*sig[2] + sig[1]**3/3)
        return np.array(logsig)
    return sig

# Compute rolling signatures on SPY returns
window = 63  # ~3 months
sig_features = []
sig_dates = []

for i in range(window, len(spy_ret)):
    ret_window = spy_ret.iloc[i-window:i].values
    # Path = cumulative returns
    path = np.cumsum(ret_window)
    sig = compute_log_signature(path, order=3)
    sig_features.append(sig)
    sig_dates.append(spy_ret.index[i])

sig_df = pd.DataFrame(sig_features, index=sig_dates, columns=[f'sig_{i}' for i in range(len(sig_features[0]))])
print(f"Signature features shape: {sig_df.shape}")
print(f"Columns: {sig_df.columns.tolist()}")

# Use signature features to predict volatility regime
# Target: high vol (next 21 days RV > 80th percentile)
rv_21 = spy_ret.rolling(21).std() * np.sqrt(252)
rv_threshold = rv_21.quantile(0.8)
target = (rv_21.shift(-21) > rv_threshold).astype(int).dropna()

# Align
common_idx = sig_df.index.intersection(target.index)
X = sig_df.loc[common_idx]
y = target.loc[common_idx]

# Walk-forward logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

split = int(len(X) * 0.7)
train_sharpes = []
test_sharpes = []

for i in range(split, len(X), 63):  # Refit quarterly
    X_train = X.iloc[:i]
    y_train = y.iloc[:i]
    X_test = X.iloc[i:i+63]
    y_test = y.iloc[i:i+63]
    
    if len(X_test) < 10 or len(y_train) < 50:
        continue
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_scaled, y_train)
    
    preds = clf.predict(X_test_scaled)
    acc = (preds == y_test.values).mean()
    
    # Strategy: SMA200 but reduce exposure when high vol predicted
    test_dates = X_test.index
    if len(test_dates) > 0:
        spy_test = spy.loc[test_dates[0]:test_dates[-1]]
        pos_base = sma_trend(spy_test, 200)
        pos_regime = pos_base.copy()
        pred_series = pd.Series(preds, index=test_dates)
        pred_series = pred_series.reindex(pos_regime.index, method='ffill').fillna(0)
        pos_regime = pos_regime * (1 - 0.5 * pred_series)  # Reduce 50% when high vol predicted
        
        ret_test = spy_test.pct_change().dropna()
        pos_base = pos_base.reindex(ret_test.index).fillna(0)
        pos_regime = pos_regime.reindex(ret_test.index).fillna(0)
        
        sh_base = sharpe(backtest(pos_base, ret_test, cost_bps=10))
        sh_reg = sharpe(backtest(pos_regime, ret_test, cost_bps=10))
        train_sharpes.append(sh_base)
        test_sharpes.append(sh_reg)

print(f"  Signature regime accuracy: {acc:.3f}")
print(f"  Base Sharpe (avg): {np.mean(train_sharpes):.3f}")
print(f"  Regime Sharpe (avg): {np.mean(test_sharpes):.3f}")

# Save signature results
sig_results = pd.DataFrame({
    'accuracy': [acc],
    'base_sharpe_mean': [np.mean(train_sharpes)],
    'regime_sharpe_mean': [np.mean(test_sharpes)],
})
sig_results.to_csv('/root/quant/iter15_signature_regime.csv', index=False)

# ============================================================
# E3: LOB MICROSTRUCTURE - ENHANCED LOB WITH ADVERSE SELECTION
# ============================================================
print("\n=== E3: LOB Microstructure with Adverse Selection ===")

class EnhancedLOB:
    """Enhanced Limit Order Book with adverse selection and queue position."""
    def __init__(self, mid_price=100, tick_size=0.01, depth=10, base_level_size=1000):
        self.mid = mid_price
        self.tick = tick_size
        self.depth = depth
        self.base_size = base_level_size
        self.bids = {i: base_level_size * (1.2 ** i) for i in range(depth)}  # Deeper = larger
        self.asks = {i: base_level_size * (1.2 ** i) for i in range(depth)}
        self.queue_position = {}  # Track queue position for our orders
        
    def get_spread_bps(self):
        best_bid = self.mid - self.tick * (self.depth // 2)
        best_ask = self.mid + self.tick * (self.depth // 2)
        return (best_ask - best_bid) / self.mid * 1e4
    
    def market_order_cost(self, shares, side='buy'):
        """Execute market order walking the book. Returns VWAP cost in bps."""
        if side == 'buy':
            levels = self.asks
            price_mult = 1
        else:
            levels = self.bids
            price_mult = -1
        
        remaining = shares
        total_cost = 0
        total_filled = 0
        
        for level in range(self.depth):
            available = levels[level]
            fill = min(remaining, available)
            if fill <= 0:
                break
            price = self.mid + price_mult * self.tick * (level + 1)
            total_cost += fill * price
            total_filled += fill
            remaining -= fill
            levels[level] -= fill
            
            # Adverse selection: informed traders take liquidity
            if fill > 0 and np.random.random() < 0.1:
                # Reduce remaining liquidity at this level (toxic flow)
                levels[level] = int(levels[level] * 0.5)
        
        if total_filled == 0:
            return 0
        
        vwap = total_cost / total_filled
        cost_bps = (vwap - self.mid) / self.mid * 1e4 * price_mult
        return cost_bps
    
    def replenish(self, rate=0.1):
        """Liquidity replenishment."""
        for level in range(self.depth):
            target = self.base_size * (1.2 ** level)
            self.bids[level] = int(self.bids[level] + rate * (target - self.bids[level]))
            self.asks[level] = int(self.asks[level] + rate * (target - self.asks[level]))
    
    def update_mid(self, new_mid):
        self.mid = new_mid

# Simulate LOB execution for our strategies
lob = EnhancedLOB(mid_price=400, tick_size=0.01, depth=20, base_level_size=5000)

# Monthly rebalance simulation with LOB costs
def simulate_lob_execution(weights, prices, lob_model, adv=5e6):
    """Simulate execution of monthly rebalance through LOB."""
    n_assets = len(weights.columns)
    costs = []
    
    for date in weights.index:
        w = weights.loc[date]
        # Assume $10M portfolio, execute proportionally
        portfolio_val = 10e6
        daily_cost = 0
        
        for asset, weight in w.items():
            if weight == 0:
                continue
            # Shares to trade
            px = prices.loc[date, asset] if asset in prices.columns else 400
            target_val = portfolio_val * abs(weight)
            shares = target_val / px
            
            # Participation rate (assume we're 1% of ADV)
            part_rate = min(shares / adv, 0.1)
            
            # Walk the book
            cost_bps = lob_model.market_order_cost(int(shares), 'buy' if weight > 0 else 'sell')
            
            # Adverse selection adjustment based on participation
            cost_bps *= (1 + part_rate * 5)  # Higher participation = more impact
            
            daily_cost += abs(weight) * cost_bps / 1e4
            
            # Update LOB mid price (simulate price impact)
            if shares > 0:
                lob_model.update_mid(px * (1 + cost_bps / 1e4 * np.sign(weight)))
        
        lob_model.replenish()
        costs.append(daily_cost)
    
    return np.array(costs)

# Test on XSec Momentum (highest turnover)
print("  Testing LOB execution on XSec Momentum...")
xsec_pos = xsec_momentum(prices, top_n=3)
xsec_ret = prices.pct_change().dropna()
xsec_pos = xsec_pos.reindex(xsec_ret.index).fillna(0)

# Simple cost comparison
base_cost = 10  # bps
lob_costs = simulate_lob_execution(xsec_pos, prices, lob)

print(f"  Base cost assumption: {base_cost} bps")
print(f"  LOB avg cost: {np.mean(lob_costs)*1e4:.1f} bps")
print(f"  LOB max cost: {np.max(lob_costs)*1e4:.1f} bps")

# Strategy with LOB costs
strat_lob = (xsec_pos * xsec_ret).sum(axis=1) - pd.Series(lob_costs, index=xsec_pos.index)
sh_base = sharpe(backtest(xsec_pos, xsec_ret, cost_bps=base_cost))
sh_lob = sharpe(strat_lob)

print(f"  Base Sharpe (10bp): {sh_base:.3f}")
print(f"  LOB Sharpe: {sh_lob:.3f}")

lob_results = pd.DataFrame({
    'metric': ['base_sharpe', 'lob_sharpe', 'avg_lob_cost_bps', 'max_lob_cost_bps'],
    'value': [sh_base, sh_lob, np.mean(lob_costs)*1e4, np.max(lob_costs)*1e4]
})
lob_results.to_csv('/root/quant/iter15_lob_microstructure.csv', index=False)

# ============================================================
# E4: JUPYTER/PLOTLY PROTOTYPING - INTERACTIVE VISUALIZATIONS
# ============================================================
print("\n=== E4: Jupyter/Plotly Prototyping Visualizations ===")

# Create comprehensive Plotly visualizations
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # 1. Equity curves with drawdown
    strat_returns = {}
    for name, func in strategies_rfsv.items():
        pos = func(spy)
        strat_returns[name] = backtest(pos, spy_ret, cost_bps=10, lag=1)
    
    # Add XSec Momentum
    xsec_pos = xsec_momentum(prices, top_n=3)
    xsec_pos = xsec_pos.reindex(spy_ret.index).fillna(0)
    strat_returns['XSecMom'] = backtest(xsec_pos, spy_ret, cost_bps=10, lag=1)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=('Equity Curves', 'Drawdowns'))
    
    for name, ret in strat_returns.items():
        cum = (1 + ret).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name=name, mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f'{name} DD', mode='lines', 
                                 line=dict(dash='dot'), showlegend=False), row=2, col=1)
    
    fig.update_layout(height=800, title='Strategy Equity Curves & Drawdowns',
                      xaxis2_title='Date', yaxis_title='Cumulative Return',
                      yaxis2_title='Drawdown %', hovermode='x unified')
    fig.write_html('/root/quant/iter15_equity_curves.html')
    fig.write_image('/root/quant/iter15_equity_curves.png')
    print("  Saved equity curves visualization")
    
    # 2. Return distribution comparison
    fig2 = go.Figure()
    for name, ret in strat_returns.items():
        fig2.add_trace(go.Histogram(x=ret*100, name=name, opacity=0.6, nbinsx=50))
    fig2.update_layout(barmode='overlay', title='Return Distributions',
                       xaxis_title='Daily Return (%)', yaxis_title='Frequency')
    fig2.write_html('/root/quant/iter15_return_dist.html')
    fig2.write_image('/root/quant/iter15_return_dist.png')
    print("  Saved return distribution visualization")
    
    # 3. Rolling Sharpe (63-day window)
    fig3 = go.Figure()
    for name, ret in strat_returns.items():
        roll_sh = ret.rolling(63).apply(lambda x: sharpe(pd.Series(x)), raw=True)
        fig3.add_trace(go.Scatter(x=roll_sh.index, y=roll_sh, name=name, mode='lines'))
    fig3.update_layout(title='Rolling 63-Day Sharpe Ratio',
                       xaxis_title='Date', yaxis_title='Sharpe')
    fig3.write_html('/root/quant/iter15_rolling_sharpe.html')
    fig3.write_image('/root/quant/iter15_rolling_sharpe.png')
    print("  Saved rolling Sharpe visualization")
    
    # 4. Correlation heatmap
    strat_df = pd.DataFrame(strat_returns)
    corr = strat_df.corr()
    fig4 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                      colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                                      text=np.round(corr.values, 2), texttemplate='%{text}'))
    fig4.update_layout(title='Strategy Correlation Matrix')
    fig4.write_html('/root/quant/iter15_corr_heatmap.html')
    fig4.write_image('/root/quant/iter15_corr_heatmap.png')
    print("  Saved correlation heatmap")
    
    # 5. Risk-return scatter
    fig5 = go.Figure()
    for name, ret in strat_returns.items():
        ann_ret = ret.mean() * 252 * 100
        ann_vol = ret.std() * np.sqrt(252) * 100
        sh = sharpe(ret)
        fig5.add_trace(go.Scatter(x=[ann_vol], y=[ann_ret], mode='markers+text',
                                  text=[f'{name}<br>Sharpe: {sh:.2f}'],
                                  textposition='top center', marker=dict(size=12)))
    fig5.update_layout(title='Risk-Return Profile',
                       xaxis_title='Annualized Volatility (%)', yaxis_title='Annualized Return (%)')
    fig5.write_html('/root/quant/iter15_risk_return.html')
    fig5.write_image('/root/quant/iter15_risk_return.png')
    print("  Saved risk-return visualization")
    
    prototyping_results = pd.DataFrame({
        'visualization': ['equity_curves', 'return_dist', 'rolling_sharpe', 'corr_heatmap', 'risk_return'],
        'status': ['saved'] * 5
    })
    prototyping_results.to_csv('/root/quant/iter15_prototyping.csv', index=False)
    
except Exception as e:
    print(f"  Plotly not available or error: {e}")
    prototyping_results = pd.DataFrame({'error': [str(e)]})
    prototyping_results.to_csv('/root/quant/iter15_prototyping.csv', index=False)

# ============================================================
# E5: ADVANCED OPTIONS STRATEGIES
# ============================================================
print("\n=== E5: Advanced Options Strategies ===")

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes Greeks."""
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 252
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 252
    
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T) / 100  # per 1% vol change
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

# 1. Protective Put (Married Put)
print("  Testing Protective Put strategy...")
put_strike_pct = 0.95  # 5% OTM
put_dte = 30  # days to expiry
r = 0.04  # risk-free rate

spy_px = spy
spy_ret_daily = spy_ret

# Simulate protective put: buy SPY + buy 95% OTM put monthly
prot_put_returns = []
for i in range(put_dte, len(spy_px) - put_dte, 21):  # Monthly roll
    S = spy_px.iloc[i]
    K = S * put_strike_pct
    T = put_dte / 252
    
    # Estimate IV from recent realized vol * 1.2 (typical VRP)
    rv = spy_ret_daily.iloc[max(0,i-21):i].std() * np.sqrt(252)
    iv = rv * 1.2
    
    put_price = black_scholes_put(S, K, T, r, iv)
    put_cost_pct = put_price / S
    
    # Next period return
    next_ret = spy_px.iloc[i+put_dte] / spy_px.iloc[i] - 1
    # Put payoff
    put_payoff = max(K - spy_px.iloc[i+put_dte], 0) / S
    
    total_ret = next_ret - put_cost_pct + put_payoff
    prot_put_returns.append(total_ret)

prot_put_returns = pd.Series(prot_put_returns, index=spy_px.index[put_dte::21][:len(prot_put_returns)])
sh_prot_put = sharpe(prot_put_returns)
print(f"  Protective Put Sharpe: {sh_prot_put:.3f}")

# 2. Covered Call (Buy-Write)
print("  Testing Covered Call strategy...")
call_strike_pct = 1.05  # 5% OTM

cov_call_returns = []
for i in range(call_dte := 30, len(spy_px) - 30, 21):
    S = spy_px.iloc[i]
    K = S * call_strike_pct
    T = call_dte / 252
    rv = spy_ret_daily.iloc[max(0,i-21):i].std() * np.sqrt(252)
    iv = rv * 1.2
    
    call_price = black_scholes_call(S, K, T, r, iv)
    call_premium_pct = call_price / S
    
    next_ret = spy_px.iloc[i+call_dte] / spy_px.iloc[i] - 1
    # Call caps upside at strike
    capped_ret = min(next_ret, (K - S) / S) + call_premium_pct
    cov_call_returns.append(capped_ret)

cov_call_returns = pd.Series(cov_call_returns, index=spy_px.index[call_dte::21][:len(cov_call_returns)])
sh_cov_call = sharpe(cov_call_returns)
print(f"  Covered Call Sharpe: {sh_cov_call:.3f}")

# 3. Collar (Protective Put + Covered Call)
print("  Testing Collar strategy...")
collar_rets = []
for i in range(30, len(spy_px) - 30, 21):
    S = spy_px.iloc[i]
    K_put = S * 0.95
    K_call = S * 1.05
    T = 30/252
    rv = spy_ret_daily.iloc[max(0,i-21):i].std() * np.sqrt(252)
    iv = rv * 1.2
    
    put_px = black_scholes_put(S, K_put, T, r, iv)
    call_px = black_scholes_call(S, K_call, T, r, iv)
    net_cost = (put_px - call_px) / S
    
    next_ret = spy_px.iloc[i+30] / spy_px.iloc[i] - 1
    put_payoff = max(K_put - spy_px.iloc[i+30], 0) / S
    call_payoff = -max(spy_px.iloc[i+30] - K_call, 0) / S
    
    total = next_ret + net_cost + put_payoff + call_payoff
    collar_rets.append(total)

collar_rets = pd.Series(collar_rets, index=spy_px.index[30::21][:len(collar_rets)])
sh_collar = sharpe(collar_rets)
print(f"  Collar Sharpe: {sh_collar:.3f}")

# 4. Volatility Targeting with Options (Delta-Hedged Straddle)
print("  Testing Delta-Hedged Straddle (Vol Arb)...")
# Simplified: straddle P&L = Vega * (IV - RV) + Gamma * (dS)^2 - Theta
straddle_pnl = []
for i in range(30, len(spy_px) - 21, 21):
    S = spy_px.iloc[i]
    K = S  # ATM
    T = 30/252
    rv = spy_ret_daily.iloc[max(0,i-21):i].std() * np.sqrt(252)
    iv = rv * 1.2
    
    greeks = bs_greeks(S, K, T, r, iv, 'call')
    # Straddle: long call + long put (ATM)
    vega = 2 * greeks['vega']  # Both legs
    gamma = 2 * greeks['gamma']
    theta = 2 * greeks['theta'] * 21  # 21 days decay
    
    # Realized move
    dS = spy_px.iloc[i+21] - S
    realized_var = (dS/S)**2 * 252/21
    
    # P&L = Vega*(IV-RV) + 0.5*Gamma*S^2*realized_var - Theta
    pnl_pct = (vega * (iv - rv) + 0.5 * gamma * S**2 * realized_var / 252 + theta) / S
    straddle_pnl.append(pnl_pct)

straddle_pnl = pd.Series(straddle_pnl, index=spy_px.index[30::21][:len(straddle_pnl)])
sh_straddle = sharpe(straddle_pnl)
print(f"  Delta-Hedged Straddle Sharpe: {sh_straddle:.3f}")

# 5. Put Spread (Risk Reversal / Put Spread Collar)
print("  Testing Put Spread (95/90)...")
put_spread_rets = []
for i in range(30, len(spy_px) - 30, 21):
    S = spy_px.iloc[i]
    K1 = S * 0.95  # Long put
    K2 = S * 0.90  # Short put
    T = 30/252
    rv = spy_ret_daily.iloc[max(0,i-21):i].std() * np.sqrt(252)
    iv = rv * 1.2
    
    long_put = black_scholes_put(S, K1, T, r, iv)
    short_put = black_scholes_put(S, K2, T, r, iv)
    net_cost = (long_put - short_put) / S
    
    next_px = spy_px.iloc[i+30]
    long_payoff = max(K1 - next_px, 0) / S
    short_payoff = -max(K2 - next_px, 0) / S
    
    total = (next_px/S - 1) + net_cost + long_payoff + short_payoff
    put_spread_rets.append(total)

put_spread_rets = pd.Series(put_spread_rets, index=spy_px.index[30::21][:len(put_spread_rets)])
sh_put_spread = sharpe(put_spread_rets)
print(f"  Put Spread (95/90) Sharpe: {sh_put_spread:.3f}")

# 6. VRP Capture (Sell ATM Straddle when IV > RV + threshold)
print("  Testing VRP Capture strategy...")
vrp_rets = []
for i in range(30, len(spy_px) - 30, 21):
    S = spy_px.iloc[i]
    K = S
    T = 30/252
    rv = spy_ret_daily.iloc[max(0,i-21):i].std() * np.sqrt(252)
    iv = rv * 1.2
    
    # Only sell if IV > RV + 2%
    if iv > rv + 0.02:
        call_px = black_scholes_call(S, K, T, r, iv)
        put_px = black_scholes_put(S, K, T, r, iv)
        premium = (call_px + put_px) / S
        
        next_px = spy_px.iloc[i+30]
        call_payoff = -max(next_px - K, 0) / S
        put_payoff = -max(K - next_px, 0) / S
        
        total = premium + call_payoff + put_payoff
    else:
        total = 0  # Sit in cash
    
    vrp_rets.append(total)

vrp_rets = pd.Series(vrp_rets, index=spy_px.index[30::21][:len(vrp_rets)])
sh_vrp = sharpe(vrp_rets)
print(f"  VRP Capture Sharpe: {sh_vrp:.3f}")

# Save options results
options_results = pd.DataFrame({
    'strategy': ['ProtectivePut', 'CoveredCall', 'Collar', 'DeltaHedgedStraddle', 'PutSpread', 'VRPCapture'],
    'sharpe': [sh_prot_put, sh_cov_call, sh_collar, sh_straddle, sh_put_spread, sh_vrp],
    'ann_ret': [r.mean()*252*100 for r in [prot_put_returns, cov_call_returns, collar_rets, straddle_pnl, put_spread_rets, vrp_rets]],
    'ann_vol': [r.std()*np.sqrt(252)*100 for r in [prot_put_returns, cov_call_returns, collar_rets, straddle_pnl, put_spread_rets, vrp_rets]],
    'max_dd': [max_dd((1+r).cumprod())*100 for r in [prot_put_returns, cov_call_returns, collar_rets, straddle_pnl, put_spread_rets, vrp_rets]]
})
options_results.to_csv('/root/quant/iter15_options_strategies.csv', index=False)

# ============================================================
# E6: COMPREHENSIVE PERFORMANCE SUMMARY
# ============================================================
print("\n=== E6: Comprehensive Performance Summary ===")

# Test all strategies on real data
all_strategies = {
    'SMA200': sma_trend(spy, 200),
    'VolTarget': vol_target(spy, target=0.10, window=21, lev_cap=2.0),
    'XSecMom': xsec_momentum(prices, top_n=3).sum(axis=1),  # Total weight
    'RSI2': __import__('strategies', fromlist=['rsi2_meanrev']).rsi2_meanrev(spy),
    'GEM': __import__('strategies', fromlist=['dual_momentum']).dual_momentum(prices[['SPY','GLD','TLT']]),
    'TSMOM': __import__('strategies', fromlist=['tsmom']).tsmom(spy),
    'MACross': __import__('strategies', fromlist=['ma_crossover']).ma_crossover(spy),
    'BuyHold': pd.Series(1.0, index=spy.index),
}

perf_results = []
for name, pos in all_strategies.items():
    pos = pos.reindex(spy_ret.index).fillna(0)
    if isinstance(pos, pd.DataFrame):
        pos = pos.sum(axis=1)
    ret = backtest(pos, spy_ret, cost_bps=10, lag=1)
    p = perf(ret, name)
    perf_results.append(p)

perf_df = pd.DataFrame(perf_results)
print(perf_df.to_string(index=False))
perf_df.to_csv('/root/quant/iter15_comprehensive_perf.csv', index=False)

# Walk-forward validation
print("\n  Walk-forward validation...")
wf_results = []
for fold in range(4):
    split = len(spy_ret) * fold // 4
    next_split = len(spy_ret) * (fold + 1) // 4
    train = spy_ret.iloc[:next_split]
    test = spy_ret.iloc[next_split:next_split + len(spy_ret)//4]
    
    if len(test) < 50:
        continue
    
    spy_train = spy.loc[train.index]
    spy_test = spy.loc[test.index]
    
    # Find best SMA window on train
    best_sh = -999
    best_w = 200
    for w in [50, 100, 150, 200, 250, 300]:
        pos = sma_trend(spy_train, w)
        ret = backtest(pos, spy_train.pct_change().dropna(), cost_bps=10)
        sh = sharpe(ret)
        if sh > best_sh:
            best_sh = sh
            best_w = w
    
    # Test on test
    pos_test = sma_trend(spy_test, best_w)
    ret_test = backtest(pos_test, spy_test.pct_change().dropna(), cost_bps=10)
    test_sh = sharpe(ret_test)
    
    wf_results.append({'fold': fold, 'best_window': best_w, 'train_sharpe': best_sh, 'test_sharpe': test_sh})

wf_df = pd.DataFrame(wf_results)
print(wf_df.to_string(index=False))
wf_df.to_csv('/root/quant/iter15_comprehensive_walkforward.csv', index=False)

# Purged K-Fold
print("\n  Purged K-Fold...")
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=False)
purge = 36  # ~1.5 months embargo
purged_sharpes = []

for train_idx, test_idx in kf.split(spy_ret):
    # Apply purge
    test_start = test_idx[0] + purge
    test_end = test_idx[-1]
    if test_start >= test_end:
        continue
    
    test_idx_purged = np.arange(test_start, test_end + 1)
    
    train_ret = spy_ret.iloc[train_idx]
    test_ret = spy_ret.iloc[test_idx_purged]
    
    spy_train = spy.loc[train_ret.index]
    spy_test = spy.loc[test_ret.index]
    
    pos_test = sma_trend(spy_test, 200)
    ret_test = backtest(pos_test, test_ret, cost_bps=10)
    purged_sharpes.append(sharpe(ret_test))

print(f"  Purged K-Fold Sharpes: {purged_sharpes}")
print(f"  Mean: {np.mean(purged_sharpes):.3f}, Std: {np.std(purged_sharpes):.3f}")

purged_df = pd.DataFrame({
    'fold_sharpes': purged_sharpes,
    'mean': [np.mean(purged_sharpes)],
    'std': [np.std(purged_sharpes)]
})
purged_df.to_csv('/root/quant/iter15_purged_cv.csv', index=False)

# Statistical validation
print("\n  Statistical validation...")
from scipy import stats

# Newey-West t-stat
for name, pos in all_strategies.items():
    pos = pos.reindex(spy_ret.index).fillna(0)
    if isinstance(pos, pd.DataFrame):
        pos = pos.sum(axis=1)
    ret = backtest(pos, spy_ret, cost_bps=10, lag=1).dropna()
    
    if len(ret) > 10:
        # Simple Newey-West with 12 lags
        n = len(ret)
        mean_ret = ret.mean()
        gamma0 = ret.var()
        
        # Autocovariances
        max_lag = min(12, n//4)
        nw_var = gamma0
        for h in range(1, max_lag+1):
            gamma_h = ret.autocorr(lag=h) * gamma0 if not np.isnan(ret.autocorr(lag=h)) else 0
            nw_var += 2 * (1 - h/(max_lag+1)) * gamma_h
        
        nw_se = np.sqrt(nw_var / n)
        nw_t = mean_ret / nw_se if nw_se > 0 else 0
        
        print(f"  {name}: NW_t = {nw_t:.3f}")

# Deflated Sharpe Ratio
def psr(sr, n, skew=0, kurt=3, benchmark=0):
    """Probabilistic Sharpe Ratio."""
    from scipy.stats import norm
    if n <= 1:
        return 0.5
    z = (sr - benchmark) * np.sqrt(n - 1) / np.sqrt(1 - skew*sr + (kurt-1)/4*sr**2)
    return norm.cdf(z)

# Cross-strategy DSR
all_sharpes = [p['Sharpe'] for p in perf_results]
sr_dispersion = np.std(all_sharpes)
print(f"\n  Strategy Sharpe dispersion: {sr_dispersion:.3f}")

for p in perf_results:
    dsr = psr(p['Sharpe'], len(spy_ret), benchmark=sr_dispersion)
    print(f"  {p['name']}: DSR(p>dispersion) = {dsr:.4f}")

val_df = pd.DataFrame(perf_results)
val_df.to_csv('/root/quant/iter15_comprehensive_validation.csv', index=False)

print("\n=== Iteration 15 Complete ===")
print("Outputs saved to /root/quant/iter15_*.csv and /root/quant/iter15_*.png")