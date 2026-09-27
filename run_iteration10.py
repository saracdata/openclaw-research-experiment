"""
Iteration #10 — QuantStart: Event-Driven Backtesting, Strategy Identification, Options Pricing, Portfolio Optimization
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import load, backtest, perf as performance
from strategies import *
import warnings
warnings.filterwarnings('ignore')

# Load data
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'AGG', 'VNQ', 'DBC', 'XLP', 'XLU', 'XLE', 'XLF', 'IEI', 'VIG', 'SCHD', 'MDY', 'XLK', 'XLV', 'VTI', 'VEA', 'VWO', 'GOVT', 'SHY', 'BIL', 'LQD', 'HYG', 'SMH']
price_df = pd.DataFrame({t: load(t)['Close'] for t in TICKERS}).dropna()
tickers = price_df.columns.tolist()

# ============================================================================
# E1. EVENT-DRIVEN BACKTESTER vs VECTORIZED
# ============================================================================
print("="*60)
print("E1: Event-Driven Backtester")
print("="*60)

class EventDrivenBacktester:
    """Simple event-driven backtester for daily data with order types."""
    def __init__(self, price_df, cost_bps=10):
        self.price_df = price_df
        self.cost_bps = cost_bps
        self.cash = 1_000_000
        self.positions = {t: 0 for t in price_df.columns}
        self.history = []
        
    def execute_order(self, ticker, order_type, quantity, price, date):
        """Execute order with slippage."""
        if order_type == 'market':
            exec_price = price * (1 + self.cost_bps/10000 * np.sign(quantity))
        elif order_type == 'limit':
            # Simplified: assume limit fills at price if favorable
            exec_price = price
        elif order_type == 'stop':
            exec_price = price * (1 + self.cost_bps/10000 * np.sign(quantity))
        else:
            exec_price = price
        cost = abs(quantity) * exec_price * self.cost_bps / 10000
        self.cash -= quantity * exec_price + cost
        self.positions[ticker] += quantity
        return exec_price, cost
    
    def run(self, signal_func, **kwargs):
        """Run backtest with signal function returning target weights."""
        dates = self.price_df.index
        portfolio_values = []
        
        for i, date in enumerate(dates):
            # Get target weights from signal function
            prices = self.price_df.iloc[:i+1]
            target_weights = signal_func(prices, **kwargs)
            
            if isinstance(target_weights, pd.Series):
                target_weights = target_weights.to_frame().T
            if target_weights.empty:
                target_weights = pd.DataFrame(0, index=[date], columns=self.price_df.columns)
            
            target_weights = target_weights.reindex(columns=self.price_df.columns).fillna(0)
            
            # Current portfolio value
            current_prices = self.price_df.loc[date]
            port_val = self.cash + sum(self.positions[t] * current_prices[t] for t in self.price_df.columns)
            
            # Calculate target positions
            target_positions = {}
            for t in self.price_df.columns:
                target_val = target_weights[t].iloc[-1] * port_val
                target_qty = int(target_val / current_prices[t])
                target_positions[t] = target_qty
            
            # Execute trades to reach target
            for t in self.price_df.columns:
                current_qty = self.positions[t]
                target_qty = target_positions[t]
                diff = target_qty - current_qty
                if diff != 0:
                    self.execute_order(t, 'market', diff, current_prices[t], date)
            
            # Record portfolio value
            port_val = self.cash + sum(self.positions[t] * current_prices[t] for t in self.price_df.columns)
            portfolio_values.append(port_val)
        
        return pd.Series(portfolio_values, index=dates)

# Test event-driven vs vectorized for SMA200
def sma_trend_signal(px, window=200):
    sma = px.rolling(window).mean()
    return (px > sma).astype(float).iloc[[-1]]

# Vectorized baseline
px = price_df['SPY']
vec_weights = sma_trend_signal(px)
vec_bt = backtest(px, vec_weights.squeeze(), cost_bps=10)
vec_perf = performance(vec_bt['net'])

# Event-driven
edb = EventDrivenBacktester(price_df[['SPY']], cost_bps=10)
ed_values = edb.run(sma_trend_signal, window=200)
ed_ret = ed_values.pct_change().fillna(0)
ed_sharpe = np.sqrt(252) * ed_ret.mean() / ed_ret.std()

print(f"Vectorized SMA200 Sharpe: {vec_perf['Sharpe']:.3f}")
print(f"Event-Driven SMA200 Sharpe: {ed_sharpe:.3f}")

# Save comparison
ed_comparison = pd.DataFrame({
    'Vectorized_Sharpe': [vec_perf['Sharpe']],
    'EventDriven_Sharpe': [ed_sharpe],
    'Difference': [vec_perf['Sharpe'] - ed_sharpe]
})
ed_comparison.to_csv('~/quant/iter10_event_driven.csv')
print("Saved iter10_event_driven.csv")

# ============================================================================
# E2. STRATEGY IDENTIFICATION: VALUE AVERAGING vs DCA vs BUY&HOLD
# ============================================================================
print("\n" + "="*60)
print("E2: Strategy Identification - Value Averaging")
print("="*60)

def value_averaging(px, target_growth=0.0005):  # 0.05% per day target
    """Value averaging: adjust position to meet target portfolio value growth."""
    target_val = 1_000_000
    positions = []
    cash = 1_000_000
    shares = 0
    
    for i, (date, price) in enumerate(px.items()):
        target_val *= (1 + target_growth)
        port_val = cash + shares * price
        target_shares = target_val / price
        trade = target_shares - shares
        cost = abs(trade) * price * 0.001  # 10 bps
        cash -= trade * price + cost
        shares = target_shares
        positions.append(shares * price)
    
    return pd.Series(positions, index=px.index)

def dollar_cost_averaging(px, monthly_invest=10000):
    """DCA: invest fixed amount monthly."""
    positions = []
    cash = 0
    shares = 0
    for i, (date, price) in enumerate(px.items()):
        if date.day == 1:  # first trading day of month
            trade = monthly_invest / price
            cost = monthly_invest * 0.001
            cash -= monthly_invest + cost
            shares += trade
        positions.append(shares * price)
    return pd.Series(positions, index=px.index)

spy = price_df['SPY']
va_vals = value_averaging(spy)
dca_vals = dollar_cost_averaging(spy)
bh_vals = spy / spy.iloc[0] * 1_000_000

# Performance
def calc_perf(vals):
    ret = vals.pct_change().fillna(0)
    ann_ret = (vals.iloc[-1] / vals.iloc[0]) ** (252/len(vals)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dd = (vals / vals.expanding().max() - 1).min()
    return {'AnnRet%': ann_ret*100, 'AnnVol%': ann_vol*100, 'Sharpe': sharpe, 'MaxDD%': dd*100}

va_perf = calc_perf(va_vals)
dca_perf = calc_perf(dca_vals)
bh_perf = calc_perf(bh_vals)

print(f"Value Averaging: {va_perf}")
print(f"DCA: {dca_perf}")
print(f"Buy & Hold: {bh_perf}")

strat_id = pd.DataFrame([va_perf, dca_perf, bh_perf], index=['ValueAveraging', 'DCA', 'BuyHold'])
strat_id.to_csv('~/quant/iter10_strategy_identification.csv')
print("Saved iter10_strategy_identification.csv")

# ============================================================================
# E3. BLACK-SCHOLES OPTIONS PRICING & IMPLIED VOLATILITY
# ============================================================================
print("\n" + "="*60)
print("E3: Black-Scholes & Implied Volatility")
print("="*60)

from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes European option price."""
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Greeks."""
    if T <= 0:
        return {}
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pdf = norm.pdf(d1)
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = pdf / (S * sigma * np.sqrt(T))
    theta = (-S*pdf*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) if option_type=='call' else \
            (-S*pdf*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))
    vega = S * pdf * np.sqrt(T)
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def implied_vol(price, S, K, T, r, option_type='call'):
    """Calculate implied volatility using Brent's method."""
    def obj(sig):
        return bs_price(S, K, T, r, sig, option_type) - price
    try:
        return brentq(obj, 0.001, 5.0)
    except:
        return np.nan

# Test with SPY and VIX proxy
# Use VIX as implied vol proxy (we don't have options chain)
# Simulate: ATM options on SPY with 30-day expiry
S = spy.iloc[-1]
K = S
T = 30/252
r = 0.04
# Use VIX data if available, else use realized vol
rv = spy.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252)
iv_proxy = rv  # In reality, IV > RV due to vol risk premium

call_price = bs_price(S, K, T, r, iv_proxy)
put_price = bs_price(S, K, T, r, iv_proxy, 'put')
greeks = bs_greeks(S, K, T, r, iv_proxy)

print(f"SPY: {S:.2f}, K: {K:.2f}, T: {T:.4f}, r: {r}, IV: {iv_proxy:.4f}")
print(f"Call Price: {call_price:.2f}, Put Price: {put_price:.2f}")
print(f"Greeks: {greeks}")

# Calculate implied vol from our theoretical price (should recover input)
iv_calc = implied_vol(call_price, S, K, T, r)
print(f"Implied Vol from call price: {iv_calc:.4f}")

# Volatility Risk Premium (VRP) estimation: IV - RV
# Use rolling 21-day realized vol as RV, VIX as IV proxy
# Since we don't have VIX, use RV * 1.2 as IV proxy (typical VRP ~20%)
vrp_series = []
for i in range(252, len(spy)):
    S = spy.iloc[i]
    rv = spy.iloc[i-252:i].pct_change().std() * np.sqrt(252)
    iv = rv * 1.2  # approximate VRP
    vrp_series.append(iv - rv)

vrp_mean = np.mean(vrp_series)
print(f"Average Volatility Risk Premium (IV - RV): {vrp_mean:.4f} ({vrp_mean*100:.2f}%)")

bs_results = pd.DataFrame({
    'SPY_Price': [S],
    'Strike': [K],
    'Time_to_Expiry': [T],
    'Rate': [r],
    'IV_Proxy': [iv_proxy],
    'Call_Price': [call_price],
    'Put_Price': [put_price],
    'Delta': [greeks['delta']],
    'Gamma': [greeks['gamma']],
    'Theta': [greeks['theta']],
    'Vega': [greeks['vega']],
    'VRP_Estimate': [vrp_mean]
})
bs_results.to_csv('~/quant/iter10_black_scholes.csv')
print("Saved iter10_black_scholes.csv")

# ============================================================================
# E4. PORTFOLIO OPTIMIZATION: MEAN-VARIANCE & BLACK-LITTERMAN
# ============================================================================
print("\n" + "="*60)
print("E4: Portfolio Optimization")
print("="*60)

from scipy.optimize import minimize

# Use 14 ETFs for optimization
etf_tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'AGG', 'VNQ', 'DBC', 'XLP', 'XLU', 'XLE']
etf_prices = price_df[etf_tickers].dropna()
returns = etf_prices.pct_change().dropna()

# Mean-Variance Optimization
mu = returns.mean() * 252
Sigma = returns.cov() * 252

def max_sharpe_portfolio(mu, Sigma, risk_free=0.02):
    n = len(mu)
    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ Sigma @ w)
        return -(port_ret - risk_free) / port_vol
    constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    bounds = [(0, 1) for _ in range(n)]  # long-only
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints)
    return res.x

def min_var_portfolio(Sigma):
    n = len(Sigma)
    def port_var(w):
        return w @ Sigma @ w
    constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    res = minimize(port_var, w0, bounds=bounds, constraints=constraints)
    return res.x

# Black-Litterman
def black_litterman(mu, Sigma, P, Q, Omega, tau=0.05, risk_free=0.02):
    """Black-Litterman model.
    P: K x n pick matrix
    Q: K x 1 view vector
    Omega: K x K uncertainty of views
    """
    n = len(mu)
    # Implied equilibrium returns
    weq = max_sharpe_portfolio(mu, Sigma, risk_free)
    pi = tau * Sigma @ weq  # Implied excess returns
    
    # BL formula
    M = np.linalg.inv(tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P
    mu_bl = np.linalg.inv(M) @ (np.linalg.inv(tau * Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    Sigma_bl = np.linalg.inv(M)
    
    # Optimize with BL returns
    w_bl = max_sharpe_portfolio(mu_bl, Sigma_bl, risk_free)
    return w_bl, mu_bl, Sigma_bl

# Define views: SPY outperforms TLT by 5%, GLD outperforms EFA by 3%
# P matrix: each row is a view
n = len(etf_tickers)
P = np.zeros((2, n))
P[0, etf_tickers.index('SPY')] = 1
P[0, etf_tickers.index('TLT')] = -1
P[1, etf_tickers.index('GLD')] = 1
P[1, etf_tickers.index('EFA')] = -1

Q = np.array([0.05, 0.03])  # 5% and 3% outperformance
Omega = np.diag([0.01**2, 0.01**2])  # uncertainty

w_mv = max_sharpe_portfolio(mu, Sigma)
w_minvar = min_var_portfolio(Sigma)
w_bl, mu_bl, Sigma_bl = black_litterman(mu, Sigma, P, Q, Omega)

# Equal weight
w_eq = np.ones(n) / n

# Backtest portfolios
def portfolio_backtest(weights, returns, cost_bps=10):
    """Simple monthly rebalance backtest."""
    port_ret = (returns @ weights).dropna()
    # Monthly rebalance: calculate turnover
    # Simplified: assume monthly rebalance to target weights
    # For simplicity, just use static weights
    turnover = 0  # ignore for now
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dd = ( (1+port_ret).cumprod() / (1+port_ret).cumprod().expanding().max() - 1 ).min()
    return {'AnnRet%': ann_ret*100, 'AnnVol%': ann_vol*100, 'Sharpe': sharpe, 'MaxDD%': dd*100, 'Weights': weights}

mv_perf = portfolio_backtest(w_mv, returns)
minvar_perf = portfolio_backtest(w_minvar, returns)
bl_perf = portfolio_backtest(w_bl, returns)
eq_perf = portfolio_backtest(w_eq, returns)

print(f"Mean-Variance: {mv_perf['Sharpe']:.3f} Sharpe, {mv_perf['AnnRet%']:.2f}% ret")
print(f"Min-Var: {minvar_perf['Sharpe']:.3f} Sharpe, {minvar_perf['AnnRet%']:.2f}% ret")
print(f"Black-Litterman: {bl_perf['Sharpe']:.3f} Sharpe, {bl_perf['AnnRet%']:.2f}% ret")
print(f"Equal-Weight: {eq_perf['Sharpe']:.3f} Sharpe, {eq_perf['AnnRet%']:.2f}% ret")

opt_results = pd.DataFrame([mv_perf, minvar_perf, bl_perf, eq_perf], 
                           index=['MeanVariance', 'MinVar', 'BlackLitterman', 'EqualWeight'])
opt_results = opt_results[['AnnRet%', 'AnnVol%', 'Sharpe', 'MaxDD%']]
opt_results.to_csv('~/quant/iter10_portfolio_optimization.csv')
print("Saved iter10_portfolio_optimization.csv")

# ============================================================================
# E5. BACKTESTING BEST PRACTICES (from QuantStart Part I/II)
# ============================================================================
print("\n" + "="*60)
print("E5: Backtesting Best Practices Validation")
print("="*60)

# Test various biases
# 1. Look-ahead bias: using future data in signal
def lookahead_signal(px, window=200):
    """WRONG: uses future data (shift(-1))"""
    sma = px.rolling(window).mean()
    return (px > sma.shift(-1)).astype(float)  # BUG: looking at tomorrow's SMA

def correct_signal(px, window=200):
    """CORRECT: uses only past data"""
    sma = px.rolling(window).mean()
    return (px > sma).astype(float)

# Test on SPY
la_weights = lookahead_signal(spy)
corr_weights = correct_signal(spy)

la_bt = backtest(spy, la_weights, cost_bps=10)
corr_bt = backtest(spy, corr_weights, cost_bps=10)

la_perf = performance(la_bt['net'])
corr_perf = performance(corr_bt['net'])

print(f"Look-ahead biased Sharpe: {la_perf['Sharpe']:.3f}")
print(f"Correct Sharpe: {corr_perf['Sharpe']:.3f}")
print(f"Inflation from look-ahead: {(la_perf['Sharpe'] - corr_perf['Sharpe'])/corr_perf['Sharpe']*100:.1f}%")

# 2. Survivorship bias: test on current S&P 500 constituents only
# We already did this in iteration 2, but let's test with a hypothetical delisted stock
# Create a "delisted" stock that went to zero
np.random.seed(42)
fake_delisted = spy.copy()
fake_delisted.iloc[-500:] = fake_delisted.iloc[-500:] * np.linspace(1, 0, 500)  # goes to zero
fake_prices = pd.DataFrame({'SPY': spy, 'DELISTED': fake_delisted})
fake_returns = fake_prices.pct_change().dropna()

# Equal weight including delisted
w_fake = np.array([0.5, 0.5])
fake_port = (fake_returns @ w_fake).dropna()
fake_sharpe = fake_port.mean() * np.sqrt(252) / fake_port.std()
real_sharpe = spy.pct_change().dropna().mean() * np.sqrt(252) / spy.pct_change().dropna().std()

print(f"With delisted stock (survivorship bias): Sharpe = {fake_sharpe:.3f}")
print(f"Without (survivor only): Sharpe = {real_sharpe:.3f}")

# 3. Data-snooping: test 100 random strategies, see best Sharpe
np.random.seed(123)
random_sharpes = []
for _ in range(100):
    # Random moving average crossover
    fast = np.random.randint(5, 50)
    slow = np.random.randint(50, 200)
    if fast >= slow:
        fast, slow = slow, fast
    sig = ma_crossover(spy, fast, slow)
    bt = backtest(spy, sig, cost_bps=10)
    pf = performance(bt['net'])
    random_sharpes.append(pf['Sharpe'])

print(f"Best of 100 random MA crossovers: {max(random_sharpes):.3f}")
print(f"Median of 100 random: {np.median(random_sharpes):.3f}")
print(f"Expected max by chance (Bonferroni): {norm.ppf(1 - 0.05/100):.3f}")

bp_results = pd.DataFrame({
    'Lookahead_Sharpe': [la_perf['Sharpe']],
    'Correct_Sharpe': [corr_perf['Sharpe']],
    'Inflation_Pct': [(la_perf['Sharpe'] - corr_perf['Sharpe'])/corr_perf['Sharpe']*100],
    'Survivorship_Sharpe': [fake_sharpe],
    'Survivor_Only_Sharpe': [real_sharpe],
    'Best_Random_Sharpe': [max(random_sharpes)],
    'Median_Random_Sharpe': [np.median(random_sharpes)]
})
bp_results.to_csv('~/quant/iter10_backtest_practices.csv')
print("Saved iter10_backtest_practices.csv")

# ============================================================================
# PLOTS
# ============================================================================
print("\nGenerating plots...")

# Equity curves for strategy identification
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(va_vals / 1e6, label='Value Averaging')
plt.plot(dca_vals / 1e6, label='DCA')
plt.plot(bh_vals / 1e6, label='Buy & Hold')
plt.title('Strategy Identification: Value Averaging vs DCA vs Buy&Hold')
plt.ylabel('Portfolio Value ($M)')
plt.legend()
plt.grid(True, alpha=0.3)

# Portfolio optimization weights
plt.subplot(2, 2, 2)
x = range(n)
plt.bar([i - 0.3 for i in x], w_mv, 0.3, label='Mean-Var')
plt.bar([i for i in x], w_bl, 0.3, label='Black-Litterman')
plt.bar([i + 0.3 for i in x], w_eq, 0.3, label='Equal-Weight')
plt.xticks(x, etf_tickers, rotation=90)
plt.title('Portfolio Weights')
plt.legend()
plt.grid(True, alpha=0.3)

# Black-Scholes Greeks
plt.subplot(2, 2, 3)
S_range = np.linspace(S*0.8, S*1.2, 50)
call_prices = [bs_price(s, K, T, r, iv_proxy) for s in S_range]
put_prices = [bs_price(s, K, T, r, iv_proxy, 'put') for s in S_range]
plt.plot(S_range, call_prices, label='Call')
plt.plot(S_range, put_prices, label='Put')
plt.axvline(S, color='k', linestyle='--', label='Current SPY')
plt.title('Black-Scholes Option Prices vs Spot')
plt.xlabel('Spot Price')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Random strategy distribution
plt.subplot(2, 2, 4)
plt.hist(random_sharpes, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(corr_perf['Sharpe'], color='red', linestyle='--', label='SMA200')
plt.axvline(max(random_sharpes), color='orange', linestyle='--', label='Best Random')
plt.title('Distribution of 100 Random MA Crossover Sharpes')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('~/quant/iter10_equity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved iter10_equity.png")

# ============================================================================
# COMPREHENSIVE PERFORMANCE SUMMARY
# ============================================================================
print("\n" + "="*60)
print("ITERATION #10 COMPREHENSIVE PERFORMANCE")
print("="*60)

# Run vectorized backtests for all main strategies with our engine
strategies = {
    'SMA200': lambda px: sma_trend(px, 200),
    'GEM': lambda px: dual_momentum(price_df[['SPY','GLD','TLT','IEF','SHY']], 126).get('SPY', pd.Series(0, index=px.index)),
    'XSecMom': lambda px: xsec_momentum(price_df[etf_tickers], 252, 21, 3).get('SPY', pd.Series(0, index=px.index)),
    'TSMOM': lambda px: tsmom(px, 252, 21),
    'RSI2': lambda px: rsi2_meanrev(px),
    'VolTarget': lambda px: vol_target(px, 0.10, 21, 2.0),
    'MACross': lambda px: ma_crossover(px, 50, 200),
    'Rev5': lambda px: short_term_reversal(price_df[etf_tickers], 5).get('SPY', pd.Series(0, index=px.index)),
}

comp_results = []
for name, func in strategies.items():
    try:
        weights = func(price_df['SPY'] if name not in ['GEM','XSecMom','Rev5'] else price_df)
        if isinstance(weights, pd.DataFrame):
            weights = weights['SPY'] if 'SPY' in weights.columns else weights.iloc[:, 0]
        bt = backtest(price_df['SPY'], weights, cost_bps=10)
        pf = performance(bt['net'])
        comp_results.append({
            'Strategy': name,
            'AnnRet%': pf['AnnRet%'],
            'AnnVol%': pf['AnnVol%'],
            'Sharpe': pf['Sharpe'],
            'MaxDD%': pf['MaxDD%'],
            'Calmar': pf['Calmar']
        })
    except Exception as e:
        print(f"Error with {name}: {e}")
        comp_results.append({
            'Strategy': name,
            'AnnRet%': np.nan, 'AnnVol%': np.nan, 'Sharpe': np.nan, 'MaxDD%': np.nan, 'Calmar': np.nan
        })

comp_df = pd.DataFrame(comp_results)
comp_df.to_csv('~/quant/iter10_comprehensive_perf.csv', index=False)
print(comp_df.to_string(index=False))

# Walk-forward validation
print("\nWalk-forward validation...")
wf_results = []
n_folds = 4
fold_size = len(spy) // n_folds
for fold in range(n_folds):
    start = fold * fold_size
    end = (fold + 1) * fold_size if fold < n_folds - 1 else len(spy)
    train = spy.iloc[:start]
    test = spy.iloc[start:end]
    
    # Optimize SMA window on train
    best_window = 200
    best_sharpe = -np.inf
    for w in [50, 100, 150, 200, 250, 300]:
        sig = sma_trend(train, w)
        bt = backtest(train, sig, cost_bps=10)
        pf = performance(bt['net'])
        if pf['Sharpe'] > best_sharpe:
            best_sharpe = pf['Sharpe']
            best_window = w
    
    # Test on test
    sig_test = sma_trend(test, best_window)
    bt_test = backtest(test, sig_test, cost_bps=10)
    pf_test = performance(bt_test['net'])
    wf_results.append({'Fold': fold+1, 'Best_Window': best_window, 'Train_Sharpe': best_sharpe, 'Test_Sharpe': pf_test['Sharpe']})

wf_df = pd.DataFrame(wf_results)
wf_df.to_csv('~/quant/iter10_comprehensive_walkforward.csv', index=False)
print(wf_df.to_string(index=False))

# Validation with purged CV (simple version)
print("\nPurged K-Fold validation...")
from sklearn.model_selection import KFold

def purged_kfold_sharpe(returns, n_splits=4, embargo_pct=0.01):
    """Simple purged K-fold for returns series."""
    n = len(returns)
    embargo = int(n * embargo_pct)
    kf = KFold(n_splits=n_splits, shuffle=False)
    sharpes = []
    
    for train_idx, test_idx in kf.split(returns):
        # Apply embargo
        test_start = test_idx[0]
        test_end = test_idx[-1]
        train_idx = train_idx[train_idx < test_start - embargo]
        
        if len(train_idx) > 100 and len(test_idx) > 100:
            train_ret = returns.iloc[train_idx]
            test_ret = returns.iloc[test_idx]
            
            # Simple strategy: sign of mean return
            signal = np.sign(train_ret.mean())
            strat_ret = test_ret * signal
            sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
            sharpes.append(sharpe)
    
    return sharpes

spy_ret = spy.pct_change().dropna()
purged_sharpes = purged_kfold_sharpe(spy_ret)
print(f"Purged K-Fold Sharpes: {[f'{s:.3f}' for s in purged_sharpes]}")
print(f"Mean: {np.mean(purged_sharpes):.3f}, Std: {np.std(purged_sharpes):.3f}")

val_df = pd.DataFrame({
    'Purged_Fold_Sharpe': purged_sharpes,
    'Mean': [np.mean(purged_sharpes)] * len(purged_sharpes),
    'Std': [np.std(purged_sharpes)] * len(purged_sharpes)
})
val_df.to_csv('~/quant/iter10_comprehensive_validation.csv', index=False)
print("Saved iter10_comprehensive_validation.csv")

print("\n" + "="*60)
print("ITERATION #10 COMPLETE")
print("="*60)
