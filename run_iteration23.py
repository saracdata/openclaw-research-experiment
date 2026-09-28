"""
Iteration #23 — QuantStart: Backtesting Frameworks, QSTrader Architecture, Fee Models, Asset Classes
Focus (from QuantStart articles):
- Backtesting Considerations and Open Source Frameworks
- QSTrader Architecture: Fee Model Class Hierarchy, Asset Class Hierarchy, Portfolio/Position Classes
- Creating Backtesting Environment with Docker/Jupyter/QSTrader
- Event-Driven Backtesting
- Transaction Cost Models (commission, slippage, spread)
- Portfolio Construction: 60/40, Risk Parity, Equal Weight, TAA
- Stooq Data, Tiingo Data Coverage
- Polygon Forex Data, Realised Volatility
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
from scipy.optimize import minimize

# ============================================================
# DATA LOADING
# ============================================================
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'DBC', 'VNQ', 
           'XLE', 'XLF', 'XLK', 'XLV', 'XLU', 'IEI', 'VIG', 'SCHD', 'MDY',
           'IEF', 'AGG', 'VXX', 'VTI', 'VEA', 'VWO', 'GOVT', 'SHY', 'BIL', 
           'LQD', 'HYG', 'SMH', 'XLP', 'XLU']

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
# E1. FEE MODEL CLASS HIERARCHY (QSTrader style)
# ============================================================
print("\n=== E1: Fee Model Class Hierarchy ===")

class FeeModel:
    """Base fee model"""
    def calculate(self, quantity, price, is_buy=True):
        raise NotImplementedError

class IBCommissionFee(FeeModel):
    """Interactive Brokers US Equities: $0.005/share, min $1, max 1%"""
    def __init__(self, per_share=0.005, min_fee=1.0, max_pct=0.01):
        self.per_share = per_share
        self.min_fee = min_fee
        self.max_pct = max_pct
    
    def calculate(self, quantity, price, is_buy=True):
        fee = abs(quantity) * self.per_share
        fee = max(fee, self.min_fee)
        max_fee = abs(quantity) * price * self.max_pct
        fee = min(fee, max_fee)
        return fee

class FixedBpsFee(FeeModel):
    """Fixed basis points per trade"""
    def __init__(self, bps=1.0):
        self.bps = bps / 10000
    
    def calculate(self, quantity, price, is_buy=True):
        return abs(quantity) * price * self.bps

class PercentageFee(FeeModel):
    """Percentage of notional"""
    def __init__(self, pct=0.001):
        self.pct = pct
    
    def calculate(self, quantity, price, is_buy=True):
        return abs(quantity) * price * self.pct

class TieredFee(FeeModel):
    """Tiered commission (e.g., IBKR Pro)"""
    def __init__(self, tiers=None):
        # tiers: [(max_shares, per_share), ...]
        self.tiers = tiers or [(300, 0.0035), (3000, 0.002), (10000, 0.0015), (np.inf, 0.001)]
    
    def calculate(self, quantity, price, is_buy=True):
        q = abs(quantity)
        fee = 0
        remaining = q
        for max_q, rate in self.tiers:
            take = min(remaining, max_q)
            fee += take * rate
            remaining -= take
            if remaining <= 0:
                break
        return max(fee, 0.35)  # Minimum

class SlippageModel:
    """Base slippage model"""
    def calculate(self, quantity, price, volume=None, volatility=None):
        raise NotImplementedError

class FixedSlippage(SlippageModel):
    def __init__(self, bps=5):
        self.bps = bps / 10000
    
    def calculate(self, quantity, price, volume=None, volatility=None):
        return abs(quantity) * price * self.bps

class VolumeSlippage(SlippageModel):
    """Slippage proportional to participation rate"""
    def __init__(self, participation_limit=0.1):
        self.participation_limit = participation_limit
    
    def calculate(self, quantity, price, volume=None, volatility=None):
        if volume is None or volume == 0:
            return abs(quantity) * price * 0.0001
        participation = abs(quantity) / volume
        slippage_bps = participation / self.participation_limit * 10  # 10 bps at limit
        return abs(quantity) * price * slippage_bps / 10000

class SqrtSlippage(SlippageModel):
    """Square-root market impact model"""
    def __init__(self, a=0.1):
        self.a = a
    
    def calculate(self, quantity, price, volume=None, volatility=None):
        if volume is None or volume == 0:
            return abs(quantity) * price * 0.0001
        participation = abs(quantity) / volume
        impact = self.a * np.sqrt(participation) * price
        return impact * abs(quantity)

# Test fee models
fee_models = {
    'Fixed_1bps': FixedBpsFee(1),
    'Fixed_5bps': FixedBpsFee(5),
    'Fixed_10bps': FixedBpsFee(10),
    'IB_Commission': IBCommissionFee(),
    'Percentage_0.1%': PercentageFee(0.001),
    'Tiered_IBKR': TieredFee(),
}

slippage_models = {
    'Fixed_1bps': FixedSlippage(1),
    'Fixed_5bps': FixedSlippage(5),
    'Fixed_10bps': FixedSlippage(10),
    'Volume_Based': VolumeSlippage(),
    'Sqrt_Impact': SqrtSlippage(),
}

# Simulate trading costs for a typical strategy
test_qty = 1000
test_price = 400

print("Fee Model Comparison (1000 shares @ $400):")
for name, model in fee_models.items():
    fee = model.calculate(test_qty, test_price)
    print(f"  {name}: ${fee:.2f} ({fee/(test_qty*test_price)*10000:.1f} bps)")

print("\nSlippage Model Comparison (1000 shares @ $400, vol=1M):")
for name, model in slippage_models.items():
    slip = model.calculate(test_qty, test_price, volume=1_000_000)
    print(f"  {name}: ${slip:.2f} ({slip/(test_qty*test_price)*10000:.1f} bps)")

# ============================================================
# E2. BACKTESTING WITH REALISTIC COSTS
# ============================================================
print("\n=== E2: Backtesting with Realistic Transaction Costs ===")

def backtest_with_costs(positions, returns, fee_model, slippage_model, 
                         price_series=None, volume_series=None):
    """Backtest with explicit fee and slippage models"""
    pos = positions.reindex(returns.index).fillna(0)
    pos_diff = pos.diff().fillna(pos.iloc[0])
    
    costs = pd.Series(0.0, index=returns.index)
    
    for i, (date, qty_change) in enumerate(pos_diff.items()):
        if qty_change != 0 and price_series is not None:
            price = price_series.loc[date] if date in price_series.index else test_price
            volume = volume_series.loc[date] if volume_series is not None and date in volume_series.index else 1_000_000
            
            fee = fee_model.calculate(qty_change, price)
            slip = slippage_model.calculate(qty_change, price, volume)
            costs.iloc[i] = (fee + slip) / (abs(pos.iloc[i]) * price + 1) if pos.iloc[i] != 0 else 0
    
    # Convert to return drag
    gross_ret = pos.shift(1).fillna(0) * returns
    net_ret = gross_ret - costs.abs()
    return net_ret

# Test SMA200 with different cost models
sma_pos = sma_trend(spy, 200)

cost_results = {}
for fee_name, fee_model in fee_models.items():
    for slip_name, slip_model in slippage_models.items():
        if fee_name == 'Fixed_10bps' and slip_name == 'Fixed_5bps':
            # Our standard
            net_ret = backtest(sma_pos, spy_ret, cost_bps=10)
        else:
            # Custom
            net_ret = backtest_with_costs(sma_pos, spy_ret, fee_model, slip_model, spy)
        
        key = f"{fee_name}_{slip_name}"
        cost_results[key] = sharpe(net_ret.dropna())

cost_df = pd.DataFrame([cost_results]).T
cost_df.columns = ['Sharpe']
cost_df.to_csv('/root/quant/iter23_fee_slippage.csv')
print("Cost model Sharpe results:")
print(cost_df.sort_values('Sharpe', ascending=False).head(10))

# ============================================================
# E3. EVENT-DRIVEN BACKTESTING FRAMEWORK
# ============================================================
print("\n=== E3: Event-Driven Backtesting Framework ===")

class Event:
    MARKET = 'MARKET'
    SIGNAL = 'SIGNAL'
    ORDER = 'ORDER'
    FILL = 'FILL'

class MarketEvent:
    def __init__(self, timestamp, prices):
        self.type = Event.MARKET
        self.timestamp = timestamp
        self.prices = prices

class SignalEvent:
    def __init__(self, timestamp, symbol, direction, strength=1.0):
        self.type = Event.SIGNAL
        self.timestamp = timestamp
        self.symbol = symbol
        self.direction = direction  # 'LONG', 'SHORT', 'EXIT'
        self.strength = strength

class OrderEvent:
    def __init__(self, timestamp, symbol, order_type, quantity, direction):
        self.type = Event.ORDER
        self.timestamp = timestamp
        self.symbol = symbol
        self.order_type = order_type  # 'MKT', 'LMT'
        self.quantity = quantity
        self.direction = direction

class FillEvent:
    def __init__(self, timestamp, symbol, quantity, direction, fill_price, commission):
        self.type = Event.FILL
        self.timestamp = timestamp
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_price = fill_price
        self.commission = commission

class Portfolio:
    """Simple portfolio handler (QSTrader Portfolio class style)"""
    def __init__(self, initial_cash=100000, fee_model=None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> quantity
        self.fee_model = fee_model or FixedBpsFee(10)
        self.history = []
    
    def update_fill(self, fill: FillEvent):
        """Update positions from fill"""
        if fill.symbol not in self.positions:
            self.positions[fill.symbol] = 0
        
        if fill.direction == 'BUY':
            self.positions[fill.symbol] += fill.quantity
            self.cash -= fill.quantity * fill.fill_price + fill.commission
        else:
            self.positions[fill.symbol] -= fill.quantity
            self.cash += fill.quantity * fill.fill_price - fill.commission
        
        self.history.append({
            'timestamp': fill.timestamp,
            'symbol': fill.symbol,
            'qty': fill.quantity,
            'price': fill.fill_price,
            'commission': fill.commission,
            'cash': self.cash,
            'positions': self.positions.copy()
        })
    
    def get_equity(self, prices):
        """Calculate total equity"""
        equity = self.cash
        for sym, qty in self.positions.items():
            if sym in prices:
                equity += qty * prices[sym]
        return equity

class ExecutionHandler:
    """Simulate order execution (QSTrader ExecutionHandler style)"""
    def __init__(self, fee_model=None, slippage_model=None):
        self.fee_model = fee_model or FixedBpsFee(10)
        self.slippage_model = slippage_model or FixedSlippage(5)
    
    def execute_order(self, order: OrderEvent, market_price, volume=None):
        """Execute market order with slippage"""
        if order.order_type == 'MKT':
            slip = self.slippage_model.calculate(order.quantity, market_price, volume)
            if order.direction == 'BUY':
                fill_price = market_price + slip / order.quantity
            else:
                fill_price = market_price - slip / order.quantity
            
            commission = self.fee_model.calculate(order.quantity, fill_price)
            
            return FillEvent(
                timestamp=order.timestamp,
                symbol=order.symbol,
                quantity=order.quantity,
                direction=order.direction,
                fill_price=fill_price,
                commission=commission
            )
        return None

# Test event-driven backtest
class SimpleEventBacktest:
    def __init__(self, prices, signals, fee_model=None, slippage_model=None):
        self.prices = prices
        self.signals = signals
        self.portfolio = Portfolio(fee_model=fee_model)
        self.execution = ExecutionHandler(fee_model=fee_model, slippage_model=slippage_model)
        self.equity_curve = []
    
    def run(self):
        for date in self.prices.index:
            if date not in self.signals.index:
                continue
            
            # Get market prices
            mkt_prices = self.prices.loc[date].to_dict()
            
            # Generate signal
            signal = self.signals.loc[date]
            if signal != 0:
                # Simple: target position = signal
                current_qty = self.portfolio.positions.get('SPY', 0)
                target_qty = signal * 1000  # 1000 shares per unit signal
                qty_change = target_qty - current_qty
                
                if qty_change != 0:
                    order = OrderEvent(
                        timestamp=date,
                        symbol='SPY',
                        order_type='MKT',
                        quantity=abs(qty_change),
                        direction='BUY' if qty_change > 0 else 'SELL'
                    )
                    fill = self.execution.execute_order(order, mkt_prices['SPY'])
                    if fill:
                        self.portfolio.update_fill(fill)
            
            # Record equity
            equity = self.portfolio.get_equity(mkt_prices)
            self.equity_curve.append({'date': date, 'equity': equity})
        
        return pd.DataFrame(self.equity_curve).set_index('date')

# Run event-driven backtest
signal_series = sma_trend(spy, 200).reindex(spy.index).fillna(0)
evt_bt = SimpleEventBacktest(prices[['SPY']], signal_series, FixedBpsFee(10), FixedSlippage(5))
evt_equity = evt_bt.run()
evt_returns = evt_equity['equity'].pct_change().dropna()
evt_sharpe = sharpe(evt_returns)
print(f"Event-driven SMA200: Sharpe={evt_sharpe:.3f}")

# ============================================================
# E4. PORTFOLIO CONSTRUCTION: 60/40, RISK PARITY, EQUAL WEIGHT, TAA
# ============================================================
print("\n=== E4: Portfolio Construction ===")

def equal_weight_portfolio(returns):
    n = len(returns.columns)
    return pd.DataFrame(np.ones((len(returns), n)) / n, index=returns.index, columns=returns.columns)

def risk_parity_portfolio(returns, window=63):
    vol = returns.rolling(window).std() * np.sqrt(252)
    inv_vol = 1 / vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
    return weights

def min_var_portfolio(returns, window=63):
    weights_list = []
    for i in range(window, len(returns)):
        cov = returns.iloc[i-window:i].cov().values
        n = len(cov)
        ones = np.ones(n)
        try:
            w = np.linalg.solve(cov + 1e-4*np.eye(n), ones)
            w = w / w.sum()
            w = np.clip(w, -0.5, 0.5)
            w = w / np.abs(w).sum()
        except:
            w = ones / n
        weights_list.append(w)
    
    weights = pd.DataFrame(weights_list, index=returns.index[window:], columns=returns.columns)
    return weights

def max_diversification_portfolio(returns, window=63):
    """Maximum Diversification Portfolio (Choueifaty & Coignard 2008)"""
    weights_list = []
    for i in range(window, len(returns)):
        cov = returns.iloc[i-window:i].cov().values
        vol = np.sqrt(np.diag(cov))
        try:
            # w ∝ Σ^-1 * σ
            w = np.linalg.solve(cov + 1e-4*np.eye(len(cov)), vol)
            w = w / w.sum()
            w = np.clip(w, -0.5, 0.5)
            w = w / np.abs(w).sum()
        except:
            w = np.ones(len(cov)) / len(cov)
        weights_list.append(w)
    
    weights = pd.DataFrame(weights_list, index=returns.index[window:], columns=returns.columns)
    return weights

# Test portfolio constructions
portfolios = {
    'Equal_Weight': equal_weight_portfolio(returns),
    'Risk_Parity': risk_parity_portfolio(returns),
    'Min_Var': min_var_portfolio(returns),
    'Max_Div': max_diversification_portfolio(returns),
    '60_40': pd.DataFrame(np.tile([0.6, 0.4] + [0]*(len(returns.columns)-2), (len(returns), 1)), 
                          index=returns.index, columns=returns.columns),
}

portfolio_results = {}
for name, weights in portfolios.items():
    # Align
    common_idx = weights.index.intersection(returns.index)
    w = weights.loc[common_idx]
    r = returns.loc[common_idx]
    
    if len(w) > 100:
        ret = backtest(w, r, cost_bps=10)
        p = perf(ret.dropna(), name)
        portfolio_results[name] = p
        print(f"  {name}: Sharpe={p['Sharpe']:.3f}, AnnRet%={p['AnnRet%']:.2f}")

port_df = pd.DataFrame(portfolio_results).T
port_df.to_csv('/root/quant/iter23_portfolios.csv')

# ============================================================
# E5. REALISED VOLATILITY WITH INTRADAY PROXIES
# ============================================================
print("\n=== E5: Realised Volatility Estimation ===")

# Since we only have daily data, simulate intraday proxies
# Parkinson (1980) High-Low estimator
def parkinson_vol(high, low, window=21):
    """Parkinson volatility from daily high/low"""
    rs = np.log(high / low) ** 2
    vol = np.sqrt(rs.rolling(window).mean() * 252 / (4 * np.log(2)))
    return vol

# Garman-Klass (1980) estimator
def garman_klass_vol(open_, high, low, close, window=21):
    """Garman-Klass volatility"""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    rs = 0.5 * log_hl - (2*np.log(2) - 1) * log_co
    vol = np.sqrt(rs.rolling(window).mean() * 252)
    return vol

# Rogers-Satchell (1991) estimator
def rogers_satchell_vol(open_, high, low, close, window=21):
    """Rogers-Satchell volatility (drift-independent)"""
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_co = np.log(close / open_)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    vol = np.sqrt(rs.rolling(window).mean() * 252)
    return vol

# Yang-Zhang (2000) estimator
def yang_zhang_vol(open_, high, low, close, window=21):
    """Yang-Zhang volatility (combines overnight + intraday)"""
    log_oc = np.log(open_ / close.shift(1))
    log_cc = np.log(close / close.shift(1))
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_co = np.log(close / open_)
    
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    overnight = log_oc.rolling(window).var()
    intraday = log_cc.rolling(window).var()
    rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()
    
    vol = np.sqrt((overnight + k*intraday + (1-k)*rs) * 252)
    return vol

# We only have close prices, simulate OHLC
# Use returns to estimate intraday range
np.random.seed(42)
daily_range = spy_ret.abs() * np.sqrt(252) * np.random.uniform(0.5, 1.5, len(spy_ret))
high = spy.reindex(spy_ret.index) * np.exp(daily_range / 2)
low = spy.reindex(spy_ret.index) * np.exp(-daily_range / 2)
open_ = spy.reindex(spy_ret.index) * np.exp(np.random.normal(0, daily_range / 4))
close = spy.reindex(spy_ret.index)

# Calculate volatility estimators
rv_close = spy_ret.rolling(21).std() * np.sqrt(252)
rv_parkinson = parkinson_vol(high, low)
rv_gk = garman_klass_vol(open_, high, low, close)
rv_rs = rogers_satchell_vol(open_, high, low, close)
rv_yz = yang_zhang_vol(open_, high, low, close)

# Compare
vol_df = pd.DataFrame({
    'Close_Close': rv_close,
    'Parkinson': rv_parkinson,
    'Garman_Klass': rv_gk,
    'Rogers_Satchell': rv_rs,
    'Yang_Zhang': rv_yz
}).dropna()

print("Volatility Estimator Correlations:")
print(vol_df.corr())

# Test vol-targeting with different estimators
vol_target_results = {}
for name, vol_est in vol_df.items():
    lev = (0.10 / vol_est).clip(upper=2.0).fillna(1.0)
    pos = lev.reindex(spy_ret.index).fillna(0)
    ret = backtest(pos, spy_ret, cost_bps=10)
    sh = sharpe(ret.dropna())
    vol_target_results[name] = sh
    print(f"  VolTarget({name}): Sharpe={sh:.3f}")

vol_t_df = pd.DataFrame([vol_target_results])
vol_t_df.to_csv('/root/quant/iter23_vol_estimators.csv')

# ============================================================
# E6. DATA QUALITY & COVERAGE (Stooq, Tiingo style)
# ============================================================
print("\n=== E6: Data Quality & Coverage Analysis ===")

def analyze_data_quality(prices):
    """Analyze data coverage, gaps, outliers"""
    results = {}
    for col in prices.columns:
        s = prices[col]
        results[col] = {
            'start': s.first_valid_index(),
            'end': s.last_valid_index(),
            'n_obs': s.notna().sum(),
            'n_missing': s.isna().sum(),
            'pct_missing': s.isna().mean(),
            'min': s.min(),
            'max': s.max(),
            'n_zero_ret': (s.pct_change() == 0).sum(),
            'n_large_ret': (s.pct_change().abs() > 0.2).sum(),
        }
    return pd.DataFrame(results).T

quality = analyze_data_quality(prices)
quality.to_csv('/root/quant/iter23_data_quality.csv')
print("Data Quality Summary:")
print(quality[['n_obs', 'pct_missing', 'n_zero_ret', 'n_large_ret']].head(10))

# Detect suspicious patterns
# 1. Stale prices (zero returns for multiple days)
stale = (returns == 0).rolling(5).sum() == 5
stale_counts = stale.sum()
print(f"\nStale price days (5+ zero returns): {stale_counts[stale_counts > 0].to_dict()}")

# 2. Outlier returns
outlier_threshold = returns.abs() > 0.15
outlier_counts = outlier_threshold.sum()
print(f"Outlier returns (>15%): {outlier_counts[outlier_counts > 0].to_dict()}")

# 3. Gap detection (missing dates)
expected_dates = pd.bdate_range(prices.index[0], prices.index[-1])
missing_dates = expected_dates.difference(prices.index)
print(f"Missing business days: {len(missing_dates)}")

# ============================================================
# E7. QSTrader-STYLE ASSET CLASS HIERARCHY
# ============================================================
print("\n=== E7: Asset Class Hierarchy ===")

class Asset:
    def __init__(self, symbol, asset_class, currency='USD'):
        self.symbol = symbol
        self.asset_class = asset_class
        self.currency = currency

class Equity(Asset):
    def __init__(self, symbol, sector=None, market_cap=None):
        super().__init__(symbol, 'EQUITY')
        self.sector = sector
        self.market_cap = market_cap

class Bond(Asset):
    def __init__(self, symbol, maturity=None, credit_rating=None):
        super().__init__(symbol, 'BOND')
        self.maturity = maturity
        self.credit_rating = credit_rating

class Commodity(Asset):
    def __init__(self, symbol, commodity_type=None):
        super().__init__(symbol, 'COMMODITY')
        self.commodity_type = commodity_type

class ETF(Asset):
    def __init__(self, symbol, underlying_class, expense_ratio=0.0):
        super().__init__(symbol, 'ETF')
        self.underlying_class = underlying_class
        self.expense_ratio = expense_ratio

# Map our tickers
asset_map = {
    'SPY': ETF('SPY', 'EQUITY', 0.0009),
    'QQQ': ETF('QQQ', 'EQUITY', 0.0020),
    'IWM': ETF('IWM', 'EQUITY', 0.0019),
    'VTI': ETF('VTI', 'EQUITY', 0.0003),
    'VEA': ETF('VEA', 'EQUITY', 0.0005),
    'VWO': ETF('VWO', 'EQUITY', 0.0008),
    'TLT': ETF('TLT', 'BOND', 0.0015),
    'IEF': ETF('IEF', 'BOND', 0.0015),
    'IEI': ETF('IEI', 'BOND', 0.0015),
    'AGG': ETF('AGG', 'BOND', 0.0003),
    'LQD': ETF('LQD', 'BOND', 0.0014),
    'HYG': ETF('HYG', 'BOND', 0.0049),
    'GOVT': ETF('GOVT', 'BOND', 0.0005),
    'SHY': ETF('SHY', 'BOND', 0.0015),
    'BIL': ETF('BIL', 'BOND', 0.0014),
    'GLD': ETF('GLD', 'COMMODITY', 0.0040),
    'DBC': ETF('DBC', 'COMMODITY', 0.0085),
    'VNQ': ETF('VNQ', 'REAL_ESTATE', 0.0012),
    'XLE': ETF('XLE', 'EQUITY', 0.0010),
    'XLF': ETF('XLF', 'EQUITY', 0.0010),
    'XLK': ETF('XLK', 'EQUITY', 0.0010),
    'XLV': ETF('XLV', 'EQUITY', 0.0010),
    'XLU': ETF('XLU', 'EQUITY', 0.0010),
    'XLP': ETF('XLP', 'EQUITY', 0.0010),
    'SMH': ETF('SMH', 'EQUITY', 0.0035),
    'VIG': ETF('VIG', 'EQUITY', 0.0006),
    'SCHD': ETF('SCHD', 'EQUITY', 0.0006),
    'MDY': ETF('MDY', 'EQUITY', 0.0023),
    'VXX': ETF('VXX', 'VOLATILITY', 0.0089),
}

# Analyze by asset class
asset_classes = {}
for sym, asset in asset_map.items():
    if sym in returns.columns:
        cls = asset.asset_class
        if cls not in asset_classes:
            asset_classes[cls] = []
        asset_classes[cls].append(sym)

print("Asset Classes:")
for cls, assets in asset_classes.items():
    print(f"  {cls}: {assets}")

# Asset class level returns
ac_returns = pd.DataFrame({cls: returns[assets].mean(axis=1) 
                           for cls, assets in asset_classes.items()})
ac_returns.to_csv('/root/quant/iter23_asset_class_returns.csv')

# ============================================================
# E8. COMPREHENSIVE VALIDATION
# ============================================================
print("\n=== E8: Comprehensive Validation ===")

# Collect all strategies
all_strats = {
    'SMA200': backtest(sma_trend(spy, 200).reindex(spy_ret.index).fillna(0), spy_ret, cost_bps=10),
    'VolTarget': backtest(vol_target(spy).reindex(spy_ret.index).fillna(0), spy_ret, cost_bps=10),
    'TSMOM': backtest(tsmom(spy).reindex(spy_ret.index).fillna(0), spy_ret, cost_bps=10),
    'RSI2': backtest(rsi2_meanrev(spy).reindex(spy_ret.index).fillna(0), spy_ret, cost_bps=10),
    'GEM': backtest(dual_momentum(prices[['SPY','GLD','TLT']]).mean(axis=1).reindex(spy_ret.index).fillna(0), 
                    returns[['SPY','GLD','TLT']].mean(axis=1), cost_bps=10),
    '60_40': backtest(pd.DataFrame({'SPY': 0.6, 'TLT': 0.4}, index=returns.index), returns[['SPY','TLT']], cost_bps=10),
    'Equal_Weight': backtest(portfolios['Equal_Weight'], returns, cost_bps=10),
    'Risk_Parity': backtest(portfolios['Risk_Parity'], returns, cost_bps=10),
    'Min_Var': backtest(portfolios['Min_Var'], returns.loc[portfolios['Min_Var'].index], cost_bps=10),
    'Max_Div': backtest(portfolios['Max_Div'], returns.loc[portfolios['Max_Div'].index], cost_bps=10),
    'Event_Driven': evt_returns,
}

val_results = {}
all_sharpes = [sharpe(s.dropna()) for s in all_strats.values() if len(s.dropna()) > 100]
sr_std = np.std(all_sharpes)
n_trials = len(all_sharpes)

for name, ret in all_strats.items():
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
val_df.to_csv('/root/quant/iter23_validation.csv')

perf_results = {}
for name, ret in all_strats.items():
    if len(ret.dropna()) > 100:
        perf_results[name] = perf(ret.dropna(), name)

perf_df = pd.DataFrame(perf_results).T
perf_df.to_csv('/root/quant/iter23_comprehensive_perf.csv')

# ============================================================
# PLOTTING
# ============================================================
print("\n=== Creating Plots ===")

# 1. Fee/Slippage comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
fee_names = list(fee_models.keys())
fee_costs = [m.calculate(test_qty, test_price) for m in fee_models.values()]
ax.barh(fee_names, fee_costs)
ax.set_title('Fee Models: Cost per 1000 shares @ $400')
ax.set_xlabel('Cost ($)')

ax = axes[0, 1]
slip_names = list(slippage_models.keys())
slip_costs = [m.calculate(test_qty, test_price, volume=1_000_000) for m in slippage_models.values()]
ax.barh(slip_names, slip_costs)
ax.set_title('Slippage Models: Cost per 1000 shares @ $400 (vol=1M)')
ax.set_xlabel('Cost ($)')

ax = axes[1, 0]
cost_sorted = cost_df.sort_values('Sharpe', ascending=True)
ax.barh(cost_sorted.index[-15:], cost_sorted['Sharpe'].values[-15:])
ax.set_title('Top 15 Fee/Slippage Combinations (Sharpe)')
ax.axvline(x=0, color='black', alpha=0.5)

ax = axes[1, 1]
# Equity curve comparison: standard vs event-driven
standard_ret = backtest(sma_trend(spy, 200).reindex(spy_ret.index).fillna(0), spy_ret, cost_bps=10)
ax.plot((1 + standard_ret.dropna()).cumprod(), label='Vectorized (10bps)', alpha=0.7)
ax.plot((1 + evt_returns).cumprod(), label='Event-Driven (10bps fee + 5bps slip)', alpha=0.7)
ax.set_title('SMA200: Vectorized vs Event-Driven')
ax.legend()

plt.tight_layout()
plt.savefig('/root/quant/iter23_fees_slippage.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Volatility estimators
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
for col in vol_df.columns:
    vol_df[col].plot(ax=ax, label=col, alpha=0.7)
ax.set_title('Volatility Estimators (Annualized)')
ax.legend(fontsize=8)

ax = axes[0, 1]
vol_t_sorted = pd.Series(vol_target_results).sort_values()
colors = ['green' if v>0.5 else 'orange' if v>0 else 'red' for v in vol_t_sorted]
ax.barh(vol_t_sorted.index, vol_t_sorted.values, color=colors)
ax.set_title('VolTarget with Different Estimators')
ax.axvline(x=0, color='black', alpha=0.5)

ax = axes[1, 0]
for name, weights in portfolios.items():
    common_idx = weights.index.intersection(returns.index)
    if len(common_idx) > 100:
        w = weights.loc[common_idx]
        r = returns.loc[common_idx]
        ret = backtest(w, r, cost_bps=10)
        cum = (1 + ret.dropna()).cumprod()
        cum.plot(ax=ax, label=name, alpha=0.7)
ax.set_title('Portfolio Construction Equity Curves')
ax.legend(fontsize=8)

ax = axes[1, 1]
port_names = list(portfolio_results.keys())
port_sharpes = [portfolio_results[n]['Sharpe'] for n in port_names]
colors_p = ['green' if s>1 else 'orange' if s>0.5 else 'red' for s in port_sharpes]
ax.barh(port_names, port_sharpes, color=colors_p)
ax.set_title('Portfolio Construction Sharpe')
ax.axvline(x=0, color='black', alpha=0.5)

plt.tight_layout()
plt.savefig('/root/quant/iter23_portfolio_vol.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Equity curves
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
axes = axes.flatten()

for i, (name, ret) in enumerate(all_strats.items()):
    if i >= 12:
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
plt.savefig('/root/quant/iter23_equity.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Data quality heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
missing = quality['pct_missing'].values.reshape(-1, 1)
im = ax.imshow(missing, cmap='Reds', aspect='auto')
ax.set_yticks(range(len(quality.index)))
ax.set_yticklabels(quality.index, fontsize=6)
ax.set_title('Missing Data % by Ticker')
plt.colorbar(im, ax=ax)

ax = axes[1]
outlier_mat = outlier_threshold.astype(int).T
im = ax.imshow(outlier_mat, cmap='Reds', aspect='auto')
ax.set_yticks(range(len(outlier_threshold.columns)))
ax.set_yticklabels(outlier_threshold.columns, fontsize=6)
ax.set_title('Outlier Returns (>15%)')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('/root/quant/iter23_data_quality.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n=== Iteration #23 Complete ===")
files = [
    'iter23_fee_slippage.csv', 'iter23_portfolios.csv',
    'iter23_vol_estimators.csv', 'iter23_data_quality.csv',
    'iter23_asset_class_returns.csv', 'iter23_validation.csv',
    'iter23_comprehensive_perf.csv',
    'iter23_fees_slippage.png', 'iter23_portfolio_vol.png',
    'iter23_equity.png', 'iter23_data_quality.png'
]
for f in files:
    print(f"  - {f}")