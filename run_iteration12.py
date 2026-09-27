"""
Iteration #12 — QuantStart: Advanced Trading Infrastructure (Position/Portfolio/PortfolioHandler),
Risk Management Overlays, Position Sizing Rules, Crypto/DeFi Strategies
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
TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'AGG', 'VNQ', 'DBC', 'XLP', 'XLU', 'XLE', 'XLF', 'IEI', 'VIG', 'SCHD', 'MDY', 'XLK', 'XLV', 'VTI', 'VEA', 'VWO', 'GOVT', 'SHY', 'BIL', 'LQD', 'HYG', 'SMH', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'JPM', 'XOM', 'BRK-B', 'UNH']
price_df = pd.DataFrame({t: load(t)['Close'] for t in TICKERS}).dropna()
spy = price_df['SPY']
etf_tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EFA', 'EEM', 'IEF', 'AGG', 'VNQ', 'DBC', 'XLP', 'XLU', 'XLE']

# ============================================================================
# E1. ADVANCED TRADING INFRASTRUCTURE: POSITION, PORTFOLIO, PORTFOLIOHANDLER
# ============================================================================
print("="*60)
print("E1: Advanced Trading Infrastructure (QuantStart ATI Series)")
print("="*60)

class Position:
    """QuantStart Position Class - tracks a single position."""
    def __init__(self, ticker, quantity=0, avg_price=0.0):
        self.ticker = ticker
        self.quantity = quantity
        self.avg_price = avg_price
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
    
    def update(self, price):
        """Update unrealized PnL."""
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (price - self.avg_price)
    
    def transact(self, quantity, price, commission=0.0):
        """Execute a transaction, update avg_price and realized PnL."""
        if self.quantity == 0:
            # Opening new position
            self.avg_price = price
            self.quantity = quantity
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Adding to position - weighted average price
            total_cost = self.quantity * self.avg_price + quantity * price
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity
        else:
            # Reducing/closing position - realize PnL
            close_qty = min(abs(self.quantity), abs(quantity))
            if self.quantity > 0:
                self.realized_pnl += close_qty * (price - self.avg_price)
            else:
                self.realized_pnl += close_qty * (self.avg_price - price)
            self.quantity += quantity
            if self.quantity == 0:
                self.avg_price = 0.0
        return commission
    
    def market_value(self, price):
        return self.quantity * price
    
    def total_pnl(self, price):
        self.update(price)
        return self.realized_pnl + self.unrealized_pnl

class Portfolio:
    """QuantStart Portfolio Class - collection of positions + cash."""
    def __init__(self, initial_cash=1_000_000):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions = {}  # ticker -> Position
        self.equity_curve = []
        self.trade_log = []
    
    def get_position(self, ticker):
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker)
        return self.positions[ticker]
    
    def update_market_values(self, prices):
        """Update all positions with current prices."""
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.update(prices[ticker])
    
    def total_equity(self, prices):
        """Calculate total portfolio equity."""
        equity = self.cash
        for ticker, pos in self.positions.items():
            if ticker in prices:
                equity += pos.market_value(prices[ticker])
        return equity
    
    def execute_signal(self, ticker, target_weight, prices, cost_bps=10):
        """Convert target weight to trade and execute."""
        current_equity = self.total_equity(prices)
        current_price = prices[ticker]
        
        target_value = target_weight * current_equity
        target_qty = int(target_value / current_price)
        
        pos = self.get_position(ticker)
        current_qty = pos.quantity
        trade_qty = target_qty - current_qty
        
        if trade_qty != 0:
            commission = abs(trade_qty) * current_price * cost_bps / 10000
            exec_price = current_price * (1 + cost_bps/10000 * np.sign(trade_qty))
            
            pos.transact(trade_qty, exec_price, commission)
            self.cash -= trade_qty * exec_price + commission
            
            self.trade_log.append({
                'ticker': ticker,
                'qty': trade_qty,
                'price': exec_price,
                'commission': commission,
                'timestamp': prices.name if hasattr(prices, 'name') else 'unknown'
            })
    
    def get_weights(self, prices):
        """Get current portfolio weights."""
        equity = self.total_equity(prices)
        weights = {}
        for ticker, pos in self.positions.items():
            if ticker in prices and pos.quantity != 0:
                weights[ticker] = pos.market_value(prices[ticker]) / equity
        return weights

class PortfolioHandler:
    """QuantStart PortfolioHandler - risk management, position sizing, signal processing."""
    def __init__(self, portfolio, risk_params=None):
        self.portfolio = portfolio
        self.risk_params = risk_params or {
            'max_position_weight': 0.20,      # Max 20% per position
            'max_sector_weight': 0.40,        # Max 40% per sector
            'max_portfolio_leverage': 1.0,    # No leverage
            'stop_loss_pct': 0.10,            # 10% stop loss
            'take_profit_pct': 0.20,          # 20% take profit
            'max_drawdown_limit': 0.15,       # 15% max portfolio DD
        }
        self.high_water_mark = portfolio.initial_cash
    
    def apply_risk_overlays(self, target_weights, prices):
        """Apply risk management overlays to target weights."""
        adjusted = target_weights.copy()
        
        # 1. Position size limits
        for ticker, weight in adjusted.items():
            if abs(weight) > self.risk_params['max_position_weight']:
                adjusted[ticker] = np.sign(weight) * self.risk_params['max_position_weight']
        
        # 2. Portfolio leverage limit
        total_leverage = sum(abs(w) for w in adjusted.values())
        if total_leverage > self.risk_params['max_portfolio_leverage']:
            scale = self.risk_params['max_portfolio_leverage'] / total_leverage
            adjusted = {k: v * scale for k, v in adjusted.items()}
        
        # 3. Stop loss / take profit on existing positions
        for ticker, pos in self.portfolio.positions.items():
            if pos.quantity != 0 and ticker in prices:
                current_price = prices[ticker]
                pnl_pct = (current_price - pos.avg_price) / pos.avg_price if pos.avg_price != 0 else 0
                
                if pos.quantity > 0:  # Long position
                    if pnl_pct <= -self.risk_params['stop_loss_pct']:
                        adjusted[ticker] = 0  # Stop loss hit
                    elif pnl_pct >= self.risk_params['take_profit_pct']:
                        adjusted[ticker] = 0  # Take profit hit
                elif pos.quantity < 0:  # Short position
                    if pnl_pct >= self.risk_params['stop_loss_pct']:
                        adjusted[ticker] = 0
                    elif pnl_pct <= -self.risk_params['take_profit_pct']:
                        adjusted[ticker] = 0
        
        # 4. Portfolio drawdown limit
        current_equity = self.portfolio.total_equity(prices)
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
        
        dd = (self.high_water_mark - current_equity) / self.high_water_mark
        if dd >= self.risk_params['max_drawdown_limit']:
            # Reduce all positions proportionally
            scale = max(0, 1 - (dd - self.risk_params['max_drawdown_limit']) / 0.05)
            adjusted = {k: v * scale for k, v in adjusted.items()}
        
        return adjusted
    
    def process_signals(self, signals, prices):
        """Main entry point: apply risk overlays then execute."""
        safe_signals = self.apply_risk_overlays(signals, prices)
        for ticker, weight in safe_signals.items():
            if ticker in prices:
                self.portfolio.execute_signal(ticker, weight, prices)
        return safe_signals

# Test the infrastructure
print("Testing Position/Portfolio/PortfolioHandler...")

# Simple SMA200 signal generator
def sma_signal(price_history, current_prices):
    if len(price_history) < 200:
        return {t: 0 for t in price_history.columns}
    sma = price_history.rolling(200).mean().iloc[-1]
    current = price_history.iloc[-1]
    weights = (current > sma).astype(float)
    # Equal weight among active positions
    n_active = weights.sum()
    if n_active > 0:
        weights = weights / n_active
    return weights.to_dict()

# Run backtest with infrastructure
portfolio = Portfolio(1_000_000)
handler = PortfolioHandler(portfolio)

dates = price_df.index
equity_curve = []

for i, date in enumerate(dates):
    current_prices = price_df.loc[date]
    current_prices.name = date
    
    # Get signals from strategy
    price_history = price_df.iloc[:i+1]
    signals = sma_signal(price_history, current_prices)
    
    # Process through PortfolioHandler (risk management)
    handler.process_signals(signals, current_prices)
    
    # Record equity
    equity = portfolio.total_equity(current_prices)
    equity_curve.append(equity)

equity_series = pd.Series(equity_curve, index=dates)
inf_returns = equity_series.pct_change().fillna(0)
inf_perf = performance(inf_returns)
print(f"Infrastructure SMA200: Sharpe={inf_perf['Sharpe']:.3f}, AnnRet%={inf_perf['AnnRet%']:.2f}, MaxDD%={inf_perf['MaxDD%']:.2f}")

# Compare with simple vectorized
vec_weights = sma_trend(spy, 200)
vec_bt = backtest(vec_weights, spy.pct_change(), cost_bps=10)
vec_perf = performance(vec_bt)
print(f"Vectorized SMA200: Sharpe={vec_perf['Sharpe']:.3f}, AnnRet%={vec_perf['AnnRet%']:.2f}, MaxDD%={vec_perf['MaxDD%']:.2f}")

infra_results = pd.DataFrame({
    'Architecture': ['Position_Portfolio_Handler', 'Vectorized'],
    'Sharpe': [inf_perf['Sharpe'], vec_perf['Sharpe']],
    'AnnRet%': [inf_perf['AnnRet%'], vec_perf['AnnRet%']],
    'MaxDD%': [inf_perf['MaxDD%'], vec_perf['MaxDD%']],
    'Calmar': [inf_perf['Calmar'], vec_perf['Calmar']]
})
infra_results.to_csv('/root/quant/iter12_infrastructure.csv', index=False)
print("Saved iter12_infrastructure.csv")

# ============================================================================
# E2. POSITION SIZING RULES COMPARISON
# ============================================================================
print("\n" + "="*60)
print("E2: Position Sizing Rules Comparison")
print("="*60)

def run_with_sizing(sizing_func, name, price_df, signal_func, cost_bps=10):
    """Run backtest with custom position sizing."""
    portfolio = Portfolio(1_000_000)
    dates = price_df.index
    equity_curve = []
    
    for i, date in enumerate(dates):
        current_prices = price_df.loc[date]
        current_prices.name = date
        
        price_history = price_df.iloc[:i+1]
        base_signals = signal_func(price_history, current_prices)
        
        # Apply position sizing
        sized_signals = sizing_func(base_signals, portfolio, current_prices)
        
        # Execute
        for ticker, weight in sized_signals.items():
            if ticker in current_prices:
                portfolio.execute_signal(ticker, weight, current_prices, cost_bps)
        
        equity_curve.append(portfolio.total_equity(current_prices))
    
    equity_series = pd.Series(equity_curve, index=dates)
    returns = equity_series.pct_change().fillna(0)
    return performance(returns)

# Signal function: equal weight momentum
def mom_signal(price_history, current_prices):
    if len(price_history) < 252:
        return {t: 0 for t in price_history.columns}
    mom = price_history.iloc[-1] / price_history.iloc[-252] - 1
    # Top 3 momentum
    top3 = mom.nlargest(3).index
    weights = {t: 1/3 if t in top3 else 0 for t in price_history.columns}
    return weights

# Sizing rules
def fixed_fractional(signals, portfolio, prices, fraction=0.1):
    """Fixed fractional: risk fixed % of equity per trade."""
    equity = portfolio.total_equity(prices)
    sized = {}
    for t, w in signals.items():
        if w != 0:
            sized[t] = np.sign(w) * fraction
    return sized

def volatility_targeted(signals, portfolio, prices, target_vol=0.15):
    """Volatility targeting: scale position to target portfolio vol."""
    # Estimate current portfolio vol from recent returns
    equity = portfolio.total_equity(prices)
    if len(portfolio.equity_curve) > 20:
        recent_eq = pd.Series(portfolio.equity_curve[-20:])
        recent_ret = recent_eq.pct_change().dropna()
        current_vol = recent_ret.std() * np.sqrt(252)
        if current_vol > 0:
            scale = target_vol / current_vol
            scale = min(scale, 2.0)  # Cap at 2x
        else:
            scale = 1.0
    else:
        scale = 1.0
    
    sized = {t: w * scale for t, w in signals.items()}
    return sized

def kelly_sizing(signals, portfolio, prices, lookback=252):
    """Kelly criterion: f = (p*b - q) / b where p=win%, b=avg_win/avg_loss."""
    # Estimate from recent strategy returns
    if len(portfolio.equity_curve) > lookback:
        recent_eq = pd.Series(portfolio.equity_curve[-lookback:])
        recent_ret = recent_eq.pct_change().dropna()
        wins = recent_ret[recent_ret > 0]
        losses = recent_ret[recent_ret < 0]
        if len(wins) > 10 and len(losses) > 10:
            p = len(wins) / len(recent_ret)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            b = avg_win / avg_loss if avg_loss > 0 else 1
            kelly_f = (p * b - (1-p)) / b if b > 0 else 0
            kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25%
        else:
            kelly_f = 0.1
    else:
        kelly_f = 0.1
    
    sized = {t: np.sign(w) * kelly_f for t, w in signals.items() if w != 0}
    return sized

def risk_parity_sizing(signals, portfolio, prices, lookback=60):
    """Risk parity: equal risk contribution."""
    # Get covariance of active assets
    active = [t for t, w in signals.items() if w != 0]
    if len(active) < 2:
        return signals
    
    rets = price_df[active].pct_change().tail(lookback).dropna()
    cov = rets.cov() * 252
    
    # Inverse volatility weights
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1 / vols
    weights = inv_vol / inv_vol.sum()
    
    sized = {t: weights[i] * np.sign(signals[t]) for i, t in enumerate(active)}
    return sized

# Test all sizing methods
sizing_methods = [
    ('Fixed_Fractional_10%', lambda s, p, pr: fixed_fractional(s, p, pr, 0.1)),
    ('Fixed_Fractional_20%', lambda s, p, pr: fixed_fractional(s, p, pr, 0.2)),
    ('Vol_Target_15%', lambda s, p, pr: volatility_targeted(s, p, pr, 0.15)),
    ('Kelly_Capped_25%', lambda s, p, pr: kelly_sizing(s, p, pr)),
    ('Risk_Parity', lambda s, p, pr: risk_parity_sizing(s, p, pr)),
]

sizing_results = []
for name, sizing_func in sizing_methods:
    perf = run_with_sizing(sizing_func, name, price_df[etf_tickers], mom_signal)
    sizing_results.append({
        'Method': name,
        'Sharpe': perf['Sharpe'],
        'AnnRet%': perf['AnnRet%'],
        'MaxDD%': perf['MaxDD%'],
        'Calmar': perf['Calmar']
    })
    print(f"  {name}: Sharpe={perf['Sharpe']:.3f}, AnnRet%={perf['AnnRet%']:.2f}, MaxDD%={perf['MaxDD%']:.2f}")

sizing_df = pd.DataFrame(sizing_results)
sizing_df.to_csv('/root/quant/iter12_position_sizing.csv', index=False)
print("Saved iter12_position_sizing.csv")

# ============================================================================
# E3. CRYPTO/DEFI STRATEGIES (SIMULATED - QUANTSTART ALTERNATIVE ASSETS)
# ============================================================================
print("\n" + "="*60)
print("E3: Crypto/DeFi Strategies (Simulated)")
print("="*60)

# Simulate crypto-like returns (high vol, 24/7, different characteristics)
# Use VXX (volatility) as crypto proxy - high vol, mean-reverting
# And create synthetic crypto assets
np.random.seed(42)
n_days = len(spy)

# Simulate 5 crypto assets with different regimes
crypto_names = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
crypto_prices = pd.DataFrame(index=spy.index)

for name in crypto_names:
    # Crypto: higher vol, some momentum, some mean reversion
    base_ret = np.random.normal(0.0005, 0.03, n_days)  # ~12% ann, 47% vol
    # Add momentum component
    mom = np.random.normal(0, 0.005, n_days)
    # Add crash risk (fat tails)
    crash = np.random.choice([0, -0.15], n_days, p=[0.995, 0.005])
    rets = base_ret + mom + crash
    prices = 100 * np.exp(np.cumsum(rets))
    crypto_prices[name] = prices

# Crypto strategies
# 1. Trend following (SMA 50/200)
def crypto_sma_trend(prices, fast=50, slow=200):
    return (prices.rolling(fast).mean() > prices.rolling(slow).mean()).astype(float)

# 2. Mean reversion (RSI)
def crypto_rsi(prices, period=14, buy=30, sell=70):
    delta = prices.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/period).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    pos = pd.Series(0.0, index=prices.index)
    pos[rsi < buy] = 1
    pos[rsi > sell] = 0
    return pos.ffill().fillna(0)

# 3. Momentum (12-1 month)
def crypto_momentum(prices, lookback=252, skip=21):
    mom = prices.shift(skip) / prices.shift(lookback + skip) - 1
    return (mom > 0).astype(float)

# Test on each crypto
crypto_results = []
for name in crypto_names:
    px = crypto_prices[name]
    
    sma_sig = crypto_sma_trend(px)
    sma_bt = backtest(sma_sig, px.pct_change(), cost_bps=20)  # Higher cost for crypto
    sma_perf = performance(sma_bt)
    
    rsi_sig = crypto_rsi(px)
    rsi_bt = backtest(rsi_sig, px.pct_change(), cost_bps=20)
    rsi_perf = performance(rsi_bt)
    
    mom_sig = crypto_momentum(px)
    mom_bt = backtest(mom_sig, px.pct_change(), cost_bps=20)
    mom_perf = performance(mom_bt)
    
    bh_ret = px.pct_change()
    bh_perf = performance(bh_ret)
    
    crypto_results.append({
        'Asset': name,
        'BuyHold_Sharpe': bh_perf['Sharpe'],
        'SMA_Sharpe': sma_perf['Sharpe'],
        'RSI_Sharpe': rsi_perf['Sharpe'],
        'Momentum_Sharpe': mom_perf['Sharpe'],
        'BuyHold_AnnRet%': bh_perf['AnnRet%'],
        'SMA_AnnRet%': sma_perf['AnnRet%'],
        'RSI_AnnRet%': rsi_perf['AnnRet%'],
        'Momentum_AnnRet%': mom_perf['AnnRet%']
    })
    print(f"  {name}: BH={bh_perf['Sharpe']:.3f}, SMA={sma_perf['Sharpe']:.3f}, RSI={rsi_perf['Sharpe']:.3f}, Mom={mom_perf['Sharpe']:.3f}")

crypto_df = pd.DataFrame(crypto_results)
crypto_df.to_csv('/root/quant/iter12_crypto.csv', index=False)
print("Saved iter12_crypto.csv")

# ============================================================================
# E4. DEFI YIELD FARMING SIMULATION
# ============================================================================
print("\n" + "="*60)
print("E4: DeFi Yield Farming Simulation")
print("="*60)

# Simulate DeFi yields: base yield + impermanent loss + smart contract risk
np.random.seed(123)
n_days = len(spy)

# Simulate stablecoin yield (e.g., USDC lending)
stable_yield = 0.05 + np.random.normal(0, 0.01, n_days) / 252  # ~5% APY
# Simulate ETH staking yield
eth_stake_yield = 0.04 + np.random.normal(0, 0.015, n_days) / 252  # ~4% APY
# Simulate LP impermanent loss (correlated with vol)
spy_ret = spy.pct_change()
il_factor = -0.5 * spy_ret.rolling(30).std() * np.sqrt(252) * 0.5  # Simplified IL

# DeFi portfolio: 50% stable yield, 30% ETH stake, 20% LP (with IL)
defi_ret = (0.5 * stable_yield + 
            0.3 * eth_stake_yield + 
            0.2 * (eth_stake_yield + il_factor))

# Add smart contract risk (rare catastrophic loss)
sc_risk = np.random.choice([0, -0.5], n_days, p=[0.999, 0.001])
defi_ret = defi_ret + sc_risk

defi_perf = performance(defi_ret.dropna())
print(f"DeFi Yield Portfolio: Sharpe={defi_perf['Sharpe']:.3f}, AnnRet%={defi_perf['AnnRet%']:.2f}, MaxDD%={defi_perf['MaxDD%']:.2f}")

# Compare with traditional
traditional_ret = spy.pct_change().dropna()
trad_perf = performance(traditional_ret)
print(f"Traditional SPY: Sharpe={trad_perf['Sharpe']:.3f}, AnnRet%={trad_perf['AnnRet%']:.2f}, MaxDD%={trad_perf['MaxDD%']:.2f}")

defi_results = pd.DataFrame({
    'Strategy': ['DeFi_Yield', 'Traditional_SPY'],
    'Sharpe': [defi_perf['Sharpe'], trad_perf['Sharpe']],
    'AnnRet%': [defi_perf['AnnRet%'], trad_perf['AnnRet%']],
    'MaxDD%': [defi_perf['MaxDD%'], trad_perf['MaxDD%']],
    'Calmar': [defi_perf['Calmar'], trad_perf['Calmar']]
})
defi_results.to_csv('/root/quant/iter12_defi.csv', index=False)
print("Saved iter12_defi.csv")

# ============================================================================
# E5. COMPREHENSIVE PERFORMANCE SUMMARY
# ============================================================================
print("\n" + "="*60)
print("ITERATION #12 COMPREHENSIVE PERFORMANCE")
print("="*60)

strategies = {
    'SMA200': lambda px: sma_trend(px, 200),
    'GEM': lambda px: dual_momentum(price_df[['SPY','GLD','TLT','IEF','SHY']], 126).get('SPY', pd.Series(0, index=px.index)),
    'XSecMom': lambda px: xsec_momentum(price_df[etf_tickers], 252, 21, 3).get('SPY', pd.Series(0, index=px.index)),
    'TSMOM': lambda px: tsmom(px, 252, 21),
    'RSI2': lambda px: rsi2_meanrev(px),
    'VolTarget': lambda px: vol_target(px, 0.10, 21, 2.0),
    'MACross': lambda px: ma_crossover(px, 50, 200),
    'Rev5': lambda px: short_term_reversal(price_df[etf_tickers], 5).get('SPY', pd.Series(0, index=px.index)),
    'Infra_SMA200': lambda px: None,  # Special case
}

comp_results = []
for name, func in strategies.items():
    try:
        if name == 'Infra_SMA200':
            pf = inf_perf
        else:
            weights = func(price_df['SPY'] if name not in ['GEM','XSecMom','Rev5'] else price_df)
            if isinstance(weights, pd.DataFrame):
                weights = weights['SPY'] if 'SPY' in weights.columns else weights.iloc[:, 0]
            bt = backtest(weights, spy.pct_change(), cost_bps=10)
            pf = performance(bt)
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
comp_df.to_csv('/root/quant/iter12_comprehensive_perf.csv', index=False)
print(comp_df.to_string(index=False))

# Walk-forward
print("\nWalk-forward validation (SMA200 with risk overlays)...")
wf_results = []
n_folds = 4
fold_size = len(spy) // n_folds
for fold in range(n_folds):
    start = fold * fold_size
    end = (fold + 1) * fold_size if fold < n_folds - 1 else len(spy)
    train = spy.iloc[:start]
    test = spy.iloc[start:end]
    
    if len(train) < 200:
        continue
    
    best_window = 200
    best_sharpe = -np.inf
    for w in [50, 100, 150, 200, 250, 300]:
        sig = sma_trend(train, w)
        bt = backtest(sig, train.pct_change(), cost_bps=10)
        pf = performance(bt)
        if pf['Sharpe'] > best_sharpe:
            best_sharpe = pf['Sharpe']
            best_window = w
    
    sig_test = sma_trend(test, best_window)
    bt_test = backtest(sig_test, test.pct_change(), cost_bps=10)
    pf_test = performance(bt_test)
    wf_results.append({'Fold': fold+1, 'Best_Window': best_window, 'Train_Sharpe': best_sharpe, 'Test_Sharpe': pf_test['Sharpe']})

wf_df = pd.DataFrame(wf_results)
wf_df.to_csv('/root/quant/iter12_comprehensive_walkforward.csv', index=False)
print(wf_df.to_string(index=False))

# Purged K-Fold
print("\nPurged K-Fold validation...")
from sklearn.model_selection import KFold

def purged_kfold_sharpe(returns, n_splits=4, embargo_pct=0.01):
    n = len(returns)
    embargo = int(n * embargo_pct)
    kf = KFold(n_splits=n_splits, shuffle=False)
    sharpes = []
    
    for train_idx, test_idx in kf.split(returns):
        test_start = test_idx[0]
        test_end = test_idx[-1]
        train_idx_purged = train_idx[train_idx < test_start - embargo]
        
        if len(train_idx_purged) > 100 and len(test_idx) > 100:
            train_ret = returns.iloc[train_idx_purged]
            test_ret = returns.iloc[test_idx]
            
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
val_df.to_csv('/root/quant/iter12_comprehensive_validation.csv', index=False)
print("Saved iter12_comprehensive_validation.csv")

# ============================================================================
# PLOTS
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Position sizing comparison
ax = axes[0, 0]
methods = [r['Method'] for r in sizing_results]
sharpes = [r['Sharpe'] for r in sizing_results]
colors = ['green' if s > 0 else 'red' for s in sharpes]
ax.barh(methods, sharpes, color=colors, alpha=0.7)
ax.set_xlabel('Sharpe Ratio')
ax.set_title('Position Sizing Methods Comparison')
ax.grid(True, alpha=0.3)

# Crypto strategy comparison
ax = axes[0, 1]
crypto_strats = ['BuyHold', 'SMA', 'RSI', 'Momentum']
for strat in crypto_strats:
    vals = [r[f'{strat}_Sharpe'] for r in crypto_results]
    ax.plot(crypto_names, vals, 'o-', label=strat, alpha=0.8)
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Crypto Strategies by Asset')
ax.legend()
ax.grid(True, alpha=0.3)

# Infrastructure equity curve
ax = axes[1, 0]
ax.plot(equity_series / 1e6, label='Position/Portfolio/Handler', color='blue')
ax.plot((1 + vec_bt).cumprod() / 1e6, label='Vectorized', color='orange', alpha=0.7)
ax.set_ylabel('Portfolio Value ($M)')
ax.set_title('Advanced Infrastructure vs Vectorized')
ax.legend()
ax.grid(True, alpha=0.3)

# DeFi vs Traditional
ax = axes[1, 1]
defi_eq = (1 + defi_ret.fillna(0)).cumprod()
trad_eq = (1 + trad_ret).cumprod()
ax.plot(defi_eq, label='DeFi Yield', color='purple')
ax.plot(trad_eq, label='SPY', color='green')
ax.set_ylabel('Cumulative Return')
ax.set_title('DeFi vs Traditional')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/quant/iter12_equity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved iter12_equity.png")

print("\n" + "="*60)
print("ITERATION #12 COMPLETE")
print("="*60)
