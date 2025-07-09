"""
Trading Strategy Templates for FPGA Implementation
Hardware-accelerated trading strategies with sub-microsecond execution

Includes:
- Arbitrage strategies
- Market making
- TWAP/VWAP execution
- Momentum strategies
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from fpga_core import FPGACore, OrderType, MarketData

class TradingStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.position_limits = {}
        self.last_execution = 0
        
    @abstractmethod
    def execute(self, fpga: FPGACore):
        """Execute strategy logic"""
        pass
        
    def set_position_limit(self, symbol: str, limit: float):
        """Set position limit for symbol"""
        self.position_limits[symbol] = limit
        
    def get_position_limit(self, symbol: str) -> float:
        """Get position limit for symbol"""
        return self.position_limits.get(symbol, 10000.0)

class ArbitrageStrategy(TradingStrategy):
    """
    Cross-exchange arbitrage strategy
    Exploits price differences between exchanges
    """
    
    def __init__(self, symbols: List[str], min_profit_bps: int = 10):
        super().__init__("Arbitrage")
        self.symbols = symbols
        self.min_profit_bps = min_profit_bps
        self.exchange_prices = {}
        self.last_arb_time = {}
        
    def execute(self, fpga: FPGACore):
        """Execute arbitrage strategy"""
        current_time = int(time.time() * 1000000)
        
        for symbol in self.symbols:
            # Skip if recently executed
            if symbol in self.last_arb_time:
                if current_time - self.last_arb_time[symbol] < 1000:  # 1ms cooldown
                    continue
                    
            # Get current prices
            book = fpga.get_order_book(symbol)
            if not book.get('bids') or not book.get('asks'):
                continue
                
            best_bid = book['bids'][0][0] if book['bids'] else 0
            best_ask = book['asks'][0][0] if book['asks'] else 0
            
            if best_bid <= 0 or best_ask <= 0:
                continue
                
            # Calculate spread
            spread_bps = (best_ask - best_bid) / best_bid * 10000
            
            # Check if arbitrage opportunity exists
            if spread_bps > self.min_profit_bps:
                # Calculate optimal trade size
                trade_size = min(1000, self.get_position_limit(symbol))
                
                # Check position limits
                current_position = fpga.get_position(symbol)
                if abs(current_position) + trade_size > self.get_position_limit(symbol):
                    continue
                    
                # Execute arbitrage
                if current_position <= 0:  # Buy if short or flat
                    fpga.submit_order(symbol, OrderType.BUY, best_ask, trade_size)
                else:  # Sell if long
                    fpga.submit_order(symbol, OrderType.SELL, best_bid, trade_size)
                    
                self.last_arb_time[symbol] = current_time
                
                print(f"Arbitrage: {symbol} spread={spread_bps:.1f}bps size={trade_size}")

class MarketMakingStrategy(TradingStrategy):
    """
    Market making strategy
    Provides liquidity by maintaining bid/ask quotes
    """
    
    def __init__(self, symbols: List[str], spread_bps: int = 5, max_position: float = 5000):
        super().__init__("MarketMaking")
        self.symbols = symbols
        self.spread_bps = spread_bps
        self.max_position = max_position
        self.active_orders = {}
        
    def execute(self, fpga: FPGACore):
        """Execute market making strategy"""
        for symbol in self.symbols:
            # Get current market state
            book = fpga.get_order_book(symbol)
            if not book.get('bids') or not book.get('asks'):
                continue
                
            mid_price = (book['bids'][0][0] + book['asks'][0][0]) / 2
            current_position = fpga.get_position(symbol)
            
            # Calculate quote prices
            spread = mid_price * self.spread_bps / 10000
            bid_price = mid_price - spread / 2
            ask_price = mid_price + spread / 2
            
            # Adjust for inventory
            skew = current_position / self.max_position * spread / 4
            bid_price -= skew
            ask_price -= skew
            
            # Calculate quote sizes
            base_size = 1000
            bid_size = base_size * max(0.1, 1 - current_position / self.max_position)
            ask_size = base_size * max(0.1, 1 + current_position / self.max_position)
            
            # Submit quotes if within position limits
            if abs(current_position - bid_size) <= self.max_position:
                fpga.submit_order(symbol, OrderType.BUY, bid_price, bid_size)
                
            if abs(current_position + ask_size) <= self.max_position:
                fpga.submit_order(symbol, OrderType.SELL, ask_price, ask_size)

class TWAPStrategy(TradingStrategy):
    """
    Time-Weighted Average Price (TWAP) strategy
    Executes large orders over time to minimize market impact
    """
    
    def __init__(self, symbol: str, target_quantity: float, duration_ms: int):
        super().__init__("TWAP")
        self.symbol = symbol
        self.target_quantity = target_quantity
        self.duration_ms = duration_ms
        self.start_time = None
        self.executed_quantity = 0
        self.slice_size = abs(target_quantity) / (duration_ms / 1000)  # shares per second
        
    def execute(self, fpga: FPGACore):
        """Execute TWAP strategy"""
        current_time = int(time.time() * 1000)
        
        if self.start_time is None:
            self.start_time = current_time
            
        # Check if strategy is complete
        elapsed_ms = current_time - self.start_time
        if elapsed_ms >= self.duration_ms:
            return
            
        # Calculate how much should be executed by now
        progress = elapsed_ms / self.duration_ms
        target_executed = self.target_quantity * progress
        remaining = target_executed - self.executed_quantity
        
        if abs(remaining) < 1:  # Less than 1 share remaining
            return
            
        # Get current market price
        book = fpga.get_order_book(self.symbol)
        if not book.get('bids') or not book.get('asks'):
            return
            
        # Determine order type and price
        if remaining > 0:  # Need to buy
            order_type = OrderType.BUY
            price = book['asks'][0][0]  # Take the offer
        else:  # Need to sell
            order_type = OrderType.SELL
            price = book['bids'][0][0]  # Hit the bid
            
        # Submit order
        order_size = min(abs(remaining), 100)  # Max 100 shares per slice
        fpga.submit_order(self.symbol, order_type, price, order_size)
        
        self.executed_quantity += order_size if remaining > 0 else -order_size
        
        print(f"TWAP: {self.symbol} executed {self.executed_quantity}/{self.target_quantity} "
              f"({progress*100:.1f}% complete)")

class VWAPStrategy(TradingStrategy):
    """
    Volume-Weighted Average Price (VWAP) strategy
    Executes orders in proportion to historical volume patterns
    """
    
    def __init__(self, symbol: str, target_quantity: float, volume_profile: List[float]):
        super().__init__("VWAP")
        self.symbol = symbol
        self.target_quantity = target_quantity
        self.volume_profile = volume_profile  # Historical volume by time bucket
        self.start_time = None
        self.executed_quantity = 0
        self.current_bucket = 0
        
    def execute(self, fpga: FPGACore):
        """Execute VWAP strategy"""
        current_time = int(time.time() * 1000)
        
        if self.start_time is None:
            self.start_time = current_time
            
        # Determine current time bucket
        elapsed_ms = current_time - self.start_time
        bucket_duration_ms = 60000  # 1 minute buckets
        self.current_bucket = min(elapsed_ms // bucket_duration_ms, len(self.volume_profile) - 1)
        
        # Calculate target execution for current bucket
        total_volume = sum(self.volume_profile)
        bucket_weight = self.volume_profile[self.current_bucket] / total_volume
        target_for_bucket = self.target_quantity * bucket_weight
        
        # Calculate participation rate (percentage of market volume to capture)
        participation_rate = 0.1  # 10% of market volume
        
        # Get current market data
        book = fpga.get_order_book(self.symbol)
        if not book.get('bids') or not book.get('asks'):
            return
            
        # Estimate current market volume (simplified)
        estimated_volume = 1000  # Would be calculated from recent trades
        max_order_size = estimated_volume * participation_rate
        
        # Calculate order size
        remaining = target_for_bucket - (self.executed_quantity * bucket_weight)
        order_size = min(abs(remaining), max_order_size, 100)
        
        if order_size < 1:
            return
            
        # Submit order
        if remaining > 0:
            order_type = OrderType.BUY
            price = book['asks'][0][0]
        else:
            order_type = OrderType.SELL
            price = book['bids'][0][0]
            
        fpga.submit_order(self.symbol, order_type, price, order_size)
        self.executed_quantity += order_size if remaining > 0 else -order_size
        
        print(f"VWAP: {self.symbol} bucket {self.current_bucket} executed {order_size} shares")

class MomentumStrategy(TradingStrategy):
    """
    Momentum strategy
    Trades based on price momentum and volume
    """
    
    def __init__(self, symbols: List[str], lookback_periods: int = 10, momentum_threshold: float = 0.001):
        super().__init__("Momentum")
        self.symbols = symbols
        self.lookback_periods = lookback_periods
        self.momentum_threshold = momentum_threshold
        self.price_history = {symbol: [] for symbol in symbols}
        self.volume_history = {symbol: [] for symbol in symbols}
        
    def execute(self, fpga: FPGACore):
        """Execute momentum strategy"""
        for symbol in self.symbols:
            # Get current market data
            book = fpga.get_order_book(symbol)
            if not book.get('bids') or not book.get('asks'):
                continue
                
            current_price = (book['bids'][0][0] + book['asks'][0][0]) / 2
            current_volume = book['bids'][0][1] + book['asks'][0][1]
            
            # Update price history
            self.price_history[symbol].append(current_price)
            self.volume_history[symbol].append(current_volume)
            
            # Maintain lookback window
            if len(self.price_history[symbol]) > self.lookback_periods:
                self.price_history[symbol].pop(0)
                self.volume_history[symbol].pop(0)
                
            # Need enough history to calculate momentum
            if len(self.price_history[symbol]) < self.lookback_periods:
                continue
                
            # Calculate momentum
            prices = np.array(self.price_history[symbol])
            volumes = np.array(self.volume_history[symbol])
            
            # Price momentum
            price_momentum = (prices[-1] - prices[0]) / prices[0]
            
            # Volume momentum
            avg_volume = np.mean(volumes[:-1])
            volume_momentum = (volumes[-1] - avg_volume) / avg_volume if avg_volume > 0 else 0
            
            # Combined momentum signal
            momentum_signal = price_momentum * (1 + volume_momentum)
            
            # Generate trades based on momentum
            if abs(momentum_signal) > self.momentum_threshold:
                trade_size = min(1000, self.get_position_limit(symbol))
                current_position = fpga.get_position(symbol)
                
                if momentum_signal > 0 and current_position < self.get_position_limit(symbol):
                    # Positive momentum - buy
                    fpga.submit_order(symbol, OrderType.BUY, book['asks'][0][0], trade_size)
                    print(f"Momentum BUY: {symbol} signal={momentum_signal:.4f}")
                    
                elif momentum_signal < 0 and current_position > -self.get_position_limit(symbol):
                    # Negative momentum - sell
                    fpga.submit_order(symbol, OrderType.SELL, book['bids'][0][0], trade_size)
                    print(f"Momentum SELL: {symbol} signal={momentum_signal:.4f}")

class PairsTradingStrategy(TradingStrategy):
    """
    Pairs trading strategy
    Trades based on relative price movements between correlated assets
    """
    
    def __init__(self, symbol1: str, symbol2: str, hedge_ratio: float = 1.0, 
                 entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        super().__init__("PairsTrading")
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.hedge_ratio = hedge_ratio
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.spread_history = []
        self.position_state = 0  # 0: no position, 1: long spread, -1: short spread
        
    def execute(self, fpga: FPGACore):
        """Execute pairs trading strategy"""
        # Get current prices
        book1 = fpga.get_order_book(self.symbol1)
        book2 = fpga.get_order_book(self.symbol2)
        
        if not (book1.get('bids') and book1.get('asks') and 
                book2.get('bids') and book2.get('asks')):
            return
            
        price1 = (book1['bids'][0][0] + book1['asks'][0][0]) / 2
        price2 = (book2['bids'][0][0] + book2['asks'][0][0]) / 2
        
        # Calculate spread
        spread = price1 - self.hedge_ratio * price2
        self.spread_history.append(spread)
        
        # Maintain history window
        if len(self.spread_history) > 100:
            self.spread_history.pop(0)
            
        if len(self.spread_history) < 20:
            return
            
        # Calculate z-score
        mean_spread = np.mean(self.spread_history)
        std_spread = np.std(self.spread_history)
        if std_spread == 0:
            return
            
        z_score = (spread - mean_spread) / std_spread
        
        # Trading logic
        trade_size = 1000
        
        if self.position_state == 0:  # No position
            if z_score > self.entry_threshold:
                # Spread is high - short spread (sell symbol1, buy symbol2)
                fpga.submit_order(self.symbol1, OrderType.SELL, book1['bids'][0][0], trade_size)
                fpga.submit_order(self.symbol2, OrderType.BUY, book2['asks'][0][0], 
                                trade_size * self.hedge_ratio)
                self.position_state = -1
                print(f"Pairs: SHORT spread {self.symbol1}/{self.symbol2} z={z_score:.2f}")
                
            elif z_score < -self.entry_threshold:
                # Spread is low - long spread (buy symbol1, sell symbol2)
                fpga.submit_order(self.symbol1, OrderType.BUY, book1['asks'][0][0], trade_size)
                fpga.submit_order(self.symbol2, OrderType.SELL, book2['bids'][0][0], 
                                trade_size * self.hedge_ratio)
                self.position_state = 1
                print(f"Pairs: LONG spread {self.symbol1}/{self.symbol2} z={z_score:.2f}")
                
        else:  # Have position
            if abs(z_score) < self.exit_threshold:
                # Close position
                if self.position_state == 1:  # Long spread
                    fpga.submit_order(self.symbol1, OrderType.SELL, book1['bids'][0][0], trade_size)
                    fpga.submit_order(self.symbol2, OrderType.BUY, book2['asks'][0][0], 
                                    trade_size * self.hedge_ratio)
                else:  # Short spread
                    fpga.submit_order(self.symbol1, OrderType.BUY, book1['asks'][0][0], trade_size)
                    fpga.submit_order(self.symbol2, OrderType.SELL, book2['bids'][0][0], 
                                    trade_size * self.hedge_ratio)
                    
                self.position_state = 0
                print(f"Pairs: CLOSE position {self.symbol1}/{self.symbol2} z={z_score:.2f}")

# Strategy factory
def create_strategy(strategy_type: str, **kwargs) -> TradingStrategy:
    """Create strategy instance"""
    if strategy_type == "arbitrage":
        return ArbitrageStrategy(**kwargs)
    elif strategy_type == "market_making":
        return MarketMakingStrategy(**kwargs)
    elif strategy_type == "twap":
        return TWAPStrategy(**kwargs)
    elif strategy_type == "vwap":
        return VWAPStrategy(**kwargs)
    elif strategy_type == "momentum":
        return MomentumStrategy(**kwargs)
    elif strategy_type == "pairs":
        return PairsTradingStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
