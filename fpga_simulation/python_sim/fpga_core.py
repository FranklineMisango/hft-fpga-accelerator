"""
FPGA Trading Accelerator Core Simulation
Hardware-accelerated low-latency trading infrastructure simulation

This module simulates FPGA behavior for trading operations including:
- Market data processing
- Order book management
- Strategy execution
- Risk controls
"""

import numpy as np
import asyncio
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

class OrderType(Enum):
    BUY = 1
    SELL = 2
    CANCEL = 3

class OrderStatus(Enum):
    PENDING = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4

@dataclass
class MarketData:
    """Market data tick structure"""
    symbol: str
    timestamp: int
    price: float
    volume: float
    bid: float
    ask: float
    
@dataclass
class Order:
    """Order structure"""
    order_id: int
    symbol: str
    order_type: OrderType
    price: float
    quantity: float
    timestamp: int
    status: OrderStatus = OrderStatus.PENDING

@dataclass
class Trade:
    """Trade execution structure"""
    trade_id: int
    order_id: int
    symbol: str
    price: float
    quantity: float
    timestamp: int
    side: OrderType

class FPGACore:
    """
    Main FPGA simulation core
    Simulates hardware-accelerated trading operations
    """
    
    def __init__(self, clock_freq_mhz=100):
        self.clock_freq_mhz = clock_freq_mhz
        self.clock_period_ns = 1000 / clock_freq_mhz
        
        # Hardware simulation state
        self.cycle_count = 0
        self.running = False
        
        # Market data processing
        self.market_data_buffer = deque(maxlen=10000)
        self.order_book = defaultdict(lambda: {'bids': [], 'asks': []})
        
        # Order management
        self.orders = {}
        self.order_counter = 0
        self.pending_orders = deque()
        
        # Strategy execution
        self.strategies = []
        self.positions = defaultdict(float)
        self.portfolio_value = 1000000.0  # $1M initial
        
        # Risk controls
        self.position_limits = defaultdict(lambda: 100000.0)  # Default $100K per symbol
        self.max_order_size = 10000.0
        
        # Performance metrics
        self.latency_stats = []
        self.throughput_stats = []
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=8)
        
    def start_simulation(self):
        """Start the FPGA simulation"""
        self.running = True
        print(f"Starting FPGA simulation at {self.clock_freq_mhz}MHz")
        
    def stop_simulation(self):
        """Stop the FPGA simulation"""
        self.running = False
        print("FPGA simulation stopped")
        
    def clock_tick(self):
        """Simulate one clock cycle"""
        if not self.running:
            return
            
        self.cycle_count += 1
        
        # Process market data (1 cycle)
        self._process_market_data()
        
        # Process orders (1 cycle)
        self._process_orders()
        
        # Execute strategies (1 cycle)
        self._execute_strategies()
        
        # Update risk controls (1 cycle)
        self._update_risk_controls()
        
    def _process_market_data(self):
        """Process incoming market data (hardware simulation)"""
        if not self.market_data_buffer:
            return
            
        # Simulate hardware parsing - 1 cycle latency
        tick = self.market_data_buffer.popleft()
        
        # Update order book
        self._update_order_book(tick)
        
    def _update_order_book(self, tick: MarketData):
        """Update order book with new tick data"""
        book = self.order_book[tick.symbol]
        
        # Simulate hardware order book update
        # In real FPGA, this would be done in parallel with CAM/TCAM
        if tick.bid > 0:
            book['bids'] = sorted([(tick.bid, tick.volume)], reverse=True)
        if tick.ask > 0:
            book['asks'] = sorted([(tick.ask, tick.volume)])
            
    def _process_orders(self):
        """Process pending orders (hardware simulation)"""
        if not self.pending_orders:
            return
            
        # Simulate order processing - 1 cycle per order
        order = self.pending_orders.popleft()
        
        # Risk check (hardware parallel)
        if not self._risk_check(order):
            order.status = OrderStatus.REJECTED
            return
            
        # Try to fill order
        if self._try_fill_order(order):
            order.status = OrderStatus.FILLED
        else:
            # Keep in pending queue
            self.pending_orders.append(order)
            
    def _risk_check(self, order: Order) -> bool:
        """Hardware risk control check"""
        # Position limit check
        current_position = self.positions[order.symbol]
        if order.order_type == OrderType.BUY:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
            
        if abs(new_position * order.price) > self.position_limits[order.symbol]:
            return False
            
        # Order size check
        if order.quantity > self.max_order_size:
            return False
            
        return True
        
    def _try_fill_order(self, order: Order) -> bool:
        """Try to fill order against order book"""
        book = self.order_book[order.symbol]
        
        if order.order_type == OrderType.BUY and book['asks']:
            best_ask = book['asks'][0]
            if order.price >= best_ask[0]:
                # Fill order
                self._execute_trade(order, best_ask[0])
                return True
                
        elif order.order_type == OrderType.SELL and book['bids']:
            best_bid = book['bids'][0]
            if order.price <= best_bid[0]:
                # Fill order
                self._execute_trade(order, best_bid[0])
                return True
                
        return False
        
    def _execute_trade(self, order: Order, fill_price: float):
        """Execute a trade"""
        trade = Trade(
            trade_id=len(self.latency_stats),
            order_id=order.order_id,
            symbol=order.symbol,
            price=fill_price,
            quantity=order.quantity,
            timestamp=int(time.time() * 1000000),  # microseconds
            side=order.order_type
        )
        
        # Update positions
        if order.order_type == OrderType.BUY:
            self.positions[order.symbol] += order.quantity
        else:
            self.positions[order.symbol] -= order.quantity
            
        # Calculate latency (simulated)
        latency_ns = self.cycle_count * self.clock_period_ns
        self.latency_stats.append(latency_ns)
        
        print(f"Trade executed: {order.symbol} {order.quantity}@{fill_price} ({latency_ns:.1f}ns)")
        
    def _execute_strategies(self):
        """Execute trading strategies"""
        for strategy in self.strategies:
            strategy.execute(self)
            
    def _update_risk_controls(self):
        """Update risk control parameters"""
        # Dynamic risk adjustment based on market conditions
        total_portfolio_value = sum(
            abs(pos * self.get_last_price(symbol)) 
            for symbol, pos in self.positions.items()
        )
        
        # Adjust position limits based on portfolio utilization
        utilization = total_portfolio_value / self.portfolio_value
        if utilization > 0.8:  # 80% utilization
            # Reduce position limits
            for symbol in self.position_limits:
                self.position_limits[symbol] *= 0.95
                
    def get_last_price(self, symbol: str) -> float:
        """Get last price for a symbol"""
        book = self.order_book[symbol]
        if book['bids'] and book['asks']:
            return (book['bids'][0][0] + book['asks'][0][0]) / 2
        return 0.0
        
    def submit_order(self, symbol: str, order_type: OrderType, price: float, quantity: float) -> int:
        """Submit a new order"""
        order_id = self.order_counter
        self.order_counter += 1
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            price=price,
            quantity=quantity,
            timestamp=int(time.time() * 1000000)
        )
        
        self.orders[order_id] = order
        self.pending_orders.append(order)
        
        return order_id
        
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an existing order"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
        
    def feed_market_data(self, market_data: MarketData):
        """Feed market data to FPGA"""
        self.market_data_buffer.append(market_data)
        
    def get_order_book(self, symbol: str) -> Dict:
        """Get current order book for symbol"""
        return dict(self.order_book[symbol])
        
    def get_position(self, symbol: str) -> float:
        """Get current position for symbol"""
        return self.positions[symbol]
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.latency_stats:
            return {}
            
        return {
            'avg_latency_ns': np.mean(self.latency_stats),
            'min_latency_ns': np.min(self.latency_stats),
            'max_latency_ns': np.max(self.latency_stats),
            'p99_latency_ns': np.percentile(self.latency_stats, 99),
            'total_trades': len(self.latency_stats),
            'cycle_count': self.cycle_count
        }
        
    def run_simulation(self, duration_ms: int = 1000):
        """Run simulation for specified duration"""
        self.start_simulation()
        
        cycles_to_run = int(duration_ms * 1000 * self.clock_freq_mhz / 1000000)
        
        for _ in range(cycles_to_run):
            self.clock_tick()
            
        self.stop_simulation()
        
        return self.get_performance_stats()
