"""
Market Data Simulator
Simulates real-time market data feeds for FPGA trading system testing

Supports multiple protocols:
- ITCH (NASDAQ)
- FIX (Financial Information Exchange)
- OUCH (Order handling)
"""

import asyncio
import numpy as np
import time
import random
import struct
from typing import List, Dict, Optional
from dataclasses import dataclass
from fpga_core import MarketData, FPGACore

class MarketDataSimulator:
    """
    Simulates realistic market data feeds
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.prices = {symbol: 100.0 + random.uniform(-50, 50) for symbol in self.symbols}
        self.running = False
        self.subscribers = []
        
        # Market parameters
        self.volatility = 0.02  # 2% volatility
        self.tick_size = 0.01
        self.spread_bps = 5  # 5 basis points spread
        
    def subscribe(self, callback):
        """Subscribe to market data updates"""
        self.subscribers.append(callback)
        
    def unsubscribe(self, callback):
        """Unsubscribe from market data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            
    def start_feed(self):
        """Start market data feed"""
        self.running = True
        print("Market data feed started")
        
    def stop_feed(self):
        """Stop market data feed"""
        self.running = False
        print("Market data feed stopped")
        
    async def generate_tick(self, symbol: str) -> MarketData:
        """Generate a single market data tick"""
        current_price = self.prices[symbol]
        
        # Random walk with volatility
        price_change = np.random.normal(0, self.volatility * current_price / 100)
        new_price = max(0.01, current_price + price_change)
        
        # Round to tick size
        new_price = round(new_price / self.tick_size) * self.tick_size
        self.prices[symbol] = new_price
        
        # Calculate bid/ask spread
        spread = new_price * self.spread_bps / 10000
        bid = new_price - spread / 2
        ask = new_price + spread / 2
        
        # Random volume
        volume = random.uniform(100, 10000)
        
        return MarketData(
            symbol=symbol,
            timestamp=int(time.time() * 1000000),  # microseconds
            price=new_price,
            volume=volume,
            bid=bid,
            ask=ask
        )
        
    async def simulate_feed(self, tick_rate_hz: int = 1000):
        """Simulate market data feed at specified rate"""
        interval = 1.0 / tick_rate_hz
        
        while self.running:
            start_time = time.time()
            
            # Generate ticks for all symbols
            for symbol in self.symbols:
                if random.random() < 0.1:  # 10% chance per symbol per tick
                    tick = await self.generate_tick(symbol)
                    
                    # Send to subscribers
                    for callback in self.subscribers:
                        await callback(tick)
                        
            # Maintain tick rate
            elapsed = time.time() - start_time
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
                
    def simulate_burst(self, symbol: str, num_ticks: int = 100) -> List[MarketData]:
        """Simulate a burst of market data"""
        ticks = []
        for _ in range(num_ticks):
            tick = asyncio.run(self.generate_tick(symbol))
            ticks.append(tick)
            
        return ticks
        
    def simulate_order_book_event(self, symbol: str, event_type: str = 'add') -> MarketData:
        """Simulate specific order book events"""
        base_price = self.prices[symbol]
        
        if event_type == 'add':
            # Add new order
            price = base_price + random.uniform(-0.5, 0.5)
            volume = random.uniform(100, 1000)
        elif event_type == 'cancel':
            # Cancel order
            price = base_price
            volume = 0
        elif event_type == 'modify':
            # Modify order
            price = base_price + random.uniform(-0.1, 0.1)
            volume = random.uniform(50, 500)
        else:
            price = base_price
            volume = random.uniform(100, 1000)
            
        return MarketData(
            symbol=symbol,
            timestamp=int(time.time() * 1000000),
            price=price,
            volume=volume,
            bid=price - 0.01,
            ask=price + 0.01
        )

class ITCHSimulator:
    """
    NASDAQ ITCH Protocol Simulator
    Simulates Level 2 market data messages
    """
    
    def __init__(self):
        self.message_types = {
            'A': 'Add Order',
            'F': 'Add Order (MPID)',
            'E': 'Order Executed',
            'C': 'Order Executed (with Price)',
            'X': 'Order Cancel',
            'D': 'Order Delete',
            'U': 'Order Replace'
        }
        
    def generate_itch_message(self, symbol: str, msg_type: str = 'A') -> bytes:
        """Generate ITCH binary message"""
        timestamp = int(time.time() * 1000000000)  # nanoseconds
        
        if msg_type == 'A':  # Add Order
            # Simplified ITCH Add Order message
            message = struct.pack(
                '>cHIQcI4sIIc',
                b'A',                    # Message Type
                0,                       # Stock Locate
                0,                       # Tracking Number
                timestamp,               # Timestamp
                b'B',                    # Buy/Sell Indicator
                12345,                   # Order Reference Number
                symbol.encode('ascii')[:4].ljust(4, b' '),  # Stock
                1000,                    # Shares
                10000,                   # Price (in 1/10000 dollars)
                b'Y'                     # Attribution
            )
            
        elif msg_type == 'E':  # Order Executed
            message = struct.pack(
                '>cHIQII',
                b'E',                    # Message Type
                0,                       # Stock Locate
                0,                       # Tracking Number
                timestamp,               # Timestamp
                12345,                   # Order Reference Number
                500                      # Executed Shares
            )
            
        else:
            # Default to Add Order
            message = self.generate_itch_message(symbol, 'A')
            
        return message
        
    def parse_itch_message(self, message: bytes) -> Dict:
        """Parse ITCH binary message"""
        msg_type = chr(message[0])
        
        if msg_type == 'A':
            # Parse Add Order message
            data = struct.unpack('>cHIQcI4sIIc', message)
            return {
                'type': 'Add Order',
                'timestamp': data[3],
                'side': data[4].decode('ascii'),
                'order_ref': data[5],
                'symbol': data[6].decode('ascii').strip(),
                'shares': data[7],
                'price': data[8] / 10000.0
            }
            
        elif msg_type == 'E':
            # Parse Order Executed message
            data = struct.unpack('>cHIQII', message)
            return {
                'type': 'Order Executed',
                'timestamp': data[3],
                'order_ref': data[4],
                'executed_shares': data[5]
            }
            
        return {'type': 'Unknown', 'raw': message}

class FIXSimulator:
    """
    FIX Protocol Simulator
    Simulates FIX messages for order handling
    """
    
    def __init__(self):
        self.seq_num = 1
        
    def generate_fix_message(self, msg_type: str, symbol: str, **kwargs) -> str:
        """Generate FIX message string"""
        timestamp = time.strftime('%Y%m%d-%H:%M:%S', time.gmtime())
        
        if msg_type == 'D':  # New Order Single
            fix_msg = (
                f"8=FIX.4.2|9=XXX|35=D|34={self.seq_num}|49=CLIENT|52={timestamp}|"
                f"56=EXCHANGE|11=ORDER123|21=1|40=2|54=1|55={symbol}|"
                f"38={kwargs.get('quantity', 1000)}|44={kwargs.get('price', 100.0)}|"
                f"59=0|10=XXX|"
            )
            
        elif msg_type == '8':  # Execution Report
            fix_msg = (
                f"8=FIX.4.2|9=XXX|35=8|34={self.seq_num}|49=EXCHANGE|52={timestamp}|"
                f"56=CLIENT|6={kwargs.get('avg_price', 100.0)}|11=ORDER123|"
                f"14={kwargs.get('cum_qty', 1000)}|17=EXEC123|20=0|"
                f"37=ORDER123|38={kwargs.get('quantity', 1000)}|39=2|"
                f"54=1|55={symbol}|150=2|151=0|10=XXX|"
            )
            
        else:
            fix_msg = f"8=FIX.4.2|9=XXX|35={msg_type}|34={self.seq_num}|10=XXX|"
            
        self.seq_num += 1
        return fix_msg
        
    def parse_fix_message(self, message: str) -> Dict:
        """Parse FIX message string"""
        fields = {}
        pairs = message.split('|')
        
        for pair in pairs:
            if '=' in pair:
                tag, value = pair.split('=', 1)
                fields[tag] = value
                
        return fields

# Example usage and testing
async def test_market_data_simulation():
    """Test market data simulation"""
    
    # Create FPGA core
    fpga = FPGACore(clock_freq_mhz=250)  # 250MHz FPGA
    
    # Create market data simulator
    simulator = MarketDataSimulator(['AAPL', 'GOOGL', 'MSFT'])
    
    # Subscribe FPGA to market data
    simulator.subscribe(lambda tick: fpga.feed_market_data(tick))
    
    # Start simulation
    simulator.start_feed()
    
    # Run for 1 second
    feed_task = asyncio.create_task(simulator.simulate_feed(tick_rate_hz=10000))
    
    # Let it run for a bit
    await asyncio.sleep(1.0)
    
    # Stop simulation
    simulator.stop_feed()
    feed_task.cancel()
    
    # Run FPGA simulation
    stats = fpga.run_simulation(duration_ms=100)
    
    print("Simulation Results:")
    print(f"Average latency: {stats.get('avg_latency_ns', 0):.1f}ns")
    print(f"Total trades: {stats.get('total_trades', 0)}")
    print(f"Cycle count: {stats.get('cycle_count', 0)}")

if __name__ == "__main__":
    asyncio.run(test_market_data_simulation())
