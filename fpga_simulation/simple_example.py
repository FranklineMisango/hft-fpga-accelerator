#!/usr/bin/env python3
"""
Simple FPGA Trading Simulation Example
Demonstrates basic simulation capabilities without heavy dependencies
"""

import asyncio
import time
import random
import sys
import os

# Add the python_sim directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_sim'))

# Simple implementations for demonstration
class SimpleMarketData:
    def __init__(self, symbol, price, volume, bid, ask):
        self.symbol = symbol
        self.timestamp = int(time.time() * 1000000)
        self.price = price
        self.volume = volume
        self.bid = bid
        self.ask = ask

class SimpleStrategy:
    def __init__(self, name):
        self.name = name
        self.trades = 0
        
    def execute(self, market_data):
        # Simple trading logic
        if random.random() < 0.1:  # 10% chance to trade
            self.trades += 1
            return {
                'action': 'BUY' if random.random() < 0.5 else 'SELL',
                'symbol': market_data.symbol,
                'price': market_data.price,
                'quantity': random.randint(100, 1000)
            }
        return None

class SimpleFPGASimulator:
    def __init__(self, clock_mhz=250):
        self.clock_mhz = clock_mhz
        self.clock_period_ns = 1000 / clock_mhz
        self.cycle_count = 0
        self.trades = []
        self.latencies = []
        self.strategies = []
        
    def add_strategy(self, strategy):
        self.strategies.append(strategy)
        
    def process_tick(self, market_data):
        """Process a single market data tick"""
        start_cycle = self.cycle_count
        
        # Simulate market data processing (1 cycle)
        self.cycle_count += 1
        
        # Execute strategies (1 cycle each)
        for strategy in self.strategies:
            self.cycle_count += 1
            trade = strategy.execute(market_data)
            
            if trade:
                # Simulate order processing (2 cycles)
                self.cycle_count += 2
                
                # Calculate latency
                latency_ns = (self.cycle_count - start_cycle) * self.clock_period_ns
                self.latencies.append(latency_ns)
                
                # Record trade
                trade['latency_ns'] = latency_ns
                trade['timestamp'] = market_data.timestamp
                self.trades.append(trade)
                
                print(f"Trade: {trade['action']} {trade['quantity']} {trade['symbol']} @ {trade['price']:.2f} ({latency_ns:.1f}ns)")
                
    def get_stats(self):
        """Get simulation statistics"""
        if not self.latencies:
            return {
                'total_trades': 0,
                'avg_latency_ns': 0,
                'cycle_count': self.cycle_count
            }
            
        return {
            'total_trades': len(self.trades),
            'avg_latency_ns': sum(self.latencies) / len(self.latencies),
            'min_latency_ns': min(self.latencies),
            'max_latency_ns': max(self.latencies),
            'cycle_count': self.cycle_count,
            'simulated_frequency_mhz': self.clock_mhz
        }

async def generate_market_data(symbols, duration_seconds=10):
    """Generate synthetic market data"""
    prices = {symbol: 100 + random.uniform(-50, 50) for symbol in symbols}
    
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        # Generate tick for random symbol
        symbol = random.choice(symbols)
        
        # Random walk price
        price_change = random.uniform(-0.5, 0.5)
        prices[symbol] = max(1.0, prices[symbol] + price_change)
        
        # Create market data
        price = prices[symbol]
        spread = price * 0.001  # 0.1% spread
        
        yield SimpleMarketData(
            symbol=symbol,
            price=price,
            volume=random.randint(100, 5000),
            bid=price - spread/2,
            ask=price + spread/2
        )
        
        await asyncio.sleep(0.001)  # 1ms between ticks

async def run_simple_simulation():
    """Run a simple FPGA trading simulation"""
    print("=== Simple FPGA Trading Simulation ===\n")
    
    # Create FPGA simulator
    fpga = SimpleFPGASimulator(clock_mhz=250)
    
    # Add strategies
    fpga.add_strategy(SimpleStrategy("Arbitrage"))
    fpga.add_strategy(SimpleStrategy("MarketMaking"))
    
    # Define symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print(f"Starting simulation with {len(symbols)} symbols...")
    print(f"FPGA clock frequency: {fpga.clock_mhz} MHz")
    print(f"Strategies: {[s.name for s in fpga.strategies]}")
    print("\nProcessing market data...\n")
    
    # Process market data
    tick_count = 0
    async for market_data in generate_market_data(symbols, duration_seconds=5):
        fpga.process_tick(market_data)
        tick_count += 1
        
        # Print periodic updates
        if tick_count % 1000 == 0:
            stats = fpga.get_stats()
            print(f"Processed {tick_count} ticks, {stats['total_trades']} trades")
    
    # Print final results
    stats = fpga.get_stats()
    
    print(f"\n=== Simulation Results ===")
    print(f"Total ticks processed: {tick_count}")
    print(f"Total trades: {stats['total_trades']}")
    print(f"Average latency: {stats['avg_latency_ns']:.1f} ns")
    print(f"Min latency: {stats['min_latency_ns']:.1f} ns")
    print(f"Max latency: {stats['max_latency_ns']:.1f} ns")
    print(f"Total cycles: {stats['cycle_count']:,}")
    print(f"Simulated frequency: {stats['simulated_frequency_mhz']} MHz")
    
    # Strategy performance
    print(f"\n=== Strategy Performance ===")
    for strategy in fpga.strategies:
        print(f"{strategy.name}: {strategy.trades} trades")
    
    # Calculate throughput
    if tick_count > 0:
        throughput = stats['cycle_count'] / tick_count
        print(f"\nThroughput: {throughput:.1f} cycles/tick")
        
    return stats

def run_benchmark():
    """Run performance benchmark"""
    print("=== Performance Benchmark ===\n")
    
    fpga = SimpleFPGASimulator(clock_mhz=250)
    
    # Benchmark clock cycles
    print("Benchmarking clock cycles...")
    start_time = time.time()
    cycles = 1000000
    
    for i in range(cycles):
        fpga.cycle_count += 1
        
    elapsed = time.time() - start_time
    cycles_per_second = cycles / elapsed
    
    print(f"Simulated {cycles:,} cycles in {elapsed:.3f} seconds")
    print(f"Performance: {cycles_per_second:,.0f} cycles/second")
    print(f"Equivalent frequency: {cycles_per_second / 1000000:.1f} MHz")
    
    # Benchmark market data processing
    print("\nBenchmarking market data processing...")
    market_data = SimpleMarketData('TEST', 100.0, 1000, 99.99, 100.01)
    
    start_time = time.time()
    iterations = 100000
    
    for i in range(iterations):
        fpga.process_tick(market_data)
        
    elapsed = time.time() - start_time
    ticks_per_second = iterations / elapsed
    
    print(f"Processed {iterations:,} ticks in {elapsed:.3f} seconds")
    print(f"Throughput: {ticks_per_second:,.0f} ticks/second")
    
    return {
        'cycles_per_second': cycles_per_second,
        'ticks_per_second': ticks_per_second
    }

def main():
    """Main function"""
    print("FPGA Trading Simulation - Simple Example")
    print("=" * 50)
    
    choice = input("\nChoose mode:\n1. Run simulation\n2. Run benchmark\n3. Both\n\nChoice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(run_simple_simulation())
    elif choice == "2":
        run_benchmark()
    elif choice == "3":
        print("Running benchmark first...\n")
        run_benchmark()
        print("\nRunning simulation...\n")
        asyncio.run(run_simple_simulation())
    else:
        print("Invalid choice. Running simulation...")
        asyncio.run(run_simple_simulation())

if __name__ == "__main__":
    main()
