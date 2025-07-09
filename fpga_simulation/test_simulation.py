#!/usr/bin/env python3
"""
Test script for FPGA Trading Simulation
Simple test to verify all components work together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'python_sim'))

import asyncio
import time
from fpga_core import FPGACore, OrderType, MarketData
from market_data_simulator import MarketDataSimulator
from strategies import ArbitrageStrategy, MomentumStrategy

async def test_basic_simulation():
    """Test basic simulation functionality"""
    print("Testing basic FPGA simulation...")
    
    # Create FPGA core
    fpga = FPGACore(clock_freq_mhz=100)
    
    # Create some test market data
    test_data = [
        MarketData("AAPL", int(time.time() * 1000000), 150.0, 1000, 149.99, 150.01),
        MarketData("GOOGL", int(time.time() * 1000000), 2800.0, 500, 2799.50, 2800.50),
        MarketData("MSFT", int(time.time() * 1000000), 300.0, 800, 299.98, 300.02),
    ]
    
    # Feed market data
    for data in test_data:
        fpga.feed_market_data(data)
    
    # Submit some test orders
    fpga.submit_order("AAPL", OrderType.BUY, 150.01, 100)
    fpga.submit_order("GOOGL", OrderType.SELL, 2799.50, 50)
    
    # Run simulation
    stats = fpga.run_simulation(duration_ms=100)
    
    print(f"✓ Basic simulation completed")
    print(f"  Cycles: {stats.get('cycle_count', 0)}")
    print(f"  Trades: {stats.get('total_trades', 0)}")
    
    return True

async def test_market_data_simulator():
    """Test market data simulator"""
    print("Testing market data simulator...")
    
    simulator = MarketDataSimulator(['AAPL', 'GOOGL'])
    received_ticks = []
    
    async def collect_tick(tick):
        received_ticks.append(tick)
    
    simulator.subscribe(collect_tick)
    simulator.start_feed()
    
    # Generate some ticks
    feed_task = asyncio.create_task(simulator.simulate_feed(tick_rate_hz=100))
    await asyncio.sleep(0.1)  # Run for 100ms
    
    simulator.stop_feed()
    feed_task.cancel()
    
    print(f"✓ Market data simulator completed")
    print(f"  Ticks received: {len(received_ticks)}")
    
    return len(received_ticks) > 0

async def test_strategies():
    """Test trading strategies"""
    print("Testing trading strategies...")
    
    fpga = FPGACore(clock_freq_mhz=100)
    
    # Add strategies
    arb_strategy = ArbitrageStrategy(['AAPL', 'GOOGL'], min_profit_bps=5)
    momentum_strategy = MomentumStrategy(['AAPL', 'GOOGL'], lookback_periods=5)
    
    fpga.strategies.append(arb_strategy)
    fpga.strategies.append(momentum_strategy)
    
    # Feed some market data
    test_data = [
        MarketData("AAPL", int(time.time() * 1000000), 150.0, 1000, 149.95, 150.05),
        MarketData("GOOGL", int(time.time() * 1000000), 2800.0, 500, 2799.00, 2801.00),
    ]
    
    for data in test_data:
        fpga.feed_market_data(data)
    
    # Run simulation
    stats = fpga.run_simulation(duration_ms=50)
    
    print(f"✓ Strategy testing completed")
    print(f"  Strategies: {len(fpga.strategies)}")
    
    return True

async def test_integrated_simulation():
    """Test integrated simulation with all components"""
    print("Testing integrated simulation...")
    
    # Create components
    fpga = FPGACore(clock_freq_mhz=200)
    simulator = MarketDataSimulator(['AAPL', 'GOOGL', 'MSFT'])
    
    # Add strategies
    arb_strategy = ArbitrageStrategy(['AAPL', 'GOOGL'], min_profit_bps=10)
    fpga.strategies.append(arb_strategy)
    
    # Connect market data
    simulator.subscribe(lambda tick: fpga.feed_market_data(tick))
    
    # Run simulation
    simulator.start_feed()
    fpga.start_simulation()
    
    # Create tasks
    market_task = asyncio.create_task(simulator.simulate_feed(tick_rate_hz=1000))
    
    # Run for a short time
    await asyncio.sleep(0.5)
    
    # Stop simulation
    simulator.stop_feed()
    market_task.cancel()
    
    stats = fpga.get_performance_stats()
    
    print(f"✓ Integrated simulation completed")
    print(f"  Trades: {stats.get('total_trades', 0)}")
    print(f"  Avg latency: {stats.get('avg_latency_ns', 0):.1f}ns")
    
    return True

def test_performance():
    """Test performance metrics"""
    print("Testing performance metrics...")
    
    fpga = FPGACore(clock_freq_mhz=250)
    
    # Simulate high-frequency trading
    start_time = time.time()
    
    for i in range(10000):
        fpga.clock_tick()
    
    elapsed = time.time() - start_time
    cycles_per_second = 10000 / elapsed
    
    print(f"✓ Performance test completed")
    print(f"  Cycles per second: {cycles_per_second:,.0f}")
    print(f"  Simulated frequency: {cycles_per_second / 1000000:.1f} MHz")
    
    return cycles_per_second > 1000000  # Should be > 1MHz

async def main():
    """Run all tests"""
    print("=== FPGA Trading Simulation Tests ===\n")
    
    tests = [
        ("Basic Simulation", test_basic_simulation()),
        ("Market Data Simulator", test_market_data_simulator()),
        ("Trading Strategies", test_strategies()),
        ("Integrated Simulation", test_integrated_simulation()),
        ("Performance", test_performance())
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_coro in tests:
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
                
            if result:
                print(f"✓ {test_name}: PASSED\n")
                passed += 1
            else:
                print(f"✗ {test_name}: FAILED\n")
                
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}\n")
    
    print(f"=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("All tests passed! The simulation is ready to use.")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
