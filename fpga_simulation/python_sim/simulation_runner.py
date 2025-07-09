"""
FPGA Trading Simulation Runner
Complete simulation environment for testing trading strategies on FPGA hardware

Features:
- Real-time market data simulation
- Multiple trading strategies
- Performance analysis
- Risk management
- Hardware latency simulation
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Any
from fpga_core import FPGACore, OrderType, MarketData
from market_data_simulator import MarketDataSimulator
from strategies import (
    ArbitrageStrategy, MarketMakingStrategy, TWAPStrategy, 
    VWAPStrategy, MomentumStrategy, PairsTradingStrategy
)

class SimulationRunner:
    """
    Main simulation runner for FPGA trading system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fpga = FPGACore(clock_freq_mhz=config.get('fpga_clock_mhz', 250))
        self.market_simulator = MarketDataSimulator(config.get('symbols', ['AAPL', 'GOOGL', 'MSFT']))
        self.strategies = []
        self.results = {}
        self.trade_log = []
        self.performance_metrics = {}
        
    def add_strategy(self, strategy_config: Dict[str, Any]):
        """Add a trading strategy to the simulation"""
        strategy_type = strategy_config['type']
        strategy_params = strategy_config.get('params', {})
        
        if strategy_type == 'arbitrage':
            strategy = ArbitrageStrategy(
                symbols=strategy_params.get('symbols', self.market_simulator.symbols),
                min_profit_bps=strategy_params.get('min_profit_bps', 10)
            )
        elif strategy_type == 'market_making':
            strategy = MarketMakingStrategy(
                symbols=strategy_params.get('symbols', self.market_simulator.symbols),
                spread_bps=strategy_params.get('spread_bps', 5),
                max_position=strategy_params.get('max_position', 5000)
            )
        elif strategy_type == 'twap':
            strategy = TWAPStrategy(
                symbol=strategy_params['symbol'],
                target_quantity=strategy_params['target_quantity'],
                duration_ms=strategy_params.get('duration_ms', 60000)
            )
        elif strategy_type == 'vwap':
            strategy = VWAPStrategy(
                symbol=strategy_params['symbol'],
                target_quantity=strategy_params['target_quantity'],
                volume_profile=strategy_params.get('volume_profile', [1]*20)
            )
        elif strategy_type == 'momentum':
            strategy = MomentumStrategy(
                symbols=strategy_params.get('symbols', self.market_simulator.symbols),
                lookback_periods=strategy_params.get('lookback_periods', 10),
                momentum_threshold=strategy_params.get('momentum_threshold', 0.001)
            )
        elif strategy_type == 'pairs':
            strategy = PairsTradingStrategy(
                symbol1=strategy_params['symbol1'],
                symbol2=strategy_params['symbol2'],
                hedge_ratio=strategy_params.get('hedge_ratio', 1.0),
                entry_threshold=strategy_params.get('entry_threshold', 2.0),
                exit_threshold=strategy_params.get('exit_threshold', 0.5)
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
            
        self.strategies.append(strategy)
        self.fpga.strategies.append(strategy)
        
        print(f"Added strategy: {strategy.name}")
        
    async def run_simulation(self, duration_seconds: int = 60):
        """Run the complete simulation"""
        print(f"Starting FPGA trading simulation for {duration_seconds} seconds...")
        
        # Setup market data feed
        self.market_simulator.subscribe(self._on_market_data)
        
        # Start components
        self.fpga.start_simulation()
        self.market_simulator.start_feed()
        
        # Create tasks
        market_task = asyncio.create_task(
            self.market_simulator.simulate_feed(
                tick_rate_hz=self.config.get('tick_rate_hz', 1000)
            )
        )
        
        fpga_task = asyncio.create_task(self._run_fpga_loop(duration_seconds))
        
        # Wait for completion
        try:
            await asyncio.wait_for(
                asyncio.gather(market_task, fpga_task),
                timeout=duration_seconds + 5
            )
        except asyncio.TimeoutError:
            print("Simulation timed out")
            
        # Cleanup
        market_task.cancel()
        fpga_task.cancel()
        
        self.market_simulator.stop_feed()
        self.fpga.stop_simulation()
        
        # Generate results
        self.results = self._generate_results()
        
        print("Simulation completed!")
        return self.results
        
    async def _on_market_data(self, tick: MarketData):
        """Handle incoming market data"""
        self.fpga.feed_market_data(tick)
        
    async def _run_fpga_loop(self, duration_seconds: int):
        """Run FPGA simulation loop"""
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Run multiple clock cycles per iteration
            for _ in range(1000):  # 1000 cycles per iteration
                self.fpga.clock_tick()
                
            await asyncio.sleep(0.001)  # 1ms sleep
            
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive simulation results"""
        fpga_stats = self.fpga.get_performance_stats()
        
        results = {
            'fpga_performance': fpga_stats,
            'positions': dict(self.fpga.positions),
            'portfolio_value': self.fpga.portfolio_value,
            'total_trades': len(self.fpga.latency_stats),
            'strategies': [strategy.name for strategy in self.strategies],
            'market_data_ticks': len(self.market_simulator.prices),
            'risk_metrics': self._calculate_risk_metrics(),
            'latency_analysis': self._analyze_latency()
        }
        
        return results
        
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics"""
        positions = self.fpga.positions
        
        # Calculate position sizes
        total_long = sum(pos for pos in positions.values() if pos > 0)
        total_short = sum(abs(pos) for pos in positions.values() if pos < 0)
        gross_position = total_long + total_short
        net_position = total_long - total_short
        
        # Calculate VaR approximation
        portfolio_volatility = 0.02  # 2% daily volatility assumption
        confidence_level = 0.95
        var_95 = self.fpga.portfolio_value * portfolio_volatility * 2.33  # 95% VaR
        
        return {
            'gross_position': gross_position,
            'net_position': net_position,
            'position_utilization': gross_position / self.fpga.portfolio_value,
            'var_95': var_95,
            'max_drawdown': 0.0  # Would need P&L history
        }
        
    def _analyze_latency(self) -> Dict[str, Any]:
        """Analyze latency performance"""
        if not self.fpga.latency_stats:
            return {}
            
        latencies = np.array(self.fpga.latency_stats)
        
        return {
            'mean_ns': float(np.mean(latencies)),
            'median_ns': float(np.median(latencies)),
            'p95_ns': float(np.percentile(latencies, 95)),
            'p99_ns': float(np.percentile(latencies, 99)),
            'p99_9_ns': float(np.percentile(latencies, 99.9)),
            'max_ns': float(np.max(latencies)),
            'std_ns': float(np.std(latencies))
        }
        
    def generate_report(self, output_file: str = "simulation_report.json"):
        """Generate detailed simulation report"""
        if not self.results:
            print("No results to report. Run simulation first.")
            return
            
        report = {
            'simulation_config': self.config,
            'results': self.results,
            'timestamp': time.time(),
            'duration': self.config.get('duration_seconds', 60)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Report saved to {output_file}")
        
    def plot_performance(self):
        """Plot performance metrics"""
        if not self.results:
            print("No results to plot. Run simulation first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Latency distribution
        if self.fpga.latency_stats:
            axes[0, 0].hist(self.fpga.latency_stats, bins=50)
            axes[0, 0].set_title('Latency Distribution (ns)')
            axes[0, 0].set_xlabel('Latency (ns)')
            axes[0, 0].set_ylabel('Frequency')
            
        # Positions
        if self.fpga.positions:
            symbols = list(self.fpga.positions.keys())
            positions = list(self.fpga.positions.values())
            axes[0, 1].bar(symbols, positions)
            axes[0, 1].set_title('Final Positions')
            axes[0, 1].set_ylabel('Position Size')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
        # Strategy performance (placeholder)
        axes[1, 0].bar(['Total Trades'], [len(self.fpga.latency_stats)])
        axes[1, 0].set_title('Trading Activity')
        axes[1, 0].set_ylabel('Number of Trades')
        
        # Risk metrics
        risk_metrics = self.results.get('risk_metrics', {})
        if risk_metrics:
            metrics = ['Gross Position', 'Net Position', 'VaR 95%']
            values = [
                risk_metrics.get('gross_position', 0),
                risk_metrics.get('net_position', 0),
                risk_metrics.get('var_95', 0)
            ]
            axes[1, 1].bar(metrics, values)
            axes[1, 1].set_title('Risk Metrics')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig('simulation_performance.png')
        plt.show()
        
    def export_trade_log(self, output_file: str = "trade_log.csv"):
        """Export trade log to CSV"""
        if not self.trade_log:
            print("No trades to export.")
            return
            
        df = pd.DataFrame(self.trade_log)
        df.to_csv(output_file, index=False)
        print(f"Trade log exported to {output_file}")

# Example configuration
DEFAULT_CONFIG = {
    'fpga_clock_mhz': 250,
    'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
    'tick_rate_hz': 10000,
    'duration_seconds': 60,
    'strategies': [
        {
            'type': 'arbitrage',
            'params': {
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'min_profit_bps': 5
            }
        },
        {
            'type': 'market_making',
            'params': {
                'symbols': ['AAPL', 'GOOGL'],
                'spread_bps': 3,
                'max_position': 10000
            }
        },
        {
            'type': 'momentum',
            'params': {
                'symbols': ['TSLA', 'NVDA'],
                'lookback_periods': 20,
                'momentum_threshold': 0.002
            }
        }
    ]
}

async def main():
    """Main simulation entry point"""
    # Create simulation runner
    runner = SimulationRunner(DEFAULT_CONFIG)
    
    # Add strategies
    for strategy_config in DEFAULT_CONFIG['strategies']:
        runner.add_strategy(strategy_config)
        
    # Run simulation
    results = await runner.run_simulation(duration_seconds=30)
    
    # Print results
    print("\n=== SIMULATION RESULTS ===")
    print(f"Total trades: {results['total_trades']}")
    print(f"Average latency: {results['latency_analysis'].get('mean_ns', 0):.1f}ns")
    print(f"P99 latency: {results['latency_analysis'].get('p99_ns', 0):.1f}ns")
    print(f"Portfolio value: ${results['portfolio_value']:,.2f}")
    
    # Generate report
    runner.generate_report()
    
    # Plot performance
    runner.plot_performance()

if __name__ == "__main__":
    asyncio.run(main())
