# FPGA Trading Simulation

A comprehensive simulation environment for hardware-accelerated high-frequency trading systems. This simulation models FPGA-based trading infrastructure with sub-microsecond latency characteristics.

## Features

### Core Components

1. **FPGA Core Simulation** (`fpga_core.py`)
   - Hardware clock simulation (configurable frequency)
   - Market data processing pipeline
   - Order management system
   - Risk control engine
   - Performance metrics collection

2. **Market Data Simulator** (`market_data_simulator.py`)
   - Real-time market data feed simulation
   - Multiple protocol support (ITCH, FIX, OUCH)
   - Configurable tick rates and volatility
   - Burst mode for stress testing

3. **Trading Strategies** (`strategies.py`)
   - Arbitrage strategy
   - Market making strategy
   - TWAP (Time-Weighted Average Price)
   - VWAP (Volume-Weighted Average Price)
   - Momentum strategy
   - Pairs trading strategy

4. **Simulation Runner** (`simulation_runner.py`)
   - Complete simulation orchestration
   - Performance analysis
   - Risk metrics calculation
   - Report generation

## Quick Start

### 1. Setup Environment

```bash
# Using Make (recommended)
make setup

# Or manually
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Basic Test

```bash
make test
```

### 3. Run Full Simulation

```bash
make simulate
```

### 4. Run Performance Benchmark

```bash
make benchmark
```

## Architecture

### Hardware Simulation

The FPGA core simulates hardware behavior with:
- **Clock-accurate timing**: Each operation takes specific clock cycles
- **Parallel processing**: Multiple operations per clock cycle
- **Hardware pipelines**: Market data processing, order management, risk controls
- **Latency modeling**: Realistic sub-microsecond latencies

### Market Data Processing

```
Market Data → Hardware Parser → Order Book Update → Strategy Execution → Order Generation
     ↓              ↓                    ↓                    ↓                ↓
   1 cycle        1 cycle            1 cycle            1 cycle          1 cycle
```

### Trading Strategies

Each strategy implements hardware-optimized algorithms:

- **Arbitrage**: Exploits price differences with minimal latency
- **Market Making**: Provides liquidity with dynamic spreads
- **TWAP/VWAP**: Executes large orders with minimal market impact
- **Momentum**: Captures short-term price movements
- **Pairs Trading**: Statistical arbitrage between correlated assets

## Performance Characteristics

### Typical Latency Results

- **Market Data Processing**: 16-32 ns
- **Order Generation**: 32-64 ns
- **Risk Check**: 16 ns
- **Total Order-to-Wire**: 64-128 ns

### Throughput

- **Market Data**: 10M+ ticks/second
- **Order Processing**: 1M+ orders/second
- **Strategy Execution**: 100K+ decisions/second

## Configuration

### Basic Configuration

```python
config = {
    'fpga_clock_mhz': 250,           # FPGA clock frequency
    'symbols': ['AAPL', 'GOOGL'],    # Trading symbols
    'tick_rate_hz': 10000,           # Market data rate
    'duration_seconds': 60,          # Simulation duration
    'strategies': [
        {
            'type': 'arbitrage',
            'params': {
                'symbols': ['AAPL', 'GOOGL'],
                'min_profit_bps': 5
            }
        }
    ]
}
```

### Strategy Configuration

```python
# Arbitrage Strategy
{
    'type': 'arbitrage',
    'params': {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'min_profit_bps': 5  # Minimum profit in basis points
    }
}

# Market Making Strategy
{
    'type': 'market_making',
    'params': {
        'symbols': ['AAPL', 'GOOGL'],
        'spread_bps': 3,      # Bid-ask spread
        'max_position': 10000 # Maximum position size
    }
}

# TWAP Strategy
{
    'type': 'twap',
    'params': {
        'symbol': 'AAPL',
        'target_quantity': 10000,  # Shares to trade
        'duration_ms': 300000      # 5 minutes
    }
}
```

## Usage Examples

### Basic Simulation

```python
import asyncio
from simulation_runner import SimulationRunner

async def main():
    config = {
        'fpga_clock_mhz': 250,
        'symbols': ['AAPL', 'GOOGL'],
        'tick_rate_hz': 1000,
        'duration_seconds': 30
    }
    
    runner = SimulationRunner(config)
    
    # Add arbitrage strategy
    runner.add_strategy({
        'type': 'arbitrage',
        'params': {
            'symbols': ['AAPL', 'GOOGL'],
            'min_profit_bps': 5
        }
    })
    
    # Run simulation
    results = await runner.run_simulation()
    
    # Print results
    print(f"Total trades: {results['total_trades']}")
    print(f"Average latency: {results['latency_analysis']['mean_ns']:.1f}ns")

asyncio.run(main())
```

### Market Data Testing

```python
from market_data_simulator import MarketDataSimulator

simulator = MarketDataSimulator(['AAPL', 'GOOGL'])

# Generate burst of market data
ticks = simulator.simulate_burst('AAPL', num_ticks=1000)

# Test ITCH protocol
from market_data_simulator import ITCHSimulator
itch = ITCHSimulator()
message = itch.generate_itch_message('AAPL', 'A')  # Add order
parsed = itch.parse_itch_message(message)
```

### Strategy Development

```python
from strategies import TradingStrategy
from fpga_core import FPGACore, OrderType

class CustomStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("CustomStrategy")
        
    def execute(self, fpga: FPGACore):
        # Custom trading logic
        for symbol in ['AAPL', 'GOOGL']:
            book = fpga.get_order_book(symbol)
            if book.get('bids') and book.get('asks'):
                # Implement your strategy here
                pass
```

## Advanced Features

### Real-time Monitoring

```python
# Monitor simulation in real-time
runner = SimulationRunner(config)
runner.add_strategy(strategy_config)

# Setup monitoring
async def monitor():
    while True:
        stats = runner.fpga.get_performance_stats()
        print(f"Trades: {stats.get('total_trades', 0)}, "
              f"Latency: {stats.get('avg_latency_ns', 0):.1f}ns")
        await asyncio.sleep(1)

# Run with monitoring
asyncio.gather(
    runner.run_simulation(),
    monitor()
)
```

### Custom Market Data

```python
from fpga_core import MarketData

# Create custom market data
custom_data = MarketData(
    symbol="CUSTOM",
    timestamp=int(time.time() * 1000000),
    price=100.0,
    volume=1000,
    bid=99.99,
    ask=100.01
)

fpga.feed_market_data(custom_data)
```

### Performance Tuning

```python
# High-frequency configuration
config = {
    'fpga_clock_mhz': 400,        # Higher clock frequency
    'tick_rate_hz': 100000,       # Higher tick rate
    'batch_size': 1000,           # Process multiple ticks per cycle
    'enable_parallel': True       # Enable parallel processing
}
```

## Testing

### Unit Tests

```bash
# Run all tests
make test

# Run specific test categories
python test_simulation.py  # All tests
```

### Performance Tests

```bash
# Benchmark performance
make benchmark

# Custom benchmark
python -c "
from fpga_core import FPGACore
import time
fpga = FPGACore(clock_freq_mhz=250)
start = time.time()
for i in range(1000000):
    fpga.clock_tick()
print(f'Performance: {1000000/(time.time()-start):,.0f} cycles/sec')
"
```

### Example Runs

```bash
# Arbitrage example
make example-arbitrage

# Market making example
make example-market-making
```

## Output and Analysis

### Simulation Results

The simulation generates comprehensive results including:

```python
{
    'fpga_performance': {
        'avg_latency_ns': 45.2,
        'p99_latency_ns': 128.5,
        'total_trades': 15420,
        'cycle_count': 25000000
    },
    'positions': {
        'AAPL': 2500,
        'GOOGL': -1200
    },
    'portfolio_value': 1000000.0,
    'risk_metrics': {
        'gross_position': 125000.0,
        'net_position': 25000.0,
        'var_95': 20000.0
    }
}
```

### Reports

- **JSON Report**: `simulation_report.json`
- **Performance Charts**: `simulation_performance.png`
- **Trade Log**: `trade_log.csv`

## Development

### Code Structure

```
fpga_simulation/
├── python_sim/
│   ├── fpga_core.py           # Main FPGA simulation
│   ├── market_data_simulator.py  # Market data feeds
│   ├── strategies.py          # Trading strategies
│   └── simulation_runner.py   # Orchestration
├── test_simulation.py         # Test suite
├── requirements.txt           # Dependencies
├── Makefile                   # Build system
└── README.md                  # This file
```

### Adding New Strategies

1. Inherit from `TradingStrategy`
2. Implement `execute(self, fpga: FPGACore)` method
3. Add to strategy factory in `strategies.py`
4. Test with simulation runner

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-strategy`
3. Add tests for new functionality
4. Ensure all tests pass: `make test`
5. Submit pull request

## Hardware Requirements

### Development
- Python 3.8+
- 8GB+ RAM
- Multi-core CPU (for parallel simulation)

### Production Simulation
- 16GB+ RAM
- High-frequency CPU (3GHz+)
- SSD storage for data logging

## License

MIT License - see LICENSE file for details.

## References

- [FPGA Trading Systems](https://example.com/fpga-trading)
- [High-Frequency Trading](https://example.com/hft)
- [Market Data Protocols](https://example.com/market-data)
- [Trading Strategy Development](https://example.com/strategies)
