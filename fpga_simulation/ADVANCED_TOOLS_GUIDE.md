# Advanced FPGA Trading Tools Usage Guide

This guide covers the advanced tools and features available in the FPGA trading accelerator simulation environment.

## 🎯 Overview

The FPGA trading accelerator now includes several advanced tools for comprehensive simulation, monitoring, and analysis:

1. **Real-time Dashboard** - Interactive web-based monitoring
2. **Performance Profiler** - Advanced performance analysis
3. **GPU-FPGA Bridge** - Hybrid acceleration pipeline
4. **Web Monitor** - Real-time web interface
5. **Enhanced Market Data** - Realistic protocol simulation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install basic requirements
pip install -r requirements.txt

# Install optional GPU acceleration
pip install cupy-cuda11x numba

# Install web monitoring dependencies
pip install streamlit plotly flask flask-socketio

# Install performance profiling tools
pip install psutil memory-profiler line-profiler
```

### 2. Run Basic Tests

```bash
# Test all components
make test-tools

# Test individual components
python tools/performance_profiler.py
python tools/gpu_fpga_bridge.py
python tools/enhanced_market_data.py
```

## 📊 Real-time Dashboard

The Streamlit-based dashboard provides interactive visualization of trading performance.

### Features:
- Real-time latency monitoring
- Throughput analysis
- P&L tracking
- System status monitoring
- Interactive charts

### Usage:
```bash
# Start dashboard
streamlit run dashboard/dashboard.py

# Access in browser
open http://localhost:8501
```

### Configuration:
- **Clock Frequency**: Adjust FPGA clock speed (100-500 MHz)
- **Tick Rate**: Control market data frequency (100-10,000 Hz)
- **Strategy Selection**: Choose from multiple trading strategies
- **Risk Controls**: Set position limits and exposure controls

## 🔬 Performance Profiler

Advanced performance analysis tool for detailed system characterization.

### Features:
- Hardware-accurate latency measurement
- Throughput analysis
- Memory usage monitoring
- Performance regression detection
- Comprehensive reporting

### Usage:
```bash
# Run performance profiling
python tools/performance_profiler.py

# Custom profiling duration
python -c "
from tools.performance_profiler import PerformanceProfiler
profiler = PerformanceProfiler()
profiler.initialize_system(clock_freq=250e6, max_orders=1000)
profiler.start_profiling(duration_seconds=60)
"
```

### Output:
- **CSV Reports**: Detailed metrics data
- **JSON Summaries**: Performance statistics
- **Performance Plots**: Visualization of key metrics
- **Console Summary**: Real-time performance overview

### Example Results:
```
🎯 FPGA TRADING SYSTEM PERFORMANCE SUMMARY
============================================================
📊 Total Samples: 30,000
⏱️  Duration: 30.0 seconds
📈 Total Trades: 15,420
💰 Total P&L: $2,347.85

🚀 LATENCY STATISTICS:
   Average: 45.2 ns
   Median:  42.1 ns
   P95:     78.5 ns
   P99:     128.9 ns
   Max:     245.7 ns

⚡ THROUGHPUT STATISTICS:
   Average: 985,432 ops/sec
   Peak:    1,234,567 ops/sec
```

## 🖥️ GPU-FPGA Bridge

Hybrid acceleration pipeline combining GPU preprocessing with FPGA execution.

### Features:
- GPU-accelerated market data preprocessing
- FPGA hardware simulation
- Performance comparison
- Automatic fallback to CPU

### Usage:
```bash
# Run GPU-FPGA bridge demo
python tools/gpu_fpga_bridge.py

# Custom configuration
python -c "
from tools.gpu_fpga_bridge import FPGABridge
bridge = FPGABridge(fpga_clock_freq=250e6)
bridge.start_processing()
# ... your processing logic
bridge.stop_processing()
"
```

### GPU Acceleration Benefits:
- **Batch Processing**: Process multiple market data points simultaneously
- **Technical Indicators**: GPU-accelerated SMA, VWAP, volatility calculations
- **Pattern Recognition**: High-throughput pattern detection
- **Performance Gains**: 10-100x speedup for batch operations

### Example Performance:
```
📊 PERFORMANCE SUMMARY:
Total processed: 50,000
GPU processed: 500 batches
FPGA processed: 50,000 ticks
Overall throughput: 1,234,567 ops/sec
GPU speedup: 85.2x
```

## 🌐 Web Monitor

Real-time web-based monitoring interface with WebSocket support.

### Features:
- Live performance charts
- System status monitoring
- Real-time alerts
- Multi-client support
- RESTful API

### Usage:
```bash
# Start web monitor
python tools/web_monitor.py

# Access in browser
open http://localhost:5000
```

### API Endpoints:
- `GET /api/status` - System status
- `GET /api/metrics` - Performance metrics
- `POST /api/start` - Start monitoring
- `POST /api/stop` - Stop monitoring
- `POST /api/reset` - Reset metrics

### WebSocket Events:
- `metrics_update` - Real-time metrics
- `status` - Connection status
- `request_metrics` - Request metrics data

## 📡 Enhanced Market Data

Realistic market data simulation with multiple protocol support.

### Supported Protocols:
- **NASDAQ ITCH 5.0**: Binary market data protocol
- **FIX 4.4**: Financial Information eXchange protocol
- **OUCH 4.2**: Order entry protocol

### Features:
- Hardware-accurate protocol parsing
- Realistic message timing
- Configurable latency simulation
- Burst mode testing
- Statistical analysis

### Usage:
```bash
# Test all protocols
python tools/enhanced_market_data.py

# Custom protocol usage
python -c "
from tools.enhanced_market_data import EnhancedMarketDataFeed, ProtocolType

# Create ITCH feed
feed = EnhancedMarketDataFeed(protocol=ProtocolType.ITCH_5_0)
feed.set_message_callback(lambda msg: print(f'Received: {msg.message_type}'))
feed.start_feed(['AAPL', 'GOOGL', 'MSFT'])
"
```

### Protocol Examples:

#### ITCH 5.0 Messages:
```python
# Add Order (Type A)
{
    'reference_number': 1234567,
    'symbol': 'AAPL',
    'shares': 1000,
    'buy_sell': 'B',
    'price': 150.25,
    'side': 'BUY'
}

# Trade (Type P)
{
    'reference_number': 1234567,
    'symbol': 'AAPL',
    'shares': 500,
    'price': 150.26,
    'match_number': 987654
}
```

#### FIX 4.4 Messages:
```python
# Market Data Snapshot
{
    'symbol': 'AAPL',
    'bid_price': 150.25,
    'ask_price': 150.26,
    'bid_size': 1000,
    'ask_size': 800,
    'spread': 0.01,
    'mid_price': 150.255
}
```

## 🔧 Configuration

### System Configuration:
```python
# FPGA Core Settings
fpga_core = FPGACore(
    clock_freq=250e6,    # 250 MHz clock
    max_orders=1000,     # Maximum concurrent orders
    pipeline_depth=4     # Processing pipeline depth
)

# Market Data Settings
market_data = MarketDataSimulator(
    tick_rate=1000,      # 1000 ticks per second
    volatility=0.02,     # 2% volatility
    symbols=['AAPL', 'GOOGL', 'MSFT']
)
```

### Performance Tuning:
- **Clock Frequency**: Higher frequency = lower latency, higher power
- **Pipeline Depth**: Deeper pipeline = higher throughput, higher latency
- **Batch Size**: Larger batches = better GPU utilization
- **Tick Rate**: Higher rate = more realistic, higher CPU usage

## 📈 Performance Optimization

### Best Practices:
1. **Use GPU acceleration** for batch processing
2. **Optimize clock frequency** for your use case
3. **Monitor memory usage** to prevent bottlenecks
4. **Use burst mode** for stress testing
5. **Profile regularly** to identify bottlenecks

### Troubleshooting:
- **High latency**: Check CPU usage, reduce tick rate
- **Low throughput**: Increase pipeline depth, use GPU acceleration
- **Memory issues**: Reduce batch size, increase cleanup frequency
- **Connection issues**: Check firewall, port availability

## 🧪 Testing and Validation

### Unit Tests:
```bash
# Run all tests
make test-tools

# Individual tool tests
python -m pytest tools/test_performance_profiler.py
python -m pytest tools/test_gpu_fpga_bridge.py
python -m pytest tools/test_enhanced_market_data.py
```

### Integration Tests:
```bash
# Full system integration test
make integration-test

# Performance regression test
make regression-test
```

### Benchmarking:
```bash
# Performance benchmark
make benchmark-tools

# Latency benchmark
make latency-benchmark

# Throughput benchmark
make throughput-benchmark
```

## 📊 Reporting and Analysis

### Generated Reports:
- **Performance Reports**: Detailed performance analysis
- **Latency Reports**: Latency distribution and statistics
- **Throughput Reports**: Throughput analysis and trends
- **Error Reports**: Error analysis and debugging information

### Report Formats:
- **CSV**: Raw data for further analysis
- **JSON**: Structured data for programmatic access
- **HTML**: Interactive reports with charts
- **PDF**: Professional reports for documentation

## 🚀 Advanced Usage

### Custom Strategies:
```python
class CustomStrategy(Strategy):
    def process_market_data(self, market_data):
        # Custom strategy logic
        if self.should_trade(market_data):
            return self.generate_signal(market_data)
        return None
```

### Custom Protocols:
```python
class CustomProtocolSimulator:
    def generate_message(self, symbol):
        # Custom protocol implementation
        return MarketDataMessage(...)
```

### Performance Monitoring:
```python
from tools.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.initialize_system()
profiler.start_profiling()

# Your trading logic here

profiler.stop_profiling()
summary = profiler.get_performance_summary()
```

## 📚 Resources

### Documentation:
- [FPGA Trading Architecture](../README.md)
- [HDL Simulation Guide](../hdl_simulation/README.md)
- [Python Simulation Guide](../README.md)

### External Resources:
- [NASDAQ ITCH Specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/nqtv-itch-v5_0.pdf)
- [FIX Protocol Documentation](https://www.fixtrading.org/standards/)
- [OUCH Protocol Specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/TradingProducts/OUCH4.2.pdf)

### Community:
- [GitHub Issues](https://github.com/your-repo/issues)
- [Discord Channel](https://discord.gg/your-channel)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/fpga-trading)

## 🤝 Contributing

### Development Setup:
```bash
# Clone repository
git clone https://github.com/your-repo/hft-fpga-accelerator.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
make test-all

# Code formatting
make format

# Linting
make lint
```

### Contribution Guidelines:
1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new features**
4. **Ensure all tests pass**
5. **Submit a pull request**

### Code Standards:
- **Python**: PEP 8 compliance
- **Documentation**: Comprehensive docstrings
- **Testing**: 90%+ test coverage
- **Performance**: Benchmark new features

---

*This guide covers the advanced features of the FPGA trading accelerator. For basic usage, see the main [README](../README.md).*
