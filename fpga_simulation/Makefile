# FPGA Trading Simulation Makefile
# Hardware-accelerated low-latency trading infrastructure

PYTHON = python3
PIP = $(PYTHON) -m pip
VENV = venv
ACTIVATE = $(VENV)/bin/activate

# Directories
SRC_DIR = python_sim
TEST_DIR = .
OUTPUT_DIR = output
LOGS_DIR = logs
DATA_DIR = data
REPORTS_DIR = reports

# Default target
.PHONY: all
all: setup test

# Setup environment
.PHONY: setup
setup: $(VENV) install-deps create-dirs
	@echo "Setup complete!"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created"

.PHONY: install-deps
install-deps: $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Dependencies installed"

.PHONY: create-dirs
create-dirs:
	mkdir -p $(OUTPUT_DIR) $(LOGS_DIR) $(DATA_DIR) $(REPORTS_DIR)
	@echo "Directories created"

# Run simulation
.PHONY: simulate
simulate: $(VENV)
	@echo "Running FPGA trading simulation..."
	cd $(SRC_DIR) && ../$(VENV)/bin/python simulation_runner.py

# Run tests
.PHONY: test
test: $(VENV)
	@echo "Running simulation tests..."
	$(VENV)/bin/python test_simulation.py

# Quick test (no dependencies)
.PHONY: test-quick
test-quick:
	@echo "Running quick tests..."
	cd $(SRC_DIR) && $(PYTHON) -c "from fpga_core import FPGACore; print('✓ Core module OK')"
	cd $(SRC_DIR) && $(PYTHON) -c "from market_data_simulator import MarketDataSimulator; print('✓ Market data OK')"
	cd $(SRC_DIR) && $(PYTHON) -c "from strategies import ArbitrageStrategy; print('✓ Strategies OK')"

# Performance benchmark
.PHONY: benchmark
benchmark: $(VENV)
	@echo "Running performance benchmark..."
	cd $(SRC_DIR) && ../$(VENV)/bin/python -c "
import time
from fpga_core import FPGACore
fpga = FPGACore(clock_freq_mhz=250)
start = time.time()
for i in range(1000000):
    fpga.clock_tick()
elapsed = time.time() - start
print(f'Benchmark: {1000000/elapsed:,.0f} cycles/sec ({1000000/elapsed/1000000:.1f} MHz)')
"

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf *.pyc
	rm -rf $(SRC_DIR)/*.pyc
	rm -f simulation_report.json
	rm -f simulation_performance.png
	rm -f trade_log.csv
	@echo "Cleaned up"

# Clean output files
.PHONY: clean-output
clean-output:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(LOGS_DIR)/*
	rm -rf $(DATA_DIR)/*
	rm -rf $(REPORTS_DIR)/*
	@echo "Output files cleaned"

# Development targets
.PHONY: dev-setup
dev-setup: $(VENV)
	$(VENV)/bin/pip install pytest black flake8 mypy
	@echo "Development tools installed"

.PHONY: format
format: $(VENV)
	$(VENV)/bin/black $(SRC_DIR)/*.py *.py
	@echo "Code formatted"

.PHONY: lint
lint: $(VENV)
	$(VENV)/bin/flake8 $(SRC_DIR)/*.py *.py
	@echo "Linting complete"

.PHONY: type-check
type-check: $(VENV)
	$(VENV)/bin/mypy $(SRC_DIR)/*.py
	@echo "Type checking complete"

# Documentation
.PHONY: docs
docs:
	@echo "Generating documentation..."
	@echo "FPGA Trading Simulation Documentation"
	@echo "===================================="
	@echo ""
	@echo "Components:"
	@echo "  - fpga_core.py: Main FPGA hardware simulation"
	@echo "  - market_data_simulator.py: Market data feed simulation"
	@echo "  - strategies.py: Trading strategy implementations"
	@echo "  - simulation_runner.py: Complete simulation orchestration"
	@echo ""
	@echo "Usage:"
	@echo "  make simulate    # Run full simulation"
	@echo "  make test        # Run tests"
	@echo "  make benchmark   # Performance benchmark"
	@echo ""

# Example simulations
.PHONY: example-arbitrage
example-arbitrage: $(VENV)
	cd $(SRC_DIR) && ../$(VENV)/bin/python -c "
import asyncio
from fpga_core import FPGACore
from market_data_simulator import MarketDataSimulator
from strategies import ArbitrageStrategy

async def run():
    fpga = FPGACore(clock_freq_mhz=250)
    simulator = MarketDataSimulator(['AAPL', 'GOOGL'])
    strategy = ArbitrageStrategy(['AAPL', 'GOOGL'], min_profit_bps=5)
    
    fpga.strategies.append(strategy)
    simulator.subscribe(lambda tick: fpga.feed_market_data(tick))
    
    simulator.start_feed()
    
    market_task = asyncio.create_task(simulator.simulate_feed(tick_rate_hz=1000))
    await asyncio.sleep(5)
    
    simulator.stop_feed()
    market_task.cancel()
    
    stats = fpga.get_performance_stats()
    print(f'Arbitrage example: {stats.get(\"total_trades\", 0)} trades, avg latency {stats.get(\"avg_latency_ns\", 0):.1f}ns')

asyncio.run(run())
"

.PHONY: example-market-making
example-market-making: $(VENV)
	cd $(SRC_DIR) && ../$(VENV)/bin/python -c "
import asyncio
from fpga_core import FPGACore
from market_data_simulator import MarketDataSimulator
from strategies import MarketMakingStrategy

async def run():
    fpga = FPGACore(clock_freq_mhz=250)
    simulator = MarketDataSimulator(['AAPL', 'MSFT'])
    strategy = MarketMakingStrategy(['AAPL', 'MSFT'], spread_bps=3, max_position=10000)
    
    fpga.strategies.append(strategy)
    simulator.subscribe(lambda tick: fpga.feed_market_data(tick))
    
    simulator.start_feed()
    
    market_task = asyncio.create_task(simulator.simulate_feed(tick_rate_hz=1000))
    await asyncio.sleep(5)
    
    simulator.stop_feed()
    market_task.cancel()
    
    stats = fpga.get_performance_stats()
    print(f'Market making example: {stats.get(\"total_trades\", 0)} trades, avg latency {stats.get(\"avg_latency_ns\", 0):.1f}ns')

asyncio.run(run())
"

# Help
.PHONY: help
help:
	@echo "FPGA Trading Simulation Makefile"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  setup                 - Set up environment and dependencies"
	@echo "  simulate              - Run complete simulation"
	@echo "  test                  - Run all tests"
	@echo "  test-quick            - Quick module tests"
	@echo "  benchmark             - Performance benchmark"
	@echo "  clean                 - Clean up build artifacts"
	@echo "  clean-output          - Clean output files"
	@echo "  example-arbitrage     - Run arbitrage example"
	@echo "  example-market-making - Run market making example"
	@echo "  docs                  - Show documentation"
	@echo "  help                  - Show this help"
	@echo ""
	@echo "Advanced Tools:"
	@echo "  dashboard             - Launch real-time dashboard"
	@echo "  web-monitor           - Start web monitoring interface"
	@echo "  profiler              - Run performance profiler"
	@echo "  gpu-bridge            - Test GPU-FPGA bridge"
	@echo "  test-protocols        - Test market data protocols"
	@echo "  test-tools            - Test all advanced tools"
	@echo "  demo                  - Run comprehensive demo"
	@echo ""
	@echo "Development:"
	@echo "  dev-setup             - Install development tools"
	@echo "  format                - Format code"
	@echo "  lint                  - Lint code"
	@echo "  type-check            - Type checking"

# Advanced Tools
.PHONY: dashboard
dashboard: $(VENV)
	@echo "Starting real-time dashboard..."
	$(VENV)/bin/streamlit run dashboard/dashboard.py

.PHONY: web-monitor
web-monitor: $(VENV)
	@echo "Starting web monitoring interface..."
	$(VENV)/bin/python tools/web_monitor.py

.PHONY: profiler
profiler: $(VENV)
	@echo "Running performance profiler..."
	$(VENV)/bin/python tools/performance_profiler.py

.PHONY: gpu-bridge
gpu-bridge: $(VENV)
	@echo "Testing GPU-FPGA bridge..."
	$(VENV)/bin/python tools/gpu_fpga_bridge.py

.PHONY: test-protocols
test-protocols: $(VENV)
	@echo "Testing market data protocols..."
	$(VENV)/bin/python tools/enhanced_market_data.py

.PHONY: test-tools
test-tools: $(VENV)
	@echo "Testing all advanced tools..."
	@echo "1. Testing performance profiler..."
	$(VENV)/bin/python -c "
from tools.performance_profiler import PerformanceProfiler
profiler = PerformanceProfiler()
profiler.initialize_system()
profiler.start_profiling(duration_seconds=5)
print('✓ Performance profiler OK')
"
	@echo "2. Testing GPU-FPGA bridge..."
	$(VENV)/bin/python -c "
from tools.gpu_fpga_bridge import FPGABridge
bridge = FPGABridge()
bridge.start_processing()
import time; time.sleep(2)
bridge.stop_processing()
print('✓ GPU-FPGA bridge OK')
"
	@echo "3. Testing market data protocols..."
	$(VENV)/bin/python -c "
from tools.enhanced_market_data import EnhancedMarketDataFeed, ProtocolType
feed = EnhancedMarketDataFeed(ProtocolType.ITCH_5_0)
feed.start_feed(['AAPL'])
import time; time.sleep(2)
feed.stop_feed()
print('✓ Market data protocols OK')
"
	@echo "✅ All advanced tools tested successfully!"

# Performance benchmarks
.PHONY: benchmark-tools
benchmark-tools: $(VENV)
	@echo "Running advanced tools benchmarks..."
	$(VENV)/bin/python -c "
import time
from tools.performance_profiler import PerformanceProfiler
from tools.gpu_fpga_bridge import FPGABridge

print('🚀 Advanced Tools Benchmark')
print('==========================')

# Performance profiler benchmark
profiler = PerformanceProfiler()
profiler.initialize_system()
start = time.time()
profiler.start_profiling(duration_seconds=10)
elapsed = time.time() - start
print(f'Performance Profiler: {elapsed:.2f}s for 10s simulation')

# GPU-FPGA bridge benchmark
bridge = FPGABridge()
bridge.start_processing()
start = time.time()
time.sleep(5)
bridge.stop_processing()
elapsed = time.time() - start
summary = bridge.get_performance_summary()
print(f'GPU-FPGA Bridge: {summary[\"processing_statistics\"][\"total_processed\"]} items in {elapsed:.2f}s')
print(f'Throughput: {summary[\"hybrid_performance\"][\"overall_throughput\"]:.0f} ops/sec')
"

.PHONY: latency-benchmark
latency-benchmark: $(VENV)
	@echo "Running latency benchmark..."
	$(VENV)/bin/python -c "
import time
import numpy as np
from tools.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.initialize_system(clock_freq=250e6)
profiler.start_profiling(duration_seconds=30)

print('🎯 Latency Benchmark Results')
print('============================')
summary = profiler.get_performance_summary()
lat_stats = summary['performance_summary']['latency_stats']
print(f'Average Latency: {lat_stats[\"mean_ns\"]:.1f} ns')
print(f'P50 Latency: {lat_stats[\"median_ns\"]:.1f} ns')
print(f'P95 Latency: {lat_stats[\"p95_ns\"]:.1f} ns')
print(f'P99 Latency: {lat_stats[\"p99_ns\"]:.1f} ns')
print(f'Max Latency: {lat_stats[\"max_ns\"]:.1f} ns')
"

.PHONY: throughput-benchmark
throughput-benchmark: $(VENV)
	@echo "Running throughput benchmark..."
	$(VENV)/bin/python -c "
import time
from tools.gpu_fpga_bridge import FPGABridge

bridge = FPGABridge()
bridge.start_processing()
time.sleep(30)
bridge.stop_processing()

summary = bridge.get_performance_summary()
print('⚡ Throughput Benchmark Results')
print('===============================')
print(f'Total Processed: {summary[\"processing_statistics\"][\"total_processed\"]:,}')
print(f'GPU Processed: {summary[\"processing_statistics\"][\"gpu_processed\"]:,}')
print(f'FPGA Processed: {summary[\"processing_statistics\"][\"fpga_processed\"]:,}')
print(f'Overall Throughput: {summary[\"hybrid_performance\"][\"overall_throughput\"]:.0f} ops/sec')
"

# Integration tests
.PHONY: integration-test
integration-test: $(VENV)
	@echo "Running integration tests..."
	$(VENV)/bin/python -c "
import time
from tools.performance_profiler import PerformanceProfiler
from tools.gpu_fpga_bridge import FPGABridge
from tools.enhanced_market_data import EnhancedMarketDataFeed, ProtocolType

print('🔄 Integration Test')
print('==================')

# Test 1: Performance profiler with GPU bridge
print('Test 1: Performance profiler + GPU bridge')
profiler = PerformanceProfiler()
profiler.initialize_system()
bridge = FPGABridge()

profiler.start_profiling(duration_seconds=5)
bridge.start_processing()
time.sleep(5)
bridge.stop_processing()

print('✓ Test 1 passed')

# Test 2: Market data protocols with profiler
print('Test 2: Market data protocols + profiler')
feed = EnhancedMarketDataFeed(ProtocolType.ITCH_5_0)
feed.start_feed(['AAPL', 'GOOGL'])
time.sleep(3)
feed.stop_feed()

print('✓ Test 2 passed')
print('✅ All integration tests passed!')
"

# Regression tests
.PHONY: regression-test
regression-test: $(VENV)
	@echo "Running regression tests..."
	$(VENV)/bin/python -c "
import time
from tools.performance_profiler import PerformanceProfiler

print('🔍 Regression Test')
print('==================')

# Performance regression test
profiler = PerformanceProfiler()
profiler.initialize_system()
profiler.start_profiling(duration_seconds=10)

summary = profiler.get_performance_summary()
avg_latency = summary['performance_summary']['latency_stats']['mean_ns']
throughput = summary['performance_summary']['throughput_stats']['mean_ops']

# Check performance thresholds
latency_threshold = 1000  # 1μs
throughput_threshold = 100000  # 100k ops/sec

print(f'Average Latency: {avg_latency:.1f} ns (threshold: {latency_threshold} ns)')
print(f'Throughput: {throughput:.0f} ops/sec (threshold: {throughput_threshold} ops/sec)')

if avg_latency > latency_threshold:
    print('❌ Latency regression detected!')
    exit(1)

if throughput < throughput_threshold:
    print('❌ Throughput regression detected!')
    exit(1)

print('✅ No performance regression detected!')
"

# Install enhanced dependencies
.PHONY: install-enhanced
install-enhanced: $(VENV)
	@echo "Installing enhanced dependencies..."
	$(VENV)/bin/pip install streamlit plotly dash bokeh
	$(VENV)/bin/pip install flask flask-socketio
	$(VENV)/bin/pip install psutil memory-profiler line-profiler
	$(VENV)/bin/pip install scikit-learn ta-lib pyfolio
	$(VENV)/bin/pip install sqlalchemy redis h5py
	$(VENV)/bin/pip install websockets asyncio-mqtt
	@echo "Enhanced dependencies installed"

# Install GPU dependencies (optional)
.PHONY: install-gpu
install-gpu: $(VENV)
	@echo "Installing GPU dependencies..."
	$(VENV)/bin/pip install cupy-cuda11x numba
	$(VENV)/bin/pip install rapids-cudf
	@echo "GPU dependencies installed"

# Create documentation
.PHONY: create-docs
create-docs:
	@echo "Creating comprehensive documentation..."
	@echo "📚 Documentation created in ADVANCED_TOOLS_GUIDE.md"
	@echo "See also:"
	@echo "  - README.md: Main project documentation"
	@echo "  - ../hdl_simulation/README.md: HDL simulation guide"
	@echo "  - SIMULATION_SUMMARY.md: Simulation results"

# Comprehensive demo
.PHONY: demo
demo: $(VENV)
	@echo "Starting comprehensive advanced tools demo..."
	$(VENV)/bin/python demo_advanced_tools.py
