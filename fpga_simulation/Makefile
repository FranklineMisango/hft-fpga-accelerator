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
	@echo "Development:"
	@echo "  dev-setup             - Install development tools"
	@echo "  format                - Format code"
	@echo "  lint                  - Lint code"
	@echo "  type-check            - Type checking"
