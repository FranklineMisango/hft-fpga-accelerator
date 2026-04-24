# HDL Simulation Makefile
# Supports both Icarus Verilog and Verilator simulation
# 
# Usage:
#   make iverilog          - Run Icarus Verilog simulation
#   make verilator         - Run Verilator simulation
#   make all               - Run all simulations
#   make clean             - Clean up generated files
#   make wave              - View waveforms with GTKWave

# Directories
RTL_DIR = rtl
TB_DIR = testbench
CPP_TB_DIR = cpp_testbench
SIM_DIR = sim

# RTL sources
RTL_SOURCES = $(RTL_DIR)/market_data_processor.v \
              $(RTL_DIR)/order_manager.v \
              $(RTL_DIR)/trading_strategy.v \
              $(RTL_DIR)/hjb_calculator.v

# Testbench sources
TB_SOURCES = $(TB_DIR)/market_data_tb.v \
             $(TB_DIR)/order_manager_tb.v \
             $(TB_DIR)/trading_strategy_tb.v \
             $(TB_DIR)/fpga_trading_system_tb.v \
             $(TB_DIR)/hjb_calculator_tb.v

# Simulation tools
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave
VERILATOR = verilator
Q_BIN ?= q

# Compilation flags
IVERILOG_FLAGS = -g2012 -Wall -Winfloop
VERILATOR_FLAGS = --cc --exe --build --trace -Wall -Wno-fatal

# Default target
.PHONY: all
all: iverilog verilator

# Create simulation directory
$(SIM_DIR):
	mkdir -p $(SIM_DIR)

# Icarus Verilog simulation targets
.PHONY: iverilog
iverilog: iverilog-market-data iverilog-order-manager iverilog-trading-strategy iverilog-integration

.PHONY: iverilog-market-data
iverilog-market-data: $(SIM_DIR)
	@echo "Running Icarus Verilog simulation for Market Data Processor..."
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(SIM_DIR)/market_data_tb \
		$(RTL_DIR)/market_data_processor.v $(TB_DIR)/market_data_tb.v
	cd $(SIM_DIR) && $(VVP) market_data_tb
	@echo "Market Data Processor simulation completed"

.PHONY: iverilog-order-manager
iverilog-order-manager: $(SIM_DIR)
	@echo "Running Icarus Verilog simulation for Order Manager..."
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(SIM_DIR)/order_manager_tb \
		$(RTL_DIR)/order_manager.v $(TB_DIR)/order_manager_tb.v
	cd $(SIM_DIR) && $(VVP) order_manager_tb
	@echo "Order Manager simulation completed"

.PHONY: iverilog-trading-strategy
iverilog-trading-strategy: $(SIM_DIR)
	@echo "Running Icarus Verilog simulation for Trading Strategy..."
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(SIM_DIR)/trading_strategy_tb \
		$(RTL_DIR)/trading_strategy.v $(TB_DIR)/trading_strategy_tb.v
	cd $(SIM_DIR) && $(VVP) trading_strategy_tb
	@echo "Trading Strategy simulation completed"

.PHONY: iverilog-hjb
iverilog-hjb: $(SIM_DIR)
	@echo "Running HJB Calculator simulation..."
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(SIM_DIR)/hjb_calculator_tb \
		$(RTL_DIR)/hjb_calculator.v $(TB_DIR)/hjb_calculator_tb.v
	cd $(SIM_DIR) && $(VVP) hjb_calculator_tb
	@echo "HJB Calculator simulation completed"

.PHONY: iverilog-maker
iverilog-maker: $(SIM_DIR)
	@echo "Running HJB market-maker simulation..."
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(SIM_DIR)/verilog_market_maker_tb \
		$(RTL_DIR)/verilog_market_maker.v \
		$(RTL_DIR)/hjb_toxicity_tracker.v \
		$(RTL_DIR)/hjb_jump_operator.v \
		$(TB_DIR)/verilog_market_maker_tb.v
	cd $(SIM_DIR) && $(VVP) verilog_market_maker_tb
	@echo "HJB market-maker simulation completed"

.PHONY: iverilog-integration
iverilog-integration: $(SIM_DIR)
	@echo "Running Icarus Verilog integration simulation..."
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(SIM_DIR)/fpga_trading_system_tb \
		$(RTL_SOURCES) $(TB_DIR)/fpga_trading_system_tb.v
	cd $(SIM_DIR) && $(VVP) fpga_trading_system_tb
	@echo "Integration simulation completed"

# Verilator simulation targets
.PHONY: verilator
verilator: verilator-cpp verilator-market-data

.PHONY: verilator-cpp
verilator-cpp: $(SIM_DIR)
	@echo "Running Verilator C++ simulation..."
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module fpga_trading_system_tb \
		-I$(RTL_DIR) \
		$(RTL_SOURCES) $(TB_DIR)/fpga_trading_system_tb.v \
		$(CPP_TB_DIR)/fpga_trading_system_test.cpp
	@echo "Verilator C++ simulation completed"

.PHONY: verilator-market-data
verilator-market-data: $(SIM_DIR)
	@echo "Running Verilator market data simulation..."
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module market_data_tb \
		-I$(RTL_DIR) \
		$(RTL_DIR)/market_data_processor.v $(TB_DIR)/market_data_tb.v \
		$(CPP_TB_DIR)/market_data_generator.cpp
	@echo "Verilator market data simulation completed"

.PHONY: verilator-hjb-lib
verilator-hjb-lib: $(SIM_DIR)
	@echo "Building Verilator HJB library..."
	$(VERILATOR) --cc --build -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
		--top-module hjb_calculator \
		-I$(RTL_DIR) \
		$(RTL_DIR)/hjb_calculator.v \
		cpp_wrapper/hjb_wrapper.cpp \
		cpp_wrapper/main.cpp \
		-CFLAGS "-fPIC" \
		-LDFLAGS "-shared -fPIC" \
		--exe
	@echo "HJB library built successfully"

# Performance benchmarks
.PHONY: benchmark
benchmark: benchmark-iverilog benchmark-verilator

.PHONY: benchmark-iverilog
benchmark-iverilog: $(SIM_DIR)
	@echo "Running Icarus Verilog performance benchmark..."
	time $(MAKE) iverilog-integration
	@echo "Icarus Verilog benchmark completed"

.PHONY: benchmark-verilator
benchmark-verilator: $(SIM_DIR)
	@echo "Running Verilator performance benchmark..."
	time $(MAKE) verilator-cpp
	@echo "Verilator benchmark completed"

# Waveform viewing
.PHONY: wave
wave: wave-market-data

.PHONY: wave-market-data
wave-market-data:
	@echo "Opening market data waveform..."
	$(GTKWAVE) $(SIM_DIR)/market_data_tb.vcd &

.PHONY: wave-order-manager
wave-order-manager:
	@echo "Opening order manager waveform..."
	$(GTKWAVE) $(SIM_DIR)/order_manager_tb.vcd &

.PHONY: wave-trading-strategy
wave-trading-strategy:
	@echo "Opening trading strategy waveform..."
	$(GTKWAVE) $(SIM_DIR)/trading_strategy_tb.vcd &

.PHONY: wave-integration
wave-integration:
	@echo "Opening integration waveform..."
	$(GTKWAVE) $(SIM_DIR)/fpga_trading_system_tb.vcd &

.PHONY: wave-cpp
wave-cpp:
	@echo "Opening C++ simulation waveform..."
	$(GTKWAVE) fpga_trading_system_cpp.vcd &

# Advanced simulation targets
.PHONY: stress-test
stress-test: $(SIM_DIR)
	@echo "Running stress test simulation..."
	$(IVERILOG) $(IVERILOG_FLAGS) -DSTRESS_TEST -o $(SIM_DIR)/stress_test_tb \
		$(RTL_SOURCES) $(TB_DIR)/fpga_trading_system_tb.v
	cd $(SIM_DIR) && $(VVP) stress_test_tb
	@echo "Stress test completed"

.PHONY: regression
regression: $(SIM_DIR)
	@echo "Running regression test suite..."
	@for test in market_data order_manager trading_strategy integration; do \
		echo "Running $$test test..."; \
		$(MAKE) iverilog-$$test; \
		if [ $$? -eq 0 ]; then \
			echo "✓ $$test test PASSED"; \
		else \
			echo "✗ $$test test FAILED"; \
			exit 1; \
		fi; \
	done
	@echo "All regression tests PASSED!"

# Code coverage (if supported)
.PHONY: coverage
coverage: $(SIM_DIR)
	@echo "Running code coverage analysis..."
	$(IVERILOG) $(IVERILOG_FLAGS) -DCOVERAGE -o $(SIM_DIR)/coverage_tb \
		$(RTL_SOURCES) $(TB_DIR)/fpga_trading_system_tb.v
	cd $(SIM_DIR) && $(VVP) coverage_tb
	@echo "Code coverage analysis completed"

# Synthesis check (basic)
.PHONY: synth-check
synth-check:
	@echo "Running synthesis check..."
	@for file in $(RTL_SOURCES); do \
		echo "Checking $$file..."; \
		$(IVERILOG) -t null -Wall $$file; \
		if [ $$? -eq 0 ]; then \
			echo "✓ $$file synthesis check PASSED"; \
		else \
			echo "✗ $$file synthesis check FAILED"; \
		fi; \
	done

# Docker simulation environment
.PHONY: docker-build
docker-build:
	@echo "Building Docker simulation environment..."
	docker build -t fpga-sim .

.PHONY: docker-run
docker-run:
	@echo "Running simulation in Docker..."
	docker run --rm -v $(PWD):/workspace fpga-sim make all

# Documentation generation
.PHONY: docs
docs:
	@echo "Generating documentation..."
	@echo "=== HDL Simulation Environment ===" > SIMULATION_GUIDE.md
	@echo "" >> SIMULATION_GUIDE.md
	@echo "## Quick Start" >> SIMULATION_GUIDE.md
	@echo "" >> SIMULATION_GUIDE.md
	@echo "1. Install dependencies:" >> SIMULATION_GUIDE.md
	@echo "   - Icarus Verilog: sudo apt install iverilog gtkwave" >> SIMULATION_GUIDE.md
	@echo "   - Verilator: sudo apt install verilator" >> SIMULATION_GUIDE.md
	@echo "" >> SIMULATION_GUIDE.md
	@echo "2. Run simulations:" >> SIMULATION_GUIDE.md
	@echo "   - make iverilog    # Run Icarus Verilog tests" >> SIMULATION_GUIDE.md
	@echo "   - make verilator   # Run Verilator tests" >> SIMULATION_GUIDE.md
	@echo "   - make wave        # View waveforms" >> SIMULATION_GUIDE.md
	@echo "" >> SIMULATION_GUIDE.md
	@echo "3. Performance testing:" >> SIMULATION_GUIDE.md
	@echo "   - make benchmark   # Run performance benchmarks" >> SIMULATION_GUIDE.md
	@echo "   - make stress-test # Run stress tests" >> SIMULATION_GUIDE.md
	@echo "" >> SIMULATION_GUIDE.md
	@echo "Documentation generated in SIMULATION_GUIDE.md"

# CUDA HJB Solver targets
.PHONY: cuda-hjb
cuda-hjb: cuda-hjb-build cuda-hjb-test

.PHONY: cuda-hjb-build
cuda-hjb-build:
	@echo "Building CUDA HJB solver..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cd backends && \
		nvcc -O3 -arch=sm_70 -std=c++11 \
			hjb_solver.cu -c -o hjb_solver.o && \
		g++ -O3 -std=c++11 hjb_test.cpp hjb_solver.o \
			-I/usr/local/cuda/include \
			-L/usr/local/cuda/lib64 -lcudart \
			-o hjb_test_exe && \
		echo "[CUDA] Build successful: backends/hjb_test_exe"; \
	else \
		echo "Error: CUDA toolkit (nvcc) not found"; \
		echo "Install CUDA from https://developer.nvidia.com/cuda-toolkit"; \
		exit 1; \
	fi

.PHONY: cuda-hjb-test
cuda-hjb-test: cuda-hjb-build
	@echo "Running CUDA HJB solver test..."
	@if [ -f backends/hjb_test_exe ]; then \
		cd backends && ./hjb_test_exe; \
	else \
		echo "Error: hjb_test_exe not found"; \
		exit 1; \
	fi

.PHONY: cuda-hjb-clean
cuda-hjb-clean:
	@echo "Cleaning CUDA HJB artifacts..."
	rm -f backends/hjb_test_exe
	rm -f backends/hjb_jump_validation_exe
	rm -f backends/hjb_solver.o
	rm -f backends/*.o
	@echo "CUDA cleanup completed"

.PHONY: cuda-hjb-validate-jump
cuda-hjb-validate-jump:
	@echo "Building jump-diffusion validation test..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cd backends && \
		g++ -O3 -std=c++11 hjb_jump_validation.cpp hjb_solver.o \
			-I/usr/local/cuda/include \
			-L/usr/local/cuda/lib64 -lcudart \
			-o hjb_jump_validation_exe && \
		echo "[CUDA] Jump validation test built: backends/hjb_jump_validation_exe"; \
		./hjb_jump_validation_exe; \
	else \
		echo "Error: CUDA toolkit not found"; \
		exit 1; \
	fi

# Clean up
.PHONY: clean
clean: cuda-hjb-clean
	rm -rf $(SIM_DIR)
	rm -f *.vcd
	rm -f *.vvp
	rm -f *.out
	rm -f *.log
	rm -f obj_dir
	rm -f *.o
	rm -f market_data_sample.csv
	rm -f SIMULATION_GUIDE.md
	@echo "Cleanup completed"

# Install dependencies
.PHONY: install-deps
install-deps:
	@echo "Installing simulation dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update; \
		sudo apt-get install -y iverilog gtkwave verilator; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y iverilog gtkwave verilator; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install icarus-verilog gtkwave verilator; \
	else \
		echo "Please install iverilog, gtkwave, and verilator manually"; \
	fi
	@echo "Dependencies installation completed"

# Binance market data pipeline for HJB input generation
.PHONY: binance-capture
binance-capture:
	@echo "Capturing Binance WebSocket data..."
	python3 scripts/capture_binance_ws.py \
		--symbol btcusdt \
		--stream bookticker \
		--output data/binance_capture.ndjson \
		--max-messages 500

.PHONY: binance-to-input
binance-to-input:
	@echo "Converting Binance capture to market_input.txt..."
	python3 scripts/binance_to_market_input.py \
		--input data/binance_capture.ndjson \
		--output market_input.txt \
		--inventory 0

.PHONY: binance-refresh-input
binance-refresh-input: binance-capture binance-to-input
	@echo "market_input.txt refreshed from Binance data"

.PHONY: live-trading-dashboard
live-trading-dashboard:
	@echo "Launching live trading dashboard..."
	python3 scripts/live_trading_dashboard.py --mode live --symbol btcusdt --stream bookticker

.PHONY: replay-trading-dashboard
replay-trading-dashboard:
	@echo "Launching replay trading dashboard..."
	python3 scripts/live_trading_dashboard.py --mode replay --input data/binance_capture.ndjson

.PHONY: live-hjb-dashboard
live-hjb-dashboard:
	@echo "Launching live dashboard with HDL HJB backend..."
	python3 scripts/live_trading_dashboard.py --mode live --backend hjb --symbol btcusdt --stream bookticker

.PHONY: live-maker-dashboard
live-maker-dashboard:
	@echo "Launching live dashboard with market-maker simulation backend..."
	python3 scripts/live_trading_dashboard.py --mode live --backend maker --symbol btcusdt --stream depth --depth-update-ms 100 --enable-taker-hedge

.PHONY: live-verilog-maker-dashboard
live-verilog-maker-dashboard:
	@echo "Launching live dashboard with Verilog quote core + maker lifecycle backend..."
	python3 scripts/live_trading_dashboard.py --mode live --backend maker --symbol btcusdt --stream depth --depth-update-ms 100 --enable-taker-hedge --maker-use-verilog-quoter

.PHONY: iverilog-maker-quoter
iverilog-maker-quoter: $(SIM_DIR)
	@echo "Running Verilog market-maker quote simulation..."
	iverilog -g2012 -Wall -Winfloop -o $(SIM_DIR)/verilog_market_maker_tb \
		rtl/verilog_market_maker.v testbench/verilog_market_maker_tb.v
	@echo "10000000,9999950,10000050,0,30,1200,1300" > market_input_verilog.txt
	vvp $(SIM_DIR)/verilog_market_maker_tb
	@echo "Verilog output:" && cat strategy_verilog_output.txt

.PHONY: clean-verilog-maker
clean-verilog-maker:
	@echo "Cleaning Verilog maker artifacts..."
	rm -f $(SIM_DIR)/verilog_market_maker_tb
	rm -f market_input_verilog.txt
	rm -f strategy_verilog_output.txt
	@echo "Verilog maker cleanup completed"

.PHONY: live-trading-dashboard-b
live-trading-dashboard-b:
	@echo "Launching kdb/q+ dashboard (Version B)..."
	@if command -v $(Q_BIN) >/dev/null 2>&1; then \
		$(Q_BIN) scripts/live_trading_dashboard_b.q -symbol BTCUSDT -pollms 150 -fillprob 0.35 -window 240; \
	else \
		echo "Error: kdb+/q executable not found in PATH (expected: '$(Q_BIN)')."; \
		echo "Install kdb+ Community Edition from https://kx.com/connect-with-us/download/"; \
		echo "Or run with an explicit binary: make live-trading-dashboard-b Q_BIN=/path/to/q"; \
		echo "Alternative dashboard: make live-trading-dashboard"; \
		exit 127; \
	fi

# Help
.PHONY: help
help:
	@echo "HDL Simulation Makefile"
	@echo "======================="
	@echo ""
	@echo "Main targets:"
	@echo "  all              - Run all simulations"
	@echo "  iverilog         - Run Icarus Verilog simulations"
	@echo "  verilator        - Run Verilator simulations"
	@echo "  clean            - Clean up generated files"
	@echo ""
	@echo "Individual module tests:"
	@echo "  iverilog-market-data     - Test market data processor"
	@echo "  iverilog-order-manager   - Test order manager"
	@echo "  iverilog-trading-strategy - Test trading strategy"
	@echo "  iverilog-integration     - Test full integration"
	@echo ""
	@echo "CUDA GPU Solver:"
	@echo "  cuda-hjb         - Build and test CUDA HJB solver"
	@echo "  cuda-hjb-build   - Build CUDA HJB solver only"
	@echo "  cuda-hjb-test    - Run CUDA HJB solver test"
	@echo "  cuda-hjb-clean   - Clean CUDA artifacts"
	@echo ""
	@echo "Waveform viewing:"
	@echo "  wave             - View market data waveform"
	@echo "  wave-*           - View specific module waveform"
	@echo ""
	@echo "Performance testing:"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  stress-test      - Run stress tests"
	@echo "  regression       - Run regression test suite"
	@echo ""
	@echo "Utilities:"
	@echo "  synth-check      - Check synthesis compatibility"
	@echo "  coverage         - Run code coverage analysis"
	@echo "  docs             - Generate documentation"
	@echo "  install-deps     - Install simulation dependencies"
	@echo "  binance-capture  - Capture Binance WebSocket snapshots"
	@echo "  binance-to-input - Convert capture to market_input.txt"
	@echo "  binance-refresh-input - Capture and convert in one command"
	@echo "  live-trading-dashboard - Open live Binance trading visualization"
	@echo "  live-hjb-dashboard - Live Binance trading visualization with HDL HJB backend"
	@echo "  live-maker-dashboard - Live Binance depth feed with lifecycle/queue/fill maker simulator"
	@echo "  live-verilog-maker-dashboard - Live maker simulator with Verilog quote core"
	@echo "  iverilog-maker-quoter - Build/run Verilog quote core smoke test"
	@echo "  clean-verilog-maker - Remove Verilog maker generated files"
	@echo "  live-trading-dashboard-b - Open kdb/q+ dashboard version"
	@echo "  replay-trading-dashboard - Replay captured data in the visualization"
	@echo "  help             - Show this help"
