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
              $(RTL_DIR)/trading_strategy.v

# Testbench sources
TB_SOURCES = $(TB_DIR)/market_data_tb.v \
             $(TB_DIR)/order_manager_tb.v \
             $(TB_DIR)/trading_strategy_tb.v \
             $(TB_DIR)/fpga_trading_system_tb.v

# Simulation tools
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave
VERILATOR = verilator

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

# Clean up
.PHONY: clean
clean:
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
	@echo "  help             - Show this help"
