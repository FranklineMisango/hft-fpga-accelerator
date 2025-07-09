# FPGA Trading Accelerator  
*Hardware-accelerated low-latency trading infrastructure*  

## Overview  
Open-source Verilog/VHDL components for building FPGA-based trading systems. Implements market data parsing, order book management, and strategy execution with sub-microsecond latency. Designed for Xilinx/Intel FPGAs with PCIe and 10G/100G Ethernet support.  

## Key Features  
### 1. Market Data Processing  
- **Protocol Decoders**: Hardware parsers for FIX/ITCH/OUCH protocols.  
- **Tick-to-Order Book**: Real-time aggregation of price/volume data.  

### 2. Order Management  
- **Order State Machine**: Cancel/replace logic with 16ns latency (simulated).  
- **Risk Controls**: Pre-trade checks (e.g., position limits) in hardware.  

### 3. Strategy Templates  
- **Arbitrage**: Cross-exchange latency optimizations.  
- **TWAP/VWAP**: Time-weighted execution with RTL reference designs.  

---  
## Hardware Requirements  
| Component          | Recommended Specs               |  
|--------------------|----------------------------------|  
| FPGA Board         | Xilinx Alveo U50 / Intel Stratix 10 MX |  
| Networking         | Solarflare X2522 NIC (10G/100G)  |  
| Host CPU           | Linux kernel ≥5.4 (with PREEMPT_RT) |  

---  
## Getting Started  
### Simulation (No Hardware Needed)  
1. Install Icarus Verilog:  
   ```bash  
   sudo apt install iverilog gtkwave  

2. Run testbench:
    ```bash
    cd sim/itch_parser && make simulate

3. Synthesis(Incase of a Physical FPGA):
    ```bash
    vivado -mode batch -source scripts/synth_zynq.tcl  

### Python FPGA Simulation
For comprehensive trading system simulation without hardware:

1. **Quick Start**:
   ```bash
   cd fpga_simulation
   python3 simple_example.py
   ```

2. **Full Simulation Setup**:
   ```bash
   cd fpga_simulation
   make setup
   make simulate
   ```

3. **Performance Testing**:
   ```bash
   cd fpga_simulation
   make benchmark
   ```

#### Simulation Features
- **Hardware-accurate timing**: Sub-microsecond latency modeling
- **Multiple trading strategies**: Arbitrage, market making, TWAP/VWAP
- **Real-time market data**: Configurable tick rates and protocols
- **Performance analysis**: Latency statistics and throughput metrics
- **Risk management**: Position limits and pre-trade checks

#### Example Results
```
=== Simulation Results ===
Total trades: 15,420
Average latency: 45.2 ns
P99 latency: 128.5 ns
Throughput: 1.2M orders/sec
```

### HDL Simulation Environment
For hardware-level simulation with virtual FPGA simulators:

1. **Install Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt install iverilog gtkwave verilator
   
   # macOS
   brew install icarus-verilog gtkwave verilator
   ```

2. **Run HDL Simulations**:
   ```bash
   cd hdl_simulation
   make all                    # Run all tests
   make iverilog              # Icarus Verilog tests
   make verilator             # Verilator tests
   ```

3. **Automated Testing**:
   ```bash
   cd hdl_simulation
   ./run_simulation.py --simulator both
   ```

4. **Waveform Analysis**:
   ```bash
   make wave                  # View with GTKWave
   ```

#### HDL Simulation Features
- **Open-source simulators**: Icarus Verilog and Verilator support
- **Comprehensive testbenches**: Market data, order management, strategies
- **C++ co-simulation**: High-performance Verilator integration
- **Performance benchmarking**: Latency and throughput analysis
- **Automated testing**: Regression suite and CI/CD ready
- **Docker environment**: Reproducible simulation setup

#### HDL Test Coverage
- ✅ Market data processing (ITCH/FIX protocols)
- ✅ Order management and execution
- ✅ Trading strategy engines
- ✅ Risk management systems
- ✅ End-to-end integration testing
- ✅ Performance characterization

#### HDL Performance Results
```
=== HDL Simulation Results ===
Market Data Latency: 16-32 ns
Order Processing: 32-64 ns
End-to-End Latency: 64-128 ns
Throughput: 1M+ orders/sec
Clock Frequency: 250 MHz
```

## 🚀 Advanced Tools & Features

### Real-time Monitoring & Visualization
- **Interactive Dashboard**: Streamlit-based real-time monitoring
- **Web Interface**: Flask-based monitoring with WebSocket support
- **Performance Profiler**: Comprehensive performance analysis
- **GPU-FPGA Bridge**: Hybrid acceleration pipeline

### Advanced Market Data
- **Protocol Support**: NASDAQ ITCH 5.0, FIX 4.4, OUCH 4.2
- **Realistic Timing**: Hardware-accurate latency simulation
- **Burst Testing**: High-frequency stress testing
- **Statistical Analysis**: Comprehensive performance metrics

### Usage Examples

#### Launch Real-time Dashboard:
```bash
cd fpga_simulation
make dashboard
# Opens at http://localhost:8501
```

#### Start Web Monitor:
```bash
cd fpga_simulation
make web-monitor
# Opens at http://localhost:5000
```

#### Run Performance Profiler:
```bash
cd fpga_simulation
make profiler
# Generates comprehensive performance reports
```

#### Test GPU-FPGA Bridge:
```bash
cd fpga_simulation
make gpu-bridge
# Tests hybrid acceleration pipeline
```

### Advanced Features
- **GPU Acceleration**: CuPy/CUDA integration for batch processing
- **Protocol Simulation**: Binary protocol parsing and generation
- **Performance Analytics**: Statistical analysis and reporting
- **Load Testing**: Configurable stress testing scenarios
- **Real-time Alerts**: Performance threshold monitoring

For detailed documentation on advanced features, see [Advanced Tools Guide](fpga_simulation/ADVANCED_TOOLS_GUIDE.md).

---
## Contributing
1. Report Issues: Use GitHub Issues for bugs/feature requests.
2. Development Flow:
    * Branch naming: feat/itch-parser or fix/orderbook-bug.
    * PRs require:
        * Passing testbench simulations.
        * Synthesis reports for target FPGAs.

## My Learning Pathways 
[TechnicalLab by Nvidia](https://developer.nvidia.com/blog/gpu-accelerate-algorithmic-trading-simulations-by-over-100x-with-numba/)
[QFin by Nvidia](https://developer.nvidia.com/blog/introduction-to-gpu-accelerated-python-for-financial-services/)
[gQuants](https://medium.com/rapids-ai/gquant-gpu-accelerated-examples-for-quantitative-analyst-tasks-8b6de44c0ac2)
