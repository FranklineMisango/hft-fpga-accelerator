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
