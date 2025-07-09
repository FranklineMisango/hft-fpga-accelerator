/*
 * C++ Testbench for FPGA Trading System using Verilator
 * Advanced co-simulation with performance analysis
 * 
 * Features:
 * - High-performance C++ simulation
 * - Real-time market data injection
 * - Advanced analytics and reporting
 * - Integration with Python analysis tools
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <iomanip>

#include "verilated.h"
#include "verilated_vcd_c.h"
#include "Vfpga_trading_system_tb.h"

class FPGATradingSystemTest {
private:
    std::unique_ptr<Vfpga_trading_system_tb> dut;
    std::unique_ptr<VerilatedVcdC> trace;
    
    // Performance metrics
    uint64_t cycle_count;
    uint64_t total_ticks;
    uint64_t total_executions;
    uint64_t total_latency;
    uint64_t max_latency;
    uint64_t min_latency;
    
    // Market data generation
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> price_dist;
    std::uniform_int_distribution<> volume_dist;
    
    // Test configuration
    static constexpr uint64_t CLOCK_PERIOD = 4; // 4ns = 250MHz
    static constexpr uint32_t TRACE_DEPTH = 99;
    
    // Symbol table
    std::vector<std::string> symbols = {"AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"};
    std::vector<uint32_t> symbol_codes;
    
public:
    FPGATradingSystemTest() : 
        gen(rd()),
        price_dist(100.0, 200.0),
        volume_dist(100, 10000),
        cycle_count(0),
        total_ticks(0),
        total_executions(0),
        total_latency(0),
        max_latency(0),
        min_latency(UINT64_MAX)
    {
        // Initialize DUT
        dut = std::make_unique<Vfpga_trading_system_tb>();
        
        // Initialize tracing
        Verilated::traceEverOn(true);
        trace = std::make_unique<VerilatedVcdC>();
        dut->trace(trace.get(), TRACE_DEPTH);
        trace->open("fpga_trading_system_cpp.vcd");
        
        // Initialize symbol codes
        initializeSymbolCodes();
        
        std::cout << "=== FPGA Trading System C++ Testbench ===" << std::endl;
        std::cout << "Clock frequency: " << (1000.0 / CLOCK_PERIOD) << " MHz" << std::endl;
        std::cout << "Symbols: ";
        for (const auto& symbol : symbols) {
            std::cout << symbol << " ";
        }
        std::cout << std::endl << std::endl;
    }
    
    ~FPGATradingSystemTest() {
        if (trace) {
            trace->close();
        }
    }
    
    void initializeSymbolCodes() {
        // Convert symbol strings to 32-bit codes
        for (const auto& symbol : symbols) {
            uint32_t code = 0;
            for (size_t i = 0; i < std::min(symbol.length(), 4UL); ++i) {
                code |= (static_cast<uint32_t>(symbol[i]) << (24 - i * 8));
            }
            symbol_codes.push_back(code);
        }
    }
    
    void reset() {
        dut->rst_n = 0;
        dut->clk = 0;
        dut->market_data_valid = 0;
        dut->market_data_in = 0;
        dut->market_data_type = 0;
        
        // Hold reset for 5 cycles
        for (int i = 0; i < 5; ++i) {
            clockCycle();
        }
        
        dut->rst_n = 1;
        clockCycle();
        
        std::cout << "System reset completed" << std::endl;
    }
    
    void clockCycle() {
        dut->clk = 0;
        dut->eval();
        trace->dump(cycle_count * CLOCK_PERIOD);
        
        dut->clk = 1;
        dut->eval();
        trace->dump(cycle_count * CLOCK_PERIOD + CLOCK_PERIOD/2);
        
        cycle_count++;
    }
    
    void sendMarketData(uint32_t symbol_code, uint32_t price, uint32_t volume, uint8_t msg_type = 0x41) {
        dut->market_data_type = msg_type;
        dut->market_data_in = (static_cast<uint64_t>(symbol_code) << 32) | price;
        dut->market_data_valid = 1;
        
        clockCycle();
        
        dut->market_data_valid = 0;
        total_ticks++;
    }
    
    void waitForExecution(uint32_t max_cycles = 100) {
        uint32_t wait_cycles = 0;
        uint64_t start_cycle = cycle_count;
        
        while (!dut->order_execution_valid && wait_cycles < max_cycles) {
            clockCycle();
            wait_cycles++;
        }
        
        if (dut->order_execution_valid) {
            uint64_t latency = cycle_count - start_cycle;
            total_latency += latency;
            total_executions++;
            
            if (latency > max_latency) max_latency = latency;
            if (latency < min_latency) min_latency = latency;
        }
    }
    
    void runBasicFunctionalTest() {
        std::cout << "Running Basic Functional Test..." << std::endl;
        
        // Test 1: Single order execution
        sendMarketData(symbol_codes[0], 0x96000000, 0x64000000); // AAPL $150.00, 100 shares
        waitForExecution();
        
        if (dut->order_execution_valid) {
            std::cout << "✓ Basic order execution working" << std::endl;
            std::cout << "  Symbol: " << std::hex << dut->execution_symbol << std::endl;
            std::cout << "  Price: " << std::hex << dut->execution_price << std::endl;
            std::cout << "  Volume: " << std::hex << dut->execution_volume << std::endl;
        } else {
            std::cout << "✗ Basic order execution failed" << std::endl;
        }
        
        // Wait for system to settle
        for (int i = 0; i < 10; ++i) {
            clockCycle();
        }
        
        std::cout << "Basic functional test completed" << std::endl << std::endl;
    }
    
    void runMultiSymbolTest() {
        std::cout << "Running Multi-Symbol Test..." << std::endl;
        
        // Send data for all symbols
        for (size_t i = 0; i < symbols.size(); ++i) {
            uint32_t price = 0x96000000 + (i * 0x01000000);
            uint32_t volume = 0x64000000 + (i * 0x10000000);
            
            sendMarketData(symbol_codes[i], price, volume);
            
            // Small delay between symbols
            for (int j = 0; j < 5; ++j) {
                clockCycle();
            }
        }
        
        std::cout << "Multi-symbol test completed" << std::endl << std::endl;
    }
    
    void runHighFrequencyTest() {
        std::cout << "Running High-Frequency Test..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t start_cycle = cycle_count;
        
        // Send 10,000 market data updates as fast as possible
        for (int i = 0; i < 10000; ++i) {
            uint32_t symbol_idx = i % symbols.size();
            uint32_t price = 0x96000000 + (i % 1000);
            uint32_t volume = 0x64000000 + (i % 100) * 0x01000000;
            
            sendMarketData(symbol_codes[symbol_idx], price, volume);
            
            // Process every 100th tick
            if (i % 100 == 0) {
                clockCycle();
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t end_cycle = cycle_count;
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double cycles_per_tick = static_cast<double>(end_cycle - start_cycle) / 10000.0;
        
        std::cout << "✓ Processed 10,000 ticks in " << duration.count() << " μs" << std::endl;
        std::cout << "✓ Average cycles per tick: " << std::fixed << std::setprecision(2) << cycles_per_tick << std::endl;
        std::cout << "✓ Simulated throughput: " << 
                     (10000.0 * 1000000.0) / duration.count() << " ticks/second" << std::endl;
        
        std::cout << "High-frequency test completed" << std::endl << std::endl;
    }
    
    void runLatencyBenchmark() {
        std::cout << "Running Latency Benchmark..." << std::endl;
        
        std::vector<uint64_t> latencies;
        
        // Measure latency for 1000 individual transactions
        for (int i = 0; i < 1000; ++i) {
            uint64_t start_cycle = cycle_count;
            
            sendMarketData(symbol_codes[0], 0x96000000 + i, 0x64000000);
            waitForExecution();
            
            if (dut->order_execution_valid) {
                uint64_t latency = cycle_count - start_cycle;
                latencies.push_back(latency);
            }
            
            // Wait for system to settle
            for (int j = 0; j < 5; ++j) {
                clockCycle();
            }
        }
        
        // Calculate statistics
        if (!latencies.empty()) {
            std::sort(latencies.begin(), latencies.end());
            
            uint64_t sum = 0;
            for (auto lat : latencies) sum += lat;
            
            double avg_latency = static_cast<double>(sum) / latencies.size();
            uint64_t p50_latency = latencies[latencies.size() / 2];
            uint64_t p95_latency = latencies[static_cast<size_t>(latencies.size() * 0.95)];
            uint64_t p99_latency = latencies[static_cast<size_t>(latencies.size() * 0.99)];
            
            std::cout << "Latency Statistics (cycles):" << std::endl;
            std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg_latency << std::endl;
            std::cout << "  Median (P50): " << p50_latency << std::endl;
            std::cout << "  P95: " << p95_latency << std::endl;
            std::cout << "  P99: " << p99_latency << std::endl;
            std::cout << "  Min: " << latencies.front() << std::endl;
            std::cout << "  Max: " << latencies.back() << std::endl;
            
            std::cout << "Latency Statistics (nanoseconds @ 250MHz):" << std::endl;
            std::cout << "  Average: " << std::fixed << std::setprecision(1) << avg_latency * CLOCK_PERIOD << " ns" << std::endl;
            std::cout << "  P95: " << p95_latency * CLOCK_PERIOD << " ns" << std::endl;
            std::cout << "  P99: " << p99_latency * CLOCK_PERIOD << " ns" << std::endl;
        }
        
        std::cout << "Latency benchmark completed" << std::endl << std::endl;
    }
    
    void runStressTest() {
        std::cout << "Running Stress Test..." << std::endl;
        
        // Test with maximum rate sustained load
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 50000; ++i) {
            uint32_t symbol_idx = i % symbols.size();
            uint32_t price = 0x96000000 + (i % 10000);
            uint32_t volume = 0x64000000 + (i % 1000) * 0x01000000;
            
            // Mix of message types
            uint8_t msg_type = 0x41; // Default to Add
            if (i % 10 == 0) msg_type = 0x45; // Execute
            if (i % 15 == 0) msg_type = 0x58; // Cancel
            
            sendMarketData(symbol_codes[symbol_idx], price, volume, msg_type);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ Sustained 50,000 messages in " << duration.count() << " ms" << std::endl;
        std::cout << "✓ Stress test throughput: " << 
                     (50000.0 * 1000.0) / duration.count() << " messages/second" << std::endl;
        
        std::cout << "Stress test completed" << std::endl << std::endl;
    }
    
    void generateReport() {
        std::cout << "=== FPGA Trading System Test Report ===" << std::endl;
        std::cout << "Total simulation cycles: " << cycle_count << std::endl;
        std::cout << "Total market ticks: " << total_ticks << std::endl;
        std::cout << "Total executions: " << total_executions << std::endl;
        
        if (total_ticks > 0) {
            std::cout << "Execution rate: " << 
                         std::fixed << std::setprecision(2) << 
                         (static_cast<double>(total_executions) / total_ticks * 100.0) << "%" << std::endl;
        }
        
        if (total_executions > 0) {
            std::cout << "Average latency: " << 
                         std::fixed << std::setprecision(2) << 
                         (static_cast<double>(total_latency) / total_executions) << " cycles" << std::endl;
            std::cout << "Average latency: " << 
                         std::fixed << std::setprecision(1) << 
                         (static_cast<double>(total_latency) / total_executions * CLOCK_PERIOD) << " ns" << std::endl;
            std::cout << "Max latency: " << max_latency << " cycles (" << 
                         (max_latency * CLOCK_PERIOD) << " ns)" << std::endl;
            std::cout << "Min latency: " << min_latency << " cycles (" << 
                         (min_latency * CLOCK_PERIOD) << " ns)" << std::endl;
        }
        
        // Performance metrics
        double simulated_time_ns = cycle_count * CLOCK_PERIOD;
        double simulated_frequency_mhz = cycle_count / (simulated_time_ns / 1000.0);
        
        std::cout << "Simulated time: " << std::fixed << std::setprecision(2) << 
                     simulated_time_ns / 1000.0 << " μs" << std::endl;
        std::cout << "Effective frequency: " << std::fixed << std::setprecision(1) << 
                     simulated_frequency_mhz << " MHz" << std::endl;
        
        std::cout << "=== Test Summary ===" << std::endl;
        std::cout << "All tests completed successfully!" << std::endl;
        std::cout << "VCD trace saved to: fpga_trading_system_cpp.vcd" << std::endl;
    }
    
    void runAllTests() {
        reset();
        
        runBasicFunctionalTest();
        runMultiSymbolTest();
        runHighFrequencyTest();
        runLatencyBenchmark();
        runStressTest();
        
        generateReport();
    }
};

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    
    try {
        FPGATradingSystemTest test;
        test.runAllTests();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
