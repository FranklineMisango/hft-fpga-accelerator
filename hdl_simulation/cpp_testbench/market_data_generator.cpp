/*
 * Market Data Generator for C++ Testbench
 * Realistic market data generation for FPGA trading system testing
 * 
 * Features:
 * - Realistic price movements
 * - Volume profile simulation
 * - Multiple message types
 * - Configurable market conditions
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
#include <cmath>

class MarketDataGenerator {
private:
    std::mt19937 gen;
    std::normal_distribution<> price_change_dist;
    std::exponential_distribution<> volume_dist;
    std::uniform_real_distribution<> uniform_dist;
    
    struct Symbol {
        std::string name;
        uint32_t code;
        double price;
        double volatility;
        uint32_t avg_volume;
        uint32_t tick_size;
    };
    
    std::vector<Symbol> symbols;
    
public:
    MarketDataGenerator() : 
        gen(std::random_device{}()),
        price_change_dist(0.0, 0.01),
        volume_dist(1.0),
        uniform_dist(0.0, 1.0)
    {
        initializeSymbols();
    }
    
    void initializeSymbols() {
        symbols = {
            {"AAPL", 0x41415054, 150.0, 0.02, 1000, 1},
            {"GOOGL", 0x474f4f47, 2800.0, 0.025, 500, 1},
            {"MSFT", 0x4d534654, 300.0, 0.02, 800, 1},
            {"TSLA", 0x54534c41, 800.0, 0.04, 1200, 1},
            {"NVDA", 0x4e564441, 500.0, 0.035, 900, 1}
        };
    }
    
    struct MarketTick {
        uint32_t symbol_code;
        uint32_t price;
        uint32_t volume;
        uint32_t bid;
        uint32_t ask;
        uint8_t msg_type;
        uint64_t timestamp;
    };
    
    MarketTick generateTick(size_t symbol_idx) {
        if (symbol_idx >= symbols.size()) {
            symbol_idx = 0;
        }
        
        Symbol& sym = symbols[symbol_idx];
        
        // Generate price change
        double price_change = price_change_dist(gen) * sym.volatility;
        sym.price += price_change;
        
        // Keep price positive
        if (sym.price < 1.0) sym.price = 1.0;
        
        // Generate volume
        uint32_t volume = static_cast<uint32_t>(
            sym.avg_volume * volume_dist(gen)
        );
        
        // Generate bid/ask spread
        double spread = sym.price * 0.001; // 0.1% spread
        uint32_t bid = static_cast<uint32_t>((sym.price - spread/2) * 1000000);
        uint32_t ask = static_cast<uint32_t>((sym.price + spread/2) * 1000000);
        
        // Determine message type
        uint8_t msg_type = 0x41; // Default to Add
        double rand_val = uniform_dist(gen);
        if (rand_val < 0.7) {
            msg_type = 0x41; // Add Order
        } else if (rand_val < 0.85) {
            msg_type = 0x45; // Execute
        } else if (rand_val < 0.95) {
            msg_type = 0x58; // Cancel
        } else {
            msg_type = 0x44; // Delete
        }
        
        return {
            sym.code,
            static_cast<uint32_t>(sym.price * 1000000),
            volume,
            bid,
            ask,
            msg_type,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count())
        };
    }
    
    std::vector<MarketTick> generateBurst(size_t num_ticks) {
        std::vector<MarketTick> ticks;
        ticks.reserve(num_ticks);
        
        for (size_t i = 0; i < num_ticks; ++i) {
            size_t symbol_idx = i % symbols.size();
            ticks.push_back(generateTick(symbol_idx));
        }
        
        return ticks;
    }
    
    void saveToFile(const std::vector<MarketTick>& ticks, const std::string& filename) {
        std::ofstream file(filename);
        
        file << "timestamp,symbol_code,price,volume,bid,ask,msg_type" << std::endl;
        
        for (const auto& tick : ticks) {
            file << tick.timestamp << ","
                 << "0x" << std::hex << tick.symbol_code << ","
                 << std::dec << tick.price << ","
                 << tick.volume << ","
                 << tick.bid << ","
                 << tick.ask << ","
                 << "0x" << std::hex << static_cast<int>(tick.msg_type) << std::endl;
        }
        
        std::cout << "Saved " << ticks.size() << " market ticks to " << filename << std::endl;
    }
    
    void printTick(const MarketTick& tick) {
        std::cout << "Tick: Symbol=0x" << std::hex << tick.symbol_code
                  << ", Price=" << std::dec << tick.price
                  << ", Volume=" << tick.volume
                  << ", Type=0x" << std::hex << static_cast<int>(tick.msg_type)
                  << std::endl;
    }
};

// Standalone market data generator utility
int main() {
    MarketDataGenerator generator;
    
    std::cout << "=== Market Data Generator ===" << std::endl;
    
    // Generate sample data
    auto ticks = generator.generateBurst(10000);
    generator.saveToFile(ticks, "market_data_sample.csv");
    
    // Print first few ticks
    std::cout << "\nFirst 10 ticks:" << std::endl;
    for (size_t i = 0; i < std::min(10UL, ticks.size()); ++i) {
        generator.printTick(ticks[i]);
    }
    
    return 0;
}
