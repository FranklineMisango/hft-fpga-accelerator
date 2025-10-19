#include "Vhjb_calculator.h"
#include "verilated.h"
#include <cmath>

extern "C" {
    struct HJBResult {
        double bid;
        double ask;
        uint32_t latency_ns;
    };

    static Vhjb_calculator* hjb_module = nullptr;
    static uint64_t main_time = 0;

    int hjb_init() {
        if (hjb_module) return 0;
        
        Verilated::commandArgs(0, (const char**)nullptr);
        hjb_module = new Vhjb_calculator;
        
        // Reset
        hjb_module->rst_n = 0;
        hjb_module->clk = 0;
        hjb_module->calculate_en = 0;
        hjb_module->eval();
        
        // Clock cycles for reset
        for (int i = 0; i < 5; i++) {
            hjb_module->clk = !hjb_module->clk;
            hjb_module->eval();
            main_time++;
        }
        
        hjb_module->rst_n = 1;
        hjb_module->eval();
        
        return 0;
    }

    int hjb_calculate(double mid_price, int32_t inventory, double volatility, HJBResult* result) {
        if (!hjb_module) return -1;
        
        uint64_t start_time = main_time;
        
        // Convert double to 64-bit representation for Verilog
        union { double d; uint64_t i; } mid_conv = {mid_price};
        union { double d; uint64_t i; } vol_conv = {volatility};
        
        hjb_module->mid_price = mid_conv.i;
        hjb_module->inventory = inventory;
        hjb_module->volatility = vol_conv.i;
        hjb_module->calculate_en = 1;
        hjb_module->eval();
        
        // Clock until done
        while (!hjb_module->calculation_done && (main_time - start_time) < 1000) {
            hjb_module->clk = !hjb_module->clk;
            hjb_module->eval();
            main_time++;
        }
        
        if (hjb_module->calculation_done) {
            // Convert results back to double
            union { uint64_t i; double d; } bid_conv = {hjb_module->optimal_bid};
            union { uint64_t i; double d; } ask_conv = {hjb_module->optimal_ask};
            
            result->bid = bid_conv.d;
            result->ask = ask_conv.d;
            result->latency_ns = hjb_module->latency_cycles * 4; // 4ns per cycle @ 250MHz
            
            hjb_module->calculate_en = 0;
            hjb_module->eval();
            return 0;
        }
        
        return -1; // Timeout
    }

    void hjb_cleanup() {
        if (hjb_module) {
            delete hjb_module;
            hjb_module = nullptr;
        }
    }
}