#include "hjb_solver.h"
#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char** argv) {
    using namespace hjb;
    
    std::cout << "=== HJB CUDA Solver Test ===" << std::endl;
    
    // Setup parameters from PDF defaults
    HJBParams params;
    params.sigma = 0.1;        // Volatility (low diffusion)
    params.mu = 0.0;           // Drift (martingale)
    params.gamma = 0.01;       // Terminal inventory penalty
    params.kappa = 0.0001;     // Running inventory penalty
    params.alpha = 1.0;        // Market impact
    params.lambda_j = 0.5;     // Jump intensity (high to see effect)
    params.mu_j = 0.0;         // Jump mean (symmetric jumps)
    params.sigma_j = 0.01;     // Jump std dev (small jumps, ±1%)
    
    params.NS = 64;            // Price grid: 64 points (larger spacing)
    params.NI = 32;            // Inventory grid: 32 points
    params.NT = 256;           // Time grid: 256 timesteps (smaller dt for stability)
    
    params.S_min = 95.0;       // Price range [95, 105]
    params.S_max = 105.0;
    params.I_min = -50.0;      // Inventory range [-50, 50]
    params.I_max = 50.0;
    params.T = 0.1;            // Terminal time = 0.1 seconds (reduced for stability)
    
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Grid: " << params.NS << " x " << params.NI << " x " << params.NT << std::endl;
    std::cout << "  Price range: [" << params.S_min << ", " << params.S_max << "]" << std::endl;
    std::cout << "  Inventory range: [" << params.I_min << ", " << params.I_max << "]" << std::endl;
    std::cout << "  Volatility σ = " << params.sigma << std::endl;
    std::cout << "  Jump intensity λ = " << params.lambda_j << std::endl;
    std::cout << std::endl;
    
    // Initialize solver
    std::cout << "Initializing solver..." << std::endl;
    HJBSolver solver(params);
    
    // Run solve
    std::cout << "Starting backward iteration..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    solver.solve();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Solve completed in " << std::fixed << std::setprecision(2) 
              << elapsed_ms << " ms" << std::endl << std::endl;
    
    // Extract quotes at several test states
    std::cout << "Sample quotes at t=0:" << std::endl;
    std::cout << std::setw(10) << "S" << std::setw(10) << "I" 
              << std::setw(12) << "Bid" << std::setw(12) << "Ask" << std::endl;
    std::cout << std::string(44, '-') << std::endl;
    
    double test_prices[] = {95.0, 100.0, 105.0};
    double test_invs[] = {-20.0, 0.0, 20.0};
    
    for (double S : test_prices) {
        for (double I : test_invs) {
            Quote q = solver.get_quotes(S, I, 0.0);
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(10) << S
                      << std::setw(10) << I
                      << std::setw(12) << q.bid_price
                      << std::setw(12) << q.ask_price << std::endl;
        }
    }
    
    std::cout << std::endl << "=== Test Complete ===" << std::endl;
    return 0;
}
