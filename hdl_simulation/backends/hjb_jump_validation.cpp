#include "hjb_solver.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char** argv) {
    using namespace hjb;
    
    std::cout << "=== Jump-Diffusion Impact Test ===" << std::endl << std::endl;
    
    // Test 1: No jumps (λ=0)
    std::cout << "[Test 1] Pure diffusion (λ=0, σ=0.1)" << std::endl;
    HJBParams params_no_jump;
    params_no_jump.sigma = 0.1;
    params_no_jump.mu = 0.0;
    params_no_jump.gamma = 0.01;
    params_no_jump.kappa = 0.0001;
    params_no_jump.alpha = 1.0;
    params_no_jump.lambda_j = 0.0;    // NO JUMPS
    params_no_jump.mu_j = 0.0;
    params_no_jump.sigma_j = 0.01;
    params_no_jump.NS = 64;
    params_no_jump.NI = 32;
    params_no_jump.NT = 256;
    params_no_jump.S_min = 95.0;
    params_no_jump.S_max = 105.0;
    params_no_jump.I_min = -50.0;
    params_no_jump.I_max = 50.0;
    params_no_jump.T = 0.1;
    
    HJBSolver solver_no_jump(params_no_jump);
    solver_no_jump.solve();
    Quote q1_nojump = solver_no_jump.get_quotes(100.0, 0.0, 0.0);
    double spread_no_jump = q1_nojump.ask_price - q1_nojump.bid_price;
    
    std::cout << "  S=100, I=0: bid=" << std::fixed << std::setprecision(4) 
              << q1_nojump.bid_price << ", ask=" << q1_nojump.ask_price 
              << ", spread=" << spread_no_jump << std::endl << std::endl;
    
    // Test 2: With jumps (λ=0.5)
    std::cout << "[Test 2] Jump-diffusion (λ=0.5, σ=0.1)" << std::endl;
    HJBParams params_with_jump;
    params_with_jump.sigma = 0.1;
    params_with_jump.mu = 0.0;
    params_with_jump.gamma = 0.01;
    params_with_jump.kappa = 0.0001;
    params_with_jump.alpha = 1.0;
    params_with_jump.lambda_j = 0.5;   // HIGH JUMP INTENSITY
    params_with_jump.mu_j = 0.0;
    params_with_jump.sigma_j = 0.01;   // ±1% jump size
    params_with_jump.NS = 64;
    params_with_jump.NI = 32;
    params_with_jump.NT = 256;
    params_with_jump.S_min = 95.0;
    params_with_jump.S_max = 105.0;
    params_with_jump.I_min = -50.0;
    params_with_jump.I_max = 50.0;
    params_with_jump.T = 0.1;
    
    HJBSolver solver_with_jump(params_with_jump);
    solver_with_jump.solve();
    Quote q2_jump = solver_with_jump.get_quotes(100.0, 0.0, 0.0);
    double spread_with_jump = q2_jump.ask_price - q2_jump.bid_price;
    
    std::cout << "  S=100, I=0: bid=" << std::fixed << std::setprecision(4) 
              << q2_jump.bid_price << ", ask=" << q2_jump.ask_price 
              << ", spread=" << spread_with_jump << std::endl << std::endl;
    
    // Compare
    double spread_increase = ((spread_with_jump - spread_no_jump) / spread_no_jump) * 100.0;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "  Spread without jumps: " << std::fixed << std::setprecision(4) << spread_no_jump << std::endl;
    std::cout << "  Spread with jumps:    " << spread_with_jump << std::endl;
    std::cout << "  Increase due to jump-diffusion: " << std::fixed << std::setprecision(1) 
              << spread_increase << "%" << std::endl << std::endl;
    
    if (spread_with_jump > spread_no_jump) {
        std::cout << "✓ PASS: Jump term increases spreads (correct physics)" << std::endl;
    } else {
        std::cout << "✗ FAIL: Jump term should increase spreads" << std::endl;
    }
    
    // Test 3: Verify Gauss-Hermite quadrature activation
    std::cout << std::endl << "[Test 3] Inventory effect with jumps" << std::endl;
    Quote q_long = solver_with_jump.get_quotes(100.0, 20.0, 0.0);
    Quote q_short = solver_with_jump.get_quotes(100.0, -20.0, 0.0);
    std::cout << "  Long inventory (I=20):  bid=" << std::fixed << std::setprecision(4)
              << q_long.bid_price << ", ask=" << q_long.ask_price << std::endl;
    std::cout << "  Short inventory (I=-20): bid=" << q_short.bid_price 
              << ", ask=" << q_short.ask_price << std::endl;
    
    std::cout << std::endl << "=== Test Complete ===" << std::endl;
    return 0;
}
