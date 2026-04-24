#ifndef HJB_SOLVER_H
#define HJB_SOLVER_H

#include <cstdint>
#include <cstring>

namespace hjb {

// Grid and solver parameters (matching PDF notation)
struct HJBParams {
    double sigma;        // Volatility
    double mu;          // Drift
    double gamma;       // Terminal inventory penalty coefficient
    double kappa;       // Running inventory penalty
    double alpha;       // Market impact baseline
    double lambda_j;    // Jump intensity
    double mu_j;        // Jump mean
    double sigma_j;     // Jump std dev
    
    int NS;             // Price grid size
    int NI;             // Inventory grid size  
    int NT;             // Time grid size
    
    double S_min;       // Min price
    double S_max;       // Max price
    double I_min;       // Min inventory
    double I_max;       // Max inventory
    double T;           // Terminal time
};

// Result: optimal quotes at current state
struct Quote {
    double bid_price;
    double ask_price;
    double bid_intensity;
    double ask_intensity;
    int convergence_iters;
};

class HJBSolver {
public:
    HJBSolver(const HJBParams& params);
    ~HJBSolver();
    
    // Solve backward from t=T to t=0, store full V grid
    void solve();
    
    // Extract optimal quotes at given state (S, I, t)
    Quote get_quotes(double S, double I, double t = 0.0) const;
    
    // Access raw value function (for debugging)
    double* get_value_function() const { return d_V; }
    
    // Access optimal quotes (if computed)
    double* get_optimal_bids() const { return d_optimal_bids; }
    double* get_optimal_asks() const { return d_optimal_asks; }
    
    // Grid accessors
    const HJBParams& get_params() const { return params_; }
    double* get_S_grid() const { return d_S; }
    double* get_I_grid() const { return d_I; }
    
private:
    HJBParams params_;
    
    // GPU memory
    double* d_V;              // Value function [NS * NI * NT]
    double* d_V_next;        // Temp buffer for next iteration
    double* d_S;              // Price grid [NS]
    double* d_I;              // Inventory grid [NI]
    double* d_optimal_bids;   // Optimal bid quotes [NS * NI * NT]
    double* d_optimal_asks;   // Optimal ask quotes [NS * NI * NT]
    double* d_params_gpu;     // Struct copy on device
    
    // Grind state
    bool solved_;
    
    // Helpers
    void allocate_gpu_memory();
    void free_gpu_memory();
    void initialize_grids();
    void initialize_boundary_condition();
    int index_3d(int i_s, int i_i, int i_t) const;
};

}  // namespace hjb

#endif // HJB_SOLVER_H
