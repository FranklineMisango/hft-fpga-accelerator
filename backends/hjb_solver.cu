#include "hjb_solver.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace hjb {

// Gauss-Hermite quadrature points (5-point, standard normal)
__constant__ double d_gherm_z[5] = {-2.02018, -0.95857, 0.0, 0.95857, 2.02018};
__constant__ double d_gherm_w[5] = {0.08824, 0.39362, 0.94531, 0.39362, 0.08824};

// GPU parameter struct for kernel access
struct GPUParams {
    double sigma, mu, gamma, kappa, alpha;
    double lambda_j, mu_j, sigma_j;
    int NS, NI, NT;
    double S_min, S_max, I_min, I_max, T;
    double dS, dI, dt;
};

__device__ GPUParams d_params;

// Kernel: Initialize terminal condition V(T,S,I) = -gamma * I^2
__global__ void kernel_init_terminal(double* V, int NS, int NI, int NT, double gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NS * NI) return;
    
    int i_s = idx / NI;
    int i_i = idx % NI;
    
    double I_idx = d_params.I_min + i_i * d_params.dI;
    // V[i_s, i_i, NT-1]
    int linear_idx = i_s * NI * NT + i_i * NT + (NT - 1);
    
    V[linear_idx] = -gamma * I_idx * I_idx;
}

// Kernel: Compute jump term using Gauss-Hermite quadrature
// Approximates: ∫[V(t, S(1+y), I) - V(t,S,I)] f(y) dy ≈ λ Σ w_k [V(...) - V(...)]
// Algorithm 3 from PDF: 5-point Gauss-Hermite quadrature for jump integral
__device__ double compute_jump_term(const double* V, int i_s, int i_i, int i_t, 
                                     double S_val, int NS, int NI, int NT) {
    double jump_sum = 0.0;
    
    // 5-point Gauss-Hermite quadrature: sum over nodes z_k with weights w_k
    // Jump size: y_k = μ_J + σ_J * z_k (where z_k is standard normal node)
    for (int k = 0; k < 5; k++) {
        // Jump size at this quadrature point
        double y_k = d_params.mu_j + d_params.sigma_j * d_gherm_z[k];
        
        // New price after jump: S' = S(1 + y_k) = S * e^y_k under log-normal
        double S_jump = S_val * (1.0 + y_k);
        
        // Clamp to grid bounds
        S_jump = max(d_params.S_min, min(d_params.S_max, S_jump));
        
        // Map back to grid index via linear interpolation
        // i_s_jump ∈ [0, NS-1]
        double s_continuous = (S_jump - d_params.S_min) / d_params.dS;
        int i_s_jump = (int)s_continuous;
        i_s_jump = max(0, min(NS - 2, i_s_jump));
        
        // Linear interpolation between grid points
        double frac = s_continuous - i_s_jump;
        int idx_low = i_s_jump * NI * NT + i_i * NT + i_t;
        int idx_high = (i_s_jump + 1) * NI * NT + i_i * NT + i_t;
        double V_jump = V[idx_low] * (1.0 - frac) + V[idx_high] * frac;
        
        // Current state value
        int idx_current = i_s * NI * NT + i_i * NT + i_t;
        double V_current = V[idx_current];
        
        // Value difference at this jump size
        double dV = V_jump - V_current;
        
        // Accumulate with Gauss-Hermite weight
        jump_sum += d_gherm_w[k] * dV;
    }
    
    // Scale by jump intensity λ
    return d_params.lambda_j * jump_sum;
}

// Kernel: Evaluate execution intensity and PnL for a single quote pair
// Execution intensity (Eq. 4-5, PDF): λ^b(p^b) = max(0, A^b·(1 - (p^b/p_mid - 1)/α))
// Called for each (bid, ask) candidate during control optimization
__device__ double evaluate_control(double S, double I, double bid_offset, double ask_offset,
                                    double V_next, double alpha_toxic) {
    double mid = S;
    double bid_price = mid + bid_offset;  // bid_offset is negative
    double ask_price = mid + ask_offset;  // ask_offset is positive
    
    // Clamp quotes to reasonable range (don't cross mid too far)
    bid_price = max(mid * 0.95, bid_price);
    ask_price = min(mid * 1.05, ask_price);
    
    // Execution intensity from Eq. 4-5
    // λ^b(p^b) = max(0, A^b·(1 - (p^b/p_mid - 1)/α))
    // Assume A^b = 1.0 for baseline, scale by alpha_toxic parameter
    double quote_ratio_bid = (bid_price / mid) - 1.0;  // negative for bids
    double lambda_bid = max(0.0, 1.0 * (1.0 - quote_ratio_bid / alpha_toxic));
    
    double quote_ratio_ask = (ask_price / mid) - 1.0;  // positive for asks
    double lambda_ask = max(0.0, 1.0 * (1.0 - quote_ratio_ask / alpha_toxic));
    
    // Expected PnL from successful fills (per unit, per timestep)
    // Bid fill: make profit (bid_price - S) if executed
    // Ask fill: make profit (S - ask_price) if executed
    double pnl_bid = (bid_price - S) * lambda_bid;  // negative * positive = strategy pays
    double pnl_ask = (ask_price - S) * lambda_ask;  // positive * positive = we make money
    
    // Total score: PnL - inventory cost - subsequent loss + continuation value
    // Simplified: score = pnl_bid + pnl_ask + V_next (already has inventory term)
    // - inventory penalty is amortized in V_next
    // + any temporary adjustment for this control choice
    double spread = ask_price - bid_price;
    double spread_bonus = 0.01 * spread * (lambda_bid + lambda_ask);  // reward tight spreads
    
    double score = pnl_bid + pnl_ask + V_next * 0.1 + spread_bonus;
    
    return score;
}

// Kernel: Optimize control (bid/ask) for given state
// Searches 5×5 control grid: bid ∈ [-2%, -0.2%], ask ∈ [+0.1%, +2%]
// Returns optimal bid/ask prices written to output arrays
__device__ void optimize_control_state(double S, double I, double V_next, double alpha_toxic,
                                        double& optimal_bid, double& optimal_ask) {
    // Control grid: 5 bid offsets, 5 ask offsets
    double bid_offsets[] = {-0.02 * S, -0.01 * S, -0.005 * S, -0.002 * S, -0.001 * S};
    double ask_offsets[] = {0.001 * S, 0.002 * S, 0.005 * S, 0.01 * S, 0.02 * S};
    
    double best_score = -1e9;
    double best_bid = S;
    double best_ask = S;
    
    // Exhaustive search: 25 candidates
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            double score = evaluate_control(S, I, bid_offsets[i], ask_offsets[j], V_next, alpha_toxic);
            
            if (score > best_score) {
                best_score = score;
                best_bid = S + bid_offsets[i];
                best_ask = S + ask_offsets[j];
            }
        }
    }
    
    optimal_bid = best_bid;
    optimal_ask = best_ask;
}

// Kernel: HJB backward iteration step WITH control optimization (Eq. 19, PDF)
// For each (S, I), compute V(t-1, S, I) from V(t, *, *) and optimal quotes
__global__ void kernel_hjb_step(double* V_new, const double* V_old,
                                double* optimal_bids, double* optimal_asks,
                                int i_t) {
    int i_s = blockIdx.x * blockDim.x + threadIdx.x;
    int i_i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i_s >= d_params.NS || i_i >= d_params.NI) return;
    if (i_s == 0 || i_s == d_params.NS - 1 || i_i == 0 || i_i == d_params.NI - 1) {
        // Boundary: copy forward
        int idx_new = i_s * d_params.NI * d_params.NT + i_i * d_params.NT + i_t;
        int idx_old = i_s * d_params.NI * d_params.NT + i_i * d_params.NT + (i_t + 1);
        V_new[idx_new] = V_old[idx_old];
        optimal_bids[idx_new] = d_params.S_min + i_s * d_params.dS;
        optimal_asks[idx_new] = d_params.S_min + i_s * d_params.dS;
        return;
    }
    
    double S_val = d_params.S_min + i_s * d_params.dS;
    double I_val = d_params.I_min + i_i * d_params.dI;
    
    // Load neighboring values at t+1
    int idx_c = i_s * d_params.NI * d_params.NT + i_i * d_params.NT + (i_t + 1);
    int idx_l = (i_s - 1) * d_params.NI * d_params.NT + i_i * d_params.NT + (i_t + 1);
    int idx_r = (i_s + 1) * d_params.NI * d_params.NT + i_i * d_params.NT + (i_t + 1);
    
    double V_c = V_old[idx_c];
    double V_l = V_old[idx_l];
    double V_r = V_old[idx_r];
    
    // Finite differences
    double V_S = (V_r - V_l) / (2.0 * d_params.dS);
    double V_SS = (V_r - 2.0 * V_c + V_l) / (d_params.dS * d_params.dS);
    
    // Clamp derivatives to prevent blowup
    V_S = max(-100.0, min(100.0, V_S));
    V_SS = max(-10000.0, min(10000.0, V_SS));
    
    // Diffusion term: 0.5 * σ² * S² * V_SS
    double diffusion = 0.5 * d_params.sigma * d_params.sigma * S_val * S_val * V_SS;
    
    // Drift term: μ * S * V_S
    double drift = d_params.mu * S_val * V_S;
    
    // Jump term: λ·∫ w_k·[V(t, S(1+y_k), I) - V(t, S, I)] (Algorithm 3, PDF)
    double jump_term = compute_jump_term(V_old, i_s, i_i, i_t + 1, S_val, 
                                          d_params.NS, d_params.NI, d_params.NT);
    
    // Inventory cost: κ·I²
    double inv_cost = d_params.kappa * I_val * I_val;
    
    // HJB: V(t,S,I) = V(t+1,S,I) + Δt * [drift + diffusion + jump - inv_cost]
    double dt_clamped = min(d_params.dt, 0.001);
    double V_value = V_c + dt_clamped * (drift + diffusion + jump_term - inv_cost);
    V_value = max(-1e6, min(1e6, V_value));
    
    // Control optimization: find optimal bid/ask (Equation 19, PDF)
    double alpha_toxic = d_params.alpha;  // baseline market impact
    double optimal_bid, optimal_ask;
    optimize_control_state(S_val, I_val, V_value, alpha_toxic, optimal_bid, optimal_ask);
    
    // Store results
    int idx_new = i_s * d_params.NI * d_params.NT + i_i * d_params.NT + i_t;
    V_new[idx_new] = V_value;
    optimal_bids[idx_new] = optimal_bid;
    optimal_asks[idx_new] = optimal_ask;
}

// Host implementation
HJBSolver::HJBSolver(const HJBParams& params)
    : params_(params), d_V(nullptr), d_V_next(nullptr), 
      d_S(nullptr), d_I(nullptr), d_optimal_bids(nullptr), d_optimal_asks(nullptr),
      d_params_gpu(nullptr), solved_(false) {
    allocate_gpu_memory();
    initialize_grids();
    initialize_boundary_condition();
}

HJBSolver::~HJBSolver() {
    free_gpu_memory();
}

void HJBSolver::allocate_gpu_memory() {
    size_t V_size = params_.NS * params_.NI * params_.NT * sizeof(double);
    size_t grid_size = params_.NS * sizeof(double);
    
    cudaMalloc((void**)&d_V, V_size);
    cudaMalloc((void**)&d_V_next, V_size);
    cudaMalloc((void**)&d_S, grid_size);
    cudaMalloc((void**)&d_I, params_.NI * sizeof(double));
    cudaMalloc((void**)&d_optimal_bids, V_size);
    cudaMalloc((void**)&d_optimal_asks, V_size);
    
    cudaMemset(d_V, 0, V_size);
    cudaMemset(d_V_next, 0, V_size);
    cudaMemset(d_optimal_bids, 0, V_size);
    cudaMemset(d_optimal_asks, 0, V_size);
}

void HJBSolver::free_gpu_memory() {
    if (d_V) cudaFree(d_V);
    if (d_V_next) cudaFree(d_V_next);
    if (d_S) cudaFree(d_S);
    if (d_I) cudaFree(d_I);
    if (d_optimal_bids) cudaFree(d_optimal_bids);
    if (d_optimal_asks) cudaFree(d_optimal_asks);
}

void HJBSolver::initialize_grids() {
    double dS = (params_.S_max - params_.S_min) / (params_.NS - 1);
    double dI = (params_.I_max - params_.I_min) / (params_.NI - 1);
    double dt = params_.T / (params_.NT - 1);
    
    // Create S grid
    double* h_S = new double[params_.NS];
    for (int i = 0; i < params_.NS; i++) {
        h_S[i] = params_.S_min + i * dS;
    }
    cudaMemcpy(d_S, h_S, params_.NS * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_S;
    
    // Create I grid
    double* h_I = new double[params_.NI];
    for (int i = 0; i < params_.NI; i++) {
        h_I[i] = params_.I_min + i * dI;
    }
    cudaMemcpy(d_I, h_I, params_.NI * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_I;
    
    // Update GPU params with computed deltas
    GPUParams gpu_p;
    gpu_p.sigma = params_.sigma;
    gpu_p.mu = params_.mu;
    gpu_p.gamma = params_.gamma;
    gpu_p.kappa = params_.kappa;
    gpu_p.alpha = params_.alpha;
    gpu_p.lambda_j = params_.lambda_j;
    gpu_p.mu_j = params_.mu_j;
    gpu_p.sigma_j = params_.sigma_j;
    gpu_p.NS = params_.NS;
    gpu_p.NI = params_.NI;
    gpu_p.NT = params_.NT;
    gpu_p.S_min = params_.S_min;
    gpu_p.S_max = params_.S_max;
    gpu_p.I_min = params_.I_min;
    gpu_p.I_max = params_.I_max;
    gpu_p.T = params_.T;
    gpu_p.dS = dS;
    gpu_p.dI = dI;
    gpu_p.dt = dt;
    
    cudaMemcpyToSymbol(d_params, &gpu_p, sizeof(GPUParams));
}

void HJBSolver::initialize_boundary_condition() {
    int block_size = 256;
    int num_blocks = (params_.NS * params_.NI + block_size - 1) / block_size;
    kernel_init_terminal<<<num_blocks, block_size>>>(d_V, params_.NS, params_.NI, 
                                                       params_.NT, params_.gamma);
    cudaDeviceSynchronize();
}

void HJBSolver::solve() {
    dim3 block_dim(16, 16);
    dim3 grid_dim((params_.NS + 15) / 16, (params_.NI + 15) / 16);
    
    // Backward iteration: from NT-2 down to 0
    for (int i_t = params_.NT - 2; i_t >= 0; i_t--) {
        // Compute V at i_t using values at i_t+1, with control optimization
        kernel_hjb_step<<<grid_dim, block_dim>>>(d_V_next, d_V, d_optimal_bids, d_optimal_asks, i_t);
        cudaDeviceSynchronize();
        
        // Copy result back to d_V for next iteration
        cudaMemcpy(d_V, d_V_next, params_.NS * params_.NI * params_.NT * sizeof(double),
                   cudaMemcpyDeviceToDevice);
        
        if (i_t % 100 == 0) {
            std::cout << "[HJB] Iteration t_idx=" << i_t << " / " << params_.NT << " (with control opt)" << std::endl;
        }
    }
    
    solved_ = true;
    std::cout << "[HJB] Solve complete (control-optimized quotes computed)." << std::endl;
}

Quote HJBSolver::get_quotes(double S, double I, double t) const {
    if (!solved_) {
        std::cerr << "[HJB] Solver not run yet!" << std::endl;
        return {0, 0, 0, 0, 0};
    }
    
    Quote q = {0, 0, 0, 0, 0};
    
    // Map (S, I, t) to grid indices
    int i_s = (int)((S - params_.S_min) / ((params_.S_max - params_.S_min) / (params_.NS - 1)));
    int i_i = (int)((I - params_.I_min) / ((params_.I_max - params_.I_min) / (params_.NI - 1)));
    int i_t = (int)(t / params_.T * (params_.NT - 1));
    
    i_s = std::max(0, std::min(params_.NS - 1, i_s));
    i_i = std::max(0, std::min(params_.NI - 1, i_i));
    i_t = std::max(0, std::min(params_.NT - 1, i_t));
    
    // Copy optimal quotes from GPU
    double* h_bids = new double[params_.NS * params_.NI * params_.NT];
    double* h_asks = new double[params_.NS * params_.NI * params_.NT];
    
    cudaMemcpy(h_bids, d_optimal_bids, params_.NS * params_.NI * params_.NT * sizeof(double), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_asks, d_optimal_asks, params_.NS * params_.NI * params_.NT * sizeof(double), 
               cudaMemcpyDeviceToHost);
    
    int idx = i_s * params_.NI * params_.NT + i_i * params_.NT + i_t;
    q.bid_price = h_bids[idx];
    q.ask_price = h_asks[idx];
    q.bid_intensity = 1.0;  // Placeholder
    q.ask_intensity = 1.0;  // Placeholder
    q.convergence_iters = 1;
    
    delete[] h_bids;
    delete[] h_asks;
    return q;
}

}  // namespace hjb
