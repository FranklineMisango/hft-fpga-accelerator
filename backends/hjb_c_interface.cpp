#include "hjb_c_interface.h"
#include "hjb_solver.h"
#include <cstring>
#include <chrono>

using namespace hjb;

/* C++ implementation - store both params and solver in handle */
struct _HPPInternal {
  HJBParams* params;
  HJBSolver* solver;
  float solve_time_ms;
};

/* Cast void* to internal struct  */
#define HANDLE(x) (reinterpret_cast<_HPPInternal*>(x))

HJBSolverHandle* hjb_create_solver() {
  _HPPInternal* handle = new _HPPInternal();
  handle->params = nullptr;
  handle->solver = nullptr;
  handle->solve_time_ms = 0.0f;
  return (HJBSolverHandle*)handle;
}

int hjb_init(HJBSolverHandle* handle,
             float sigma, float mu, float gamma, float kappa,
             float alpha, float lambda,
             int NS, int NI, int NT,
             float S_min, float S_max, float I_min, float I_max, float T) {
  _HPPInternal* h = HANDLE(handle);
  if (!h) return 0;

  try {
    if (h->params) delete h->params;
    if (h->solver) delete h->solver;

    h->params = new HJBParams();
    h->params->sigma = sigma;
    h->params->mu = mu;
    h->params->gamma = gamma;
    h->params->kappa = kappa;
    h->params->alpha = alpha;
    h->params->lambda_j = lambda;  /* Note: C interface param is lambda */
    h->params->NS = NS;
    h->params->NI = NI;
    h->params->NT = NT;
    h->params->S_min = S_min;
    h->params->S_max = S_max;
    h->params->I_min = I_min;
    h->params->I_max = I_max;
    h->params->T = T;

    /* Create solver with initialized params */
    h->solver = new HJBSolver(*h->params);
    return 1;
  } catch (...) {
    return 0;
  }
}

int hjb_solve(HJBSolverHandle* handle) {
  _HPPInternal* h = HANDLE(handle);
  if (!h || !h->solver) return 0;

  try {
    auto start = std::chrono::high_resolution_clock::now();
    h->solver->solve();
    auto end = std::chrono::high_resolution_clock::now();
    h->solve_time_ms = 
      std::chrono::duration<float, std::milli>(end - start).count();
    return 1;
  } catch (...) {
    return 0;
  }
}

int hjb_get_quote(HJBSolverHandle* handle, float S, float I, int t,
                  Quote* out_quote) {
  _HPPInternal* h = HANDLE(handle);
  if (!h || !h->solver || !out_quote) return 0;

  try {
    hjb::Quote internal_q = h->solver->get_quotes((double)S, (double)I, (double)t);
    /* Copy internal quote to output (cast double to float for C interface) */
    out_quote->bid_price = (float)internal_q.bid_price;
    out_quote->ask_price = (float)internal_q.ask_price;
    out_quote->bid_intensity = (float)internal_q.bid_intensity;
    out_quote->ask_intensity = (float)internal_q.ask_intensity;
    out_quote->convergence_iters = internal_q.convergence_iters;
    return 1;
  } catch (...) {
    return 0;
  }
}

float hjb_get_bid_offset(HJBSolverHandle* handle, float S, float I, int t) {
  _HPPInternal* h = HANDLE(handle);
  if (!h || !h->solver || S <= 0.0f) return -999.0f;

  try {
    hjb::Quote q = h->solver->get_quotes((double)S, (double)I, (double)t);
    return 100.0f * (float)(q.bid_price / S - 1.0);
  } catch (...) {
    return -999.0f;
  }
}

float hjb_get_ask_offset(HJBSolverHandle* handle, float S, float I, int t) {
  _HPPInternal* h = HANDLE(handle);
  if (!h || !h->solver || S <= 0.0f) return -999.0f;

  try {
    hjb::Quote q = h->solver->get_quotes((double)S, (double)I, (double)t);
    return 100.0f * (float)(q.ask_price / S - 1.0);
  } catch (...) {
    return -999.0f;
  }
}

long hjb_get_gpu_memory_used(HJBSolverHandle* handle) {
  _HPPInternal* h = HANDLE(handle);
  if (!h || !h->solver) return 0;
  return 64 * 1024 * 1024;
}

float hjb_get_solve_time_ms(HJBSolverHandle* handle) {
  _HPPInternal* h = HANDLE(handle);
  if (!h) return 0.0f;
  return h->solve_time_ms;
}

void hjb_destroy_solver(HJBSolverHandle* handle) {
  _HPPInternal* h = HANDLE(handle);
  if (h) {
    if (h->params) delete h->params;
    if (h->solver) delete h->solver;
    delete h;
  }
}
