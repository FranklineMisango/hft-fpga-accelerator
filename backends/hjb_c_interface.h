#ifndef HJB_C_INTERFACE_H
#define HJB_C_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque solver handle - implementation detail hidden from interface */
typedef void* HJBSolverHandle;

/* Quote struct for C interface - renamed to avoid hjb::Quote conflict */
typedef struct {
  float bid_price;
  float ask_price;
  float bid_intensity;
  float ask_intensity;
  int convergence_iters;
} HJBQuote_C;

/* Create solver instance */
HJBSolverHandle* hjb_create_solver();

/* Initialize solver with parameters
 * Returns 1 on success, 0 on failure */
int hjb_init(HJBSolverHandle* handle,
             float sigma, float mu, float gamma, float kappa,
             float alpha, float lambda,
             int NS, int NI, int NT,
             float S_min, float S_max, float I_min, float I_max, float T);

/* Run backward solving
 * Returns 1 on success, 0 on failure */
int hjb_solve(HJBSolverHandle* handle);

/* Get quote for given (S, I, t)
 * Returns 1 on success, 0 on out-of-bounds */
int hjb_get_quote(HJBSolverHandle* handle, float S, float I, int t,
                  HJBQuote_C* out_quote);

/* Get bid at given (S, I, t) as percentage offset from S
 * Returns offset or -999.0 on error */
float hjb_get_bid_offset(HJBSolverHandle* handle, float S, float I, int t);

/* Get ask at given (S, I, t) as percentage offset from S
 * Returns offset or -999.0 on error */
float hjb_get_ask_offset(HJBSolverHandle* handle, float S, float I, int t);

/* Get solver statistics
 * Returns total GPU memory used in bytes */
long hjb_get_gpu_memory_used(HJBSolverHandle* handle);

/* Get solver solve time in milliseconds */
float hjb_get_solve_time_ms(HJBSolverHandle* handle);

/* Destroy solver instance */
void hjb_destroy_solver(HJBSolverHandle* handle);

#ifdef __cplusplus
}
#endif

#endif
