"""
HJB Market-Making Solver - Python CFFI Interface

Provides Python access to the GPU-accelerated HJB backward PDE solver
for optimal market-making quote generation.

Example:
    from hjb_cffi import HJBSolver
    
    solver = HJBSolver()
    solver.initialize(sigma=0.1, mu=0.0, gamma=0.01, ...)
    solver.solve()
    bid, ask = solver.get_quotes(S=100.0, I=0, t=0)
"""

import cffi
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

# Define C interface via CFFI
ffi = cffi.FFI()

# Load C header definitions
ffi.cdef("""
    typedef struct {
        float bid_price;
        float ask_price;
        float bid_intensity;
        float ask_intensity;
        int convergence_iters;
    } Quote;

    typedef struct HJBSolverHandle HJBSolverHandle;

    HJBSolverHandle* hjb_create_solver();
    int hjb_init(HJBSolverHandle* handle,
                 float sigma, float mu, float gamma, float kappa,
                 float alpha, float lambda,
                 int NS, int NI, int NT,
                 float S_min, float S_max, float I_min, float I_max, float T);
    int hjb_solve(HJBSolverHandle* handle);
    int hjb_get_quote(HJBSolverHandle* handle, float S, float I, int t, Quote* out_quote);
    float hjb_get_bid_offset(HJBSolverHandle* handle, float S, float I, int t);
    float hjb_get_ask_offset(HJBSolverHandle* handle, float S, float I, int t);
    long hjb_get_gpu_memory_used(HJBSolverHandle* handle);
    float hjb_get_solve_time_ms(HJBSolverHandle* handle);
    void hjb_destroy_solver(HJBSolverHandle* handle);
""")

# Load compiled shared library
_backend_dir = Path(__file__).parent
_lib_path = _backend_dir / "libhjb_solver.so"

if not _lib_path.exists():
    raise RuntimeError(
        f"libhjb_solver.so not found at {_lib_path}. "
        f"Run 'make cuda-cffi-build' to compile."
    )

lib = ffi.dlopen(str(_lib_path))


@dataclass
class Quote:
    """Market quote with bid/ask prices and execution intensities."""
    bid_price: float
    ask_price: float
    bid_intensity: float
    ask_intensity: float
    convergence_iters: int

    @property
    def spread_bps(self) -> float:
        """Spread in basis points (1 bps = 0.01%)."""
        if self.bid_price > 0:
            return 10000.0 * (self.ask_price / self.bid_price - 1.0)
        return 0.0

    @property
    def mid_price(self) -> float:
        """Midpoint of bid-ask spread."""
        return (self.bid_price + self.ask_price) / 2.0

    def to_dict(self) -> dict:
        """Export quote as dictionary."""
        return {
            "bid": self.bid_price,
            "ask": self.ask_price,
            "spread_bps": self.spread_bps,
            "mid": self.mid_price,
            "bid_intensity": self.bid_intensity,
            "ask_intensity": self.ask_intensity,
        }


@dataclass
class SolverParams:
    """HJB solver model parameters."""
    sigma: float = 0.1          # Volatility
    mu: float = 0.0             # Drift
    gamma: float = 0.01         # Inventory penalty
    kappa: float = 0.0001       # Order fill rate
    alpha: float = 0.01         # Execution sensitivity
    lambda_jump: float = 0.5    # Jump intensity

    # Grid parameters
    grid_points_S: int = 64     # Price grid points
    grid_points_I: int = 32     # Inventory grid points
    grid_points_t: int = 256    # Time grid points

    # Domain
    S_min: float = 95.0         # Min price
    S_max: float = 105.0        # Max price
    I_min: float = -50.0        # Min inventory
    I_max: float = 50.0         # Max inventory
    horizon: float = 0.1        # Time horizon (seconds)

    @classmethod
    def from_json(cls, json_path: str) -> "SolverParams":
        """Load parameters from JSON config file."""
        with open(json_path) as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        """Export parameters as dictionary."""
        return {
            "sigma": self.sigma,
            "mu": self.mu,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "lambda": self.lambda_jump,
            "grid": {
                "S": self.grid_points_S,
                "I": self.grid_points_I,
                "t": self.grid_points_t,
            },
            "domain": {
                "S_range": [self.S_min, self.S_max],
                "I_range": [self.I_min, self.I_max],
                "horizon": self.horizon,
            },
        }


class HJBSolver:
    """Python wrapper for GPU-accelerated HJB market-making solver."""

    def __init__(self):
        """Create new solver instance."""
        self._handle = lib.hjb_create_solver()
        if not self._handle:
            raise RuntimeError("Failed to create HJB solver instance")
        self._params: Optional[SolverParams] = None
        self._solve_time_ms: float = 0.0

    def initialize(self, params: Optional[SolverParams] = None, **kwargs) -> None:
        """
        Initialize solver with parameters.

        Args:
            params: SolverParams object, or None to use kwargs
            **kwargs: Individual parameter values if params is None

        Raises:
            ValueError: If initialization fails
        """
        if params is None:
            params = SolverParams(**kwargs)
        self._params = params

        ret = lib.hjb_init(
            self._handle,
            params.sigma,
            params.mu,
            params.gamma,
            params.kappa,
            params.alpha,
            params.lambda_jump,
            params.grid_points_S,
            params.grid_points_I,
            params.grid_points_t,
            params.S_min,
            params.S_max,
            params.I_min,
            params.I_max,
            params.horizon,
        )
        if not ret:
            raise ValueError("Failed to initialize HJB solver")

    def solve(self) -> float:
        """
        Run backward HJB iteration to convergence.

        Returns:
            Solve time in milliseconds

        Raises:
            RuntimeError: If solve fails or solver not initialized
        """
        if not self._handle:
            raise RuntimeError("Solver not initialized")

        ret = lib.hjb_solve(self._handle)
        if not ret:
            raise RuntimeError("Failed to solve HJB backward PDE")

        self._solve_time_ms = lib.hjb_get_solve_time_ms(self._handle)
        return self._solve_time_ms

    def get_quotes(self, S: float, I: int, t: int) -> Quote:
        """
        Get optimal bid/ask quotes for given market state.

        Args:
            S: Current mid-price
            I: Current inventory
            t: Time index (0 = start, NT-1 = terminal)

        Returns:
            Quote object with bid/ask prices and intensities

        Raises:
            RuntimeError: If quote retrieval fails
        """
        c_quote = ffi.new("Quote*")
        ret = lib.hjb_get_quote(self._handle, float(S), int(I), int(t), c_quote)
        if not ret:
            raise RuntimeError(f"Failed to get quote for S={S}, I={I}, t={t}")

        return Quote(
            bid_price=c_quote.bid_price,
            ask_price=c_quote.ask_price,
            bid_intensity=c_quote.bid_intensity,
            ask_intensity=c_quote.ask_intensity,
            convergence_iters=c_quote.convergence_iters,
        )

    def get_bid_offset(self, S: float, I: int, t: int) -> float:
        """Get bid offset as percentage from mid (e.g., -0.18 = -0.18%)."""
        offset = lib.hjb_get_bid_offset(self._handle, float(S), int(I), int(t))
        if offset <= -999.0:
            raise RuntimeError(f"Failed to get bid offset for S={S}, I={I}, t={t}")
        return offset

    def get_ask_offset(self, S: float, I: int, t: int) -> float:
        """Get ask offset as percentage from mid (e.g., +1.92 = +1.92%)."""
        offset = lib.hjb_get_ask_offset(self._handle, float(S), int(I), int(t))
        if offset <= -999.0:
            raise RuntimeError(f"Failed to get ask offset for S={S}, I={I}, t={t}")
        return offset

    @property
    def solve_time_ms(self) -> float:
        """Total solve time in milliseconds."""
        return self._solve_time_ms

    @property
    def gpu_memory_mb(self) -> float:
        """GPU memory used in MB."""
        bytes_used = lib.hjb_get_gpu_memory_used(self._handle)
        return bytes_used / (1024.0 * 1024.0)

    @property
    def params(self) -> Optional[SolverParams]:
        """Returns solver parameters."""
        return self._params

    def __del__(self):
        """Clean up GPU resources."""
        if self._handle:
            lib.hjb_destroy_solver(self._handle)
            self._handle = None


if __name__ == "__main__":
    # Quick test
    print("HJB CFFI Module Loaded Successfully")
    print(f"C library path: {_lib_path}")
    print(f"Quote fields: {Quote.__dataclass_fields__.keys()}")
