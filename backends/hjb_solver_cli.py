#!/usr/bin/env python3
"""
HJB Market-Making Solver - Standalone CLI Tool

Reads market state and solver parameters from JSON, runs GPU solver,
outputs optimal quotes to CSV.

Usage:
    python hjb_solver_cli.py config.json market_state.json --output quotes.csv
    
Example config.json:
    {
        "sigma": 0.1,
        "mu": 0.0,
        "gamma": 0.01,
        "kappa": 0.0001,
        "alpha": 0.01,
        "lambda_jump": 0.5,
        "grid_points_S": 64,
        "grid_points_I": 32,
        "grid_points_t": 256,
        "S_min": 95.0,
        "S_max": 105.0,
        "I_min": -50.0,
        "I_max": 50.0,
        "horizon": 0.1
    }

Example market_state.json:
    {
        "states": [
            {"S": 100.0, "I": 0, "t": 0},
            {"S": 100.0, "I": 10, "t": 0},
            {"S": 100.0, "I": -10, "t": 0}
        ]
    }
"""

import argparse
import json
import sys
import csv
from pathlib import Path
from typing import List, Dict
from hjb_cffi import HJBSolver, SolverParams, Quote


def load_config(config_path: str) -> SolverParams:
    """Load solver parameters from JSON file."""
    with open(config_path) as f:
        data = json.load(f)
    return SolverParams(**data)


def load_market_states(states_path: str) -> List[Dict]:
    """Load market states from JSON file."""
    with open(states_path) as f:
        data = json.load(f)
    return data.get("states", [])


def run_solver(config: SolverParams, verbose: bool = False) -> HJBSolver:
    """
    Initialize and run the HJB solver.

    Args:
        config: Solver parameters
        verbose: Print progress information

    Returns:
        HJBSolver instance with solved value function

    Raises:
        RuntimeError: If solver fails
    """
    if verbose:
        print(f"[*] Creating solver...")

    solver = HJBSolver()

    if verbose:
        print(f"[*] Initializing with parameters:")
        print(f"    Grid: {config.grid_points_S}×{config.grid_points_I}×{config.grid_points_t}")
        print(f"    Price range: [{config.S_min}, {config.S_max}]")
        print(f"    Inventory range: [{config.I_min}, {config.I_max}]")
        print(f"    Volatility: {config.sigma}")
        print(f"    Jump intensity: {config.lambda_jump}")

    solver.initialize(config)

    if verbose:
        print(f"[*] Running backward iteration...")

    solve_time = solver.solve()

    if verbose:
        print(f"[✓] Solve complete in {solve_time:.2f}ms")
        print(f"[✓] GPU memory used: {solver.gpu_memory_mb:.1f}MB")

    return solver


def generate_quotes(solver: HJBSolver, states: List[Dict], verbose: bool = False) -> List[Dict]:
    """
    Generate quotes for given market states.

    Args:
        solver: Solved HJB solver
        states: List of market states (S, I, t)
        verbose: Print progress

    Returns:
        List of quote dictionaries
    """
    quotes = []
    for i, state in enumerate(states):
        S = state["S"]
        I = state.get("I", 0)
        t = state.get("t", 0)

        try:
            quote = solver.get_quotes(S, I, t)
            bid_offset = solver.get_bid_offset(S, I, t)
            ask_offset = solver.get_ask_offset(S, I, t)

            record = {
                "S": S,
                "I": I,
                "t": t,
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "bid_offset_pct": bid_offset,
                "ask_offset_pct": ask_offset,
                "spread_bps": quote.spread_bps,
                "mid": quote.mid_price,
                "bid_intensity": quote.bid_intensity,
                "ask_intensity": quote.ask_intensity,
            }
            quotes.append(record)

            if verbose and (i + 1) % max(1, len(states) // 10) == 0:
                print(f"[*] Generated {i+1}/{len(states)} quotes...")

        except Exception as e:
            if verbose:
                print(f"[!] Error for state S={S}, I={I}, t={t}: {e}", file=sys.stderr)

    if verbose:
        print(f"[✓] Generated {len(quotes)} quotes")

    return quotes


def write_csv(quotes: List[Dict], output_path: str) -> None:
    """Write quotes to CSV file."""
    if not quotes:
        print("[!] No quotes to write", file=sys.stderr)
        return

    fieldnames = quotes[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(quotes)

    print(f"[✓] Wrote {len(quotes)} quotes to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HJB Market-Making Solver - Standalone CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate quotes for preset states
  python hjb_solver_cli.py config.json market_state.json -o quotes.csv
  
  # Generate quotes for price range (auto-generate states)
  python hjb_solver_cli.py config.json -o quotes.csv --S-range 95 105 --I-range -20 20
  
  # Verbose mode with performance info
  python hjb_solver_cli.py config.json market_state.json -o quotes.csv -v
        """,
    )

    parser.add_argument("config", help="JSON file with solver parameters")
    parser.add_argument(
        "market_states",
        nargs="?",
        help="JSON file with market states (optional if using --S-range)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output CSV file for quotes"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--S-range",
        nargs=2,
        type=float,
        help="Auto-generate states: price range min max",
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--I-range",
        nargs=2,
        type=int,
        help="Auto-generate states: inventory range min max",
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "-t", "--time-index",
        type=int,
        default=0,
        help="Time index for quotes (default: 0 = start of period)",
    )

    args = parser.parse_args()

    try:
        # Load config
        if args.verbose:
            print(f"[*] Loading config from {args.config}...")
        config = load_config(args.config)

        # Load or generate market states
        if args.S_range and args.I_range:
            # Auto-generate states
            if args.verbose:
                print(f"[*] Auto-generating states...")
            S_min, S_max = args.S_range
            I_min, I_max = args.I_range
            states = []
            for S in [S_min + (S_max - S_min) * i / 9 for i in range(10)]:
                for I in range(I_min, I_max + 1, max(1, (I_max - I_min) // 4)):
                    states.append({"S": S, "I": I, "t": args.time_index})
        else:
            if not args.market_states:
                parser.error(
                    "Must provide market_states.json or --S-range and --I-range"
                )
            if args.verbose:
                print(f"[*] Loading market states from {args.market_states}...")
            states = load_market_states(args.market_states)

        # Run solver
        solver = run_solver(config, verbose=args.verbose)

        # Generate quotes
        if args.verbose:
            print(f"[*] Generating quotes for {len(states)} states...")
        quotes = generate_quotes(solver, states, verbose=args.verbose)

        # Write output
        write_csv(quotes, args.output)

        if args.verbose:
            print(f"[✓] Done!")

    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
