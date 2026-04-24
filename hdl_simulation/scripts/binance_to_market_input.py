#!/usr/bin/env python3
"""Convert captured Binance NDJSON into market_input.txt format for hjb_calculator_tb."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from binance_order_book import LocalOrderBook, top_of_book_from_payload


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_mid(payload: Dict[str, Any], stream: str) -> Optional[float]:
    if stream == "bookticker":
        bid = to_float(payload.get("b"))
        ask = to_float(payload.get("a"))
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None
        return (bid + ask) / 2.0

    if stream == "trade":
        trade_price = to_float(payload.get("p"))
        if trade_price is None or trade_price <= 0:
            return None
        return trade_price

    if stream == "depth20":
        top = top_of_book_from_payload(payload)
        if top is None:
            return None
        bid, _, ask, _ = top
        return (bid + ask) / 2.0

    return None


def annualized_vol(prices: List[float]) -> float:
    if len(prices) < 3:
        return 0.3

    returns: List[float] = []
    for prev, curr in zip(prices, prices[1:]):
        if prev <= 0 or curr <= 0:
            continue
        returns.append(math.log(curr / prev))

    if len(returns) < 2:
        return 0.3

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(max(variance, 0.0))

    # Approximate annualization from per-tick stdev: sigma * sqrt(N)
    # N is chosen as 31,536,000 one-second intervals for a simple conservative estimate.
    sigma = std * math.sqrt(31_536_000)
    return min(max(sigma, 0.01), 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Binance capture to market_input.txt")
    parser.add_argument(
        "--input",
        default="data/binance_capture.ndjson",
        help="NDJSON file produced by capture_binance_ws.py",
    )
    parser.add_argument(
        "--output",
        default="market_input.txt",
        help="Output path for HJB testbench input",
    )
    parser.add_argument(
        "--inventory",
        type=int,
        default=0,
        help="Inventory integer to place in output",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="How many latest prices to use for volatility estimate",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    mids: List[float] = []
    depth_symbol: Optional[str] = None
    depth_book: Optional[LocalOrderBook] = None
    depth_ready = False
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            stream = str(row.get("stream", "")).lower()
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                continue

            if depth_symbol is None:
                depth_symbol = str(row.get("symbol", "BTCUSDT") or "BTCUSDT").upper()
                depth_book = LocalOrderBook(symbol=depth_symbol)

            mid = extract_mid(payload=payload, stream=stream)
            if mid is None and stream == "depth":
                if not depth_ready:
                    try:
                        from binance_order_book import fetch_depth_snapshot

                        if depth_book is None:
                            continue
                        depth_book.apply_snapshot(fetch_depth_snapshot(symbol=depth_symbol or "BTCUSDT"))
                        depth_ready = True
                    except Exception:
                        continue
                if depth_book is not None and depth_book.apply_delta(payload):
                    top = depth_book.top_of_book()
                    if top is not None:
                        bid, _, ask, _ = top
                        mid = (bid + ask) / 2.0
                elif depth_book is not None:
                    try:
                        from binance_order_book import fetch_depth_snapshot

                        depth_book.apply_snapshot(fetch_depth_snapshot(symbol=depth_symbol or "BTCUSDT"))
                        if depth_book.apply_delta(payload):
                            top = depth_book.top_of_book()
                            if top is not None:
                                bid, _, ask, _ = top
                                mid = (bid + ask) / 2.0
                    except Exception:
                        continue

            if mid is not None and mid > 0:
                mids.append(mid)

    if not mids:
        raise RuntimeError("No valid mid-prices were parsed from capture file")

    mids = mids[-max(args.window, 3):]
    mid_price = mids[-1]
    volatility = annualized_vol(mids)

    output_path = Path(args.output)
    output_path.write_text(f"{mid_price:.8f},{args.inventory},{volatility:.6f}\n", encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"mid_price={mid_price:.8f} inventory={args.inventory} volatility={volatility:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
