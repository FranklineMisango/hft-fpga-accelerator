#!/usr/bin/env python3
"""Capture Binance WebSocket events to NDJSON for deterministic replay."""

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from websocket import WebSocketApp

from binance_order_book import build_stream_url


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CaptureState:
    def __init__(self, output_path: Path, max_messages: int, quiet: bool):
        self.output_path = output_path
        self.max_messages = max_messages
        self.quiet = quiet
        self.count = 0
        self.stop_requested = False
        self.start_time = time.time()
        self.file = output_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self.file.close()

    def should_stop(self) -> bool:
        return self.stop_requested or (self.max_messages > 0 and self.count >= self.max_messages)

    def write_event(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":"))
        self.file.write(line + "\n")
        self.file.flush()
        self.count += 1
        if not self.quiet and self.count % 50 == 0:
            elapsed = time.time() - self.start_time
            print(f"Captured {self.count} messages in {elapsed:.1f}s", flush=True)


def build_stream(symbol: str, stream: str) -> str:
    return build_stream_url(symbol, stream)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Binance WebSocket data")
    parser.add_argument("--symbol", default="btcusdt", help="Binance symbol, e.g. btcusdt")
    parser.add_argument(
        "--stream",
        choices=["bookticker", "trade", "depth", "depth20"],
        default="bookticker",
        help="Stream type to capture",
    )
    parser.add_argument(
        "--output",
        default="data/binance_capture.ndjson",
        help="Output NDJSON file",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=500,
        help="Stop after this many messages, 0 means unlimited",
    )
    parser.add_argument(
        "--duration-sec",
        type=int,
        default=0,
        help="Optional duration cutoff in seconds, 0 means unlimited",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = build_stream(args.symbol, args.stream)
    state = CaptureState(output_path=output_path, max_messages=args.max_messages, quiet=args.quiet)

    print(f"Connecting to {url}")
    print(f"Writing NDJSON to {output_path}")

    deadline = time.time() + args.duration_sec if args.duration_sec > 0 else 0

    ws_app: WebSocketApp

    def request_stop(*_: Any) -> None:
        state.stop_requested = True
        try:
            ws_app.close()
        except Exception:
            pass

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def on_open(_: WebSocketApp) -> None:
        if not args.quiet:
            print(f"Connected at {utc_now_iso()}")

    def on_message(ws: WebSocketApp, message: str) -> None:
        if deadline and time.time() >= deadline:
            state.stop_requested = True
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        envelope = {
            "captured_at": utc_now_iso(),
            "symbol": args.symbol.lower(),
            "stream": args.stream,
            "payload": data,
        }
        state.write_event(envelope)

        if state.should_stop():
            ws.close()

    def on_error(_: WebSocketApp, error: Any) -> None:
        print(f"WebSocket error: {error}", file=sys.stderr)

    def on_close(_: WebSocketApp, code: Any, reason: Any) -> None:
        if not args.quiet:
            print(f"Closed: code={code} reason={reason}")

    ws_app = WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    try:
        ws_app.run_forever(ping_interval=20, ping_timeout=10)
    finally:
        state.close()

    print(f"Capture complete. Messages written: {state.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
