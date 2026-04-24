#!/usr/bin/env python3
"""Binance market-data helpers for local order-book simulation.

This module keeps the websocket parsing and local book-synchronization logic
out of the dashboard so the live path stays readable.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from websocket import WebSocketApp


def utc_ts() -> float:
    return time.time()


def build_stream_url(symbol: str, stream: str, update_speed_ms: int = 100) -> str:
    symbol_l = symbol.lower()
    if stream == "bookticker":
        return f"wss://stream.binance.com:9443/ws/{symbol_l}@bookTicker"
    if stream == "trade":
        return f"wss://stream.binance.com:9443/ws/{symbol_l}@trade"
    if stream == "depth":
        return f"wss://stream.binance.com:9443/ws/{symbol_l}@depth@{update_speed_ms}ms"
    if stream == "depth20":
        return f"wss://stream.binance.com:9443/ws/{symbol_l}@depth20@{update_speed_ms}ms"
    raise ValueError(f"Unsupported stream: {stream}")


def fetch_depth_snapshot(symbol: str, limit: int = 1000) -> Dict[str, Any]:
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit={limit}"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")
    snapshot = json.loads(payload)
    if not isinstance(snapshot, dict):
        raise ValueError("Depth snapshot response was not a JSON object")
    return snapshot


def _best_level(levels: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(levels, list) or not levels:
        return None
    first = levels[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return None
    try:
        price = float(first[0])
        quantity = float(first[1])
    except (TypeError, ValueError):
        return None
    if price <= 0 or quantity < 0:
        return None
    return price, quantity


def top_of_book_from_payload(payload: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(payload, dict):
        return None

    if "b" in payload and "a" in payload:
        bid = _best_level(payload.get("b"))
        ask = _best_level(payload.get("a"))
        if bid is None or ask is None:
            return None
        return bid[0], bid[1], ask[0], ask[1]

    if "bids" in payload and "asks" in payload:
        bid = _best_level(payload.get("bids"))
        ask = _best_level(payload.get("asks"))
        if bid is None or ask is None:
            return None
        return bid[0], bid[1], ask[0], ask[1]

    return None


def payload_to_tick(
    *,
    symbol: str,
    stream: str,
    payload: Dict[str, Any],
    source: str,
) -> Optional[Dict[str, Any]]:
    if stream == "bookticker":
        bid = float(payload["b"])
        ask = float(payload["a"])
        mid = (bid + ask) / 2.0
        source_ts = float(payload.get("E", 0.0)) / 1000.0
        recv_ts = utc_ts()
        if source_ts <= 0:
            source_ts = recv_ts
        return {
            "kind": "tick",
            "source_ts": source_ts,
            "recv_ts": recv_ts,
            "ts": recv_ts,
            "symbol": symbol.upper(),
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "source": source,
        }

    if stream == "trade":
        price = float(payload["p"])
        spread = max(price * 0.0005, 0.01)
        source_ts = float(payload.get("E", 0.0)) / 1000.0
        recv_ts = utc_ts()
        if source_ts <= 0:
            source_ts = recv_ts
        return {
            "kind": "tick",
            "source_ts": source_ts,
            "recv_ts": recv_ts,
            "ts": recv_ts,
            "symbol": symbol.upper(),
            "mid": price,
            "bid": price - spread / 2.0,
            "ask": price + spread / 2.0,
            "source": source,
        }

    top = top_of_book_from_payload(payload)
    if top is None:
        return None

    bid, bid_qty, ask, ask_qty = top
    mid = (bid + ask) / 2.0
    source_ts = float(payload.get("E", payload.get("eventTime", 0.0))) / 1000.0
    recv_ts = utc_ts()
    if source_ts <= 0:
        source_ts = recv_ts
    return {
        "kind": "tick",
        "source_ts": source_ts,
        "recv_ts": recv_ts,
        "ts": recv_ts,
        "symbol": symbol.upper(),
        "mid": mid,
        "bid": bid,
        "ask": ask,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "source": source,
    }


class LocalOrderBook:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update_id: Optional[int] = None

    def apply_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.last_update_id = int(snapshot["lastUpdateId"])
        self.bids = self._load_side(snapshot.get("bids", []))
        self.asks = self._load_side(snapshot.get("asks", []))

    @staticmethod
    def _load_side(levels: Iterable[Any]) -> Dict[float, float]:
        loaded: Dict[float, float] = {}
        for level in levels:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            try:
                price = float(level[0])
                quantity = float(level[1])
            except (TypeError, ValueError):
                continue
            if price > 0 and quantity > 0:
                loaded[price] = quantity
        return loaded

    def _update_side(self, side: Dict[float, float], levels: Iterable[Any]) -> None:
        for level in levels:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            try:
                price = float(level[0])
                quantity = float(level[1])
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            if quantity <= 0:
                side.pop(price, None)
            else:
                side[price] = quantity

    def apply_delta(self, payload: Dict[str, Any]) -> bool:
        update_id = int(payload.get("u", payload.get("lastUpdateId", 0)))
        first_update_id = int(payload.get("U", update_id))
        if self.last_update_id is None:
            return False
        if update_id <= self.last_update_id:
            return False
        if first_update_id != self.last_update_id + 1:
            return False

        self._update_side(self.bids, payload.get("b", payload.get("bids", [])))
        self._update_side(self.asks, payload.get("a", payload.get("asks", [])))
        self.last_update_id = update_id
        return True

    def apply_partial(self, payload: Dict[str, Any]) -> bool:
        top = top_of_book_from_payload(payload)
        if top is None:
            return False
        bid, bid_qty, ask, ask_qty = top
        self.bids = {bid: bid_qty}
        self.asks = {ask: ask_qty}
        update_id = int(payload.get("u", payload.get("lastUpdateId", 0)))
        if update_id > 0:
            self.last_update_id = update_id
        return True

    def top_of_book(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.bids or not self.asks:
            return None
        best_bid = max(self.bids)
        best_ask = min(self.asks)
        return best_bid, self.bids[best_bid], best_ask, self.asks[best_ask]


def start_depth_feed(
    pump: Any,
    symbol: str,
    stream: str = "depth",
    update_speed_ms: int = 100,
    snapshot_limit: int = 1000,
) -> threading.Thread:
    """Start a Binance depth feed and emit normalized tick events."""

    raw_queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=50_000)
    url = build_stream_url(symbol, stream, update_speed_ms=update_speed_ms)
    book = LocalOrderBook(symbol)

    def on_message(_: WebSocketApp, message: str) -> None:
        try:
            raw_queue.put_nowait(message)
        except queue.Full:
            try:
                raw_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                raw_queue.put_nowait(message)
            except queue.Full:
                pass

    def on_error(_: WebSocketApp, error: Any) -> None:
        print(f"WebSocket error: {error}")

    def on_close(_: WebSocketApp, code: Any, reason: Any) -> None:
        print(f"WebSocket closed: code={code} reason={reason}")
        try:
            raw_queue.put_nowait(None)
        except queue.Full:
            pass

    app = WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)

    def ws_runner() -> None:
        while not pump.stop_requested.is_set():
            app.run_forever(ping_interval=20, ping_timeout=10)
            if not pump.stop_requested.is_set():
                time.sleep(1.0)

    def emit_tick(payload: Dict[str, Any], source: str) -> None:
        top = payload_to_tick(symbol=symbol, stream=stream, payload=payload, source=source)
        if top is not None:
            pump.put(top)

    def coordinator() -> None:
        if stream == "depth":
            try:
                snapshot = fetch_depth_snapshot(symbol, limit=snapshot_limit)
                book.apply_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - network-dependent path
                print(f"Depth snapshot fetch failed: {exc}")
                return

        while not pump.stop_requested.is_set():
            try:
                message = raw_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if message is None:
                return

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            if stream == "depth20":
                if not book.apply_partial(payload):
                    continue
                emit_tick(payload, "binance-depth20")
                continue

            if book.last_update_id is None:
                continue

            update_id = int(payload.get("u", payload.get("lastUpdateId", 0)))
            first_update_id = int(payload.get("U", update_id))
            if update_id <= book.last_update_id:
                continue

            if first_update_id <= book.last_update_id + 1 <= update_id:
                if not book.apply_delta(payload):
                    continue
                emit_tick(payload, "binance-depth")
                continue

            # Gap in the depth sequence; refresh the snapshot to stay in sync.
            try:
                snapshot = fetch_depth_snapshot(symbol, limit=snapshot_limit)
                book.apply_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - network-dependent path
                print(f"Depth resync failed: {exc}")
                continue

    threading.Thread(target=ws_runner, daemon=True).start()
    thread = threading.Thread(target=coordinator, daemon=True)
    thread.start()
    return thread