#!/usr/bin/env python3
"""Live trading dashboard for Binance or replayed market data.

This is a lightweight visualization layer on top of the existing simulation
workflow. It does not replace the RTL testbenches; instead, it shows a live
market feed, synthetic order generation, execution events, and simulated
end-to-end latency in a rolling chart.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import re
import shutil
import subprocess
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Tuple

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from websocket import WebSocketApp

from binance_order_book import start_depth_feed


@dataclass
class MarketTick:
    source_ts: float
    recv_ts: float
    ts: float
    symbol: str
    mid: float
    bid: float
    ask: float
    source: str
    bid_qty: float = 0.0
    ask_qty: float = 0.0

    @property
    def source_lag_us(self) -> float:
        return max((self.recv_ts - self.source_ts) * 1_000_000.0, 0.0)

    @property
    def dashboard_lag_us(self) -> float:
        return max((self.ts - self.recv_ts) * 1_000_000.0, 0.0)


@dataclass
class OrderEvent:
    order_id: int
    side: str
    source_ts: float
    recv_ts: float
    signal_ts: float
    exec_ts: float
    price: float
    quantity: float = 1.0

    @property
    def queue_lag_us(self) -> float:
        return max((self.signal_ts - self.recv_ts) * 1_000_000.0, 0.0)


@dataclass
class ExecEvent:
    order_id: int
    side: str
    source_ts: float
    recv_ts: float
    signal_ts: float
    exec_ts: float
    price: float
    latency_us: float
    quantity: float = 1.0
    liquidity: str = "maker"
    fee_paid: float = 0.0

    @property
    def end_to_end_us(self) -> float:
        return max((self.exec_ts - self.source_ts) * 1_000_000.0, 0.0)


@dataclass
class QuoteEvent:
    quote_id: int
    source_ts: float
    recv_ts: float
    sim_start_ts: float
    sim_end_ts: float
    mid: float
    bid: float
    ask: float
    latency_cycles: int
    latency_ns: float
    volatility: float

    @property
    def compute_lag_us(self) -> float:
        return max((self.sim_end_ts - self.sim_start_ts) * 1_000_000.0, 0.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_hjb_quote_output(output_path: Path) -> tuple[float, float]:
    text = output_path.read_text(encoding="utf-8").strip()
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Unexpected HJB quote output: {text!r}")
    return float(parts[0]), float(parts[1])


def _parse_latency_cycles(stdout: str) -> int:
    match = re.search(r"Latency:\s*(\d+)\s*cycles", stdout)
    if not match:
        raise ValueError("Unable to parse HJB latency from simulator output")
    return int(match.group(1))


def _annualized_volatility(mids: Deque[float]) -> float:
    if len(mids) < 3:
        return 0.3
    values = list(mids)
    returns = []
    for prev, curr in zip(values, values[1:]):
        if prev > 0 and curr > 0:
            returns.append(math.log(curr / prev))
    if len(returns) < 2:
        return 0.3
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return min(max(math.sqrt(max(variance, 0.0)) * math.sqrt(31_536_000), 0.01), 2.0)


class VerilogQuoteEngine:
    def __init__(self, tick_size: float = 0.01, latency_guard: float = 2.0) -> None:
        self.tick_size = max(tick_size, 0.0000001)
        self.latency_guard = max(latency_guard, 0.05)
        self.repo_root = _repo_root()
        self.input_path = self.repo_root / "market_input_verilog.txt"
        self.output_path = self.repo_root / "strategy_verilog_output.txt"
        self.exe_path = self.repo_root / "sim" / "verilog_market_maker_tb"
        self._ensure_built()

    def _ensure_built(self) -> None:
        if shutil.which("iverilog") is None:
            raise RuntimeError("iverilog is required for --maker-use-verilog-quoter but was not found in PATH")
        if shutil.which("vvp") is None:
            raise RuntimeError("vvp is required for --maker-use-verilog-quoter but was not found in PATH")
        sim_dir = self.repo_root / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-Wall",
                "-Winfloop",
                "-o",
                str(self.exe_path),
                str(self.repo_root / "rtl" / "verilog_market_maker.v"),
                str(self.repo_root / "testbench" / "verilog_market_maker_tb.v"),
            ],
            cwd=self.repo_root,
            check=True,
        )

    def quote(self, tick: "MarketTick", inventory: float, volatility: float) -> Tuple[float, float, bool, bool]:
        mid_ticks = int(round(tick.mid / self.tick_size))
        bid_ticks = int(round(tick.bid / self.tick_size))
        ask_ticks = int(round(tick.ask / self.tick_size))
        inv_milli = int(round(inventory * 1000.0))
        vol_bp = int(round(volatility * 10_000.0))
        bid_qty_milli = int(round(float(getattr(tick, "bid_qty", 0.0) or 0.0) * 1000.0))
        ask_qty_milli = int(round(float(getattr(tick, "ask_qty", 0.0) or 0.0) * 1000.0))

        self.input_path.write_text(
            f"{mid_ticks},{bid_ticks},{ask_ticks},{inv_milli},{vol_bp},{bid_qty_milli},{ask_qty_milli}\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            ["vvp", str(self.exe_path)],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=self.latency_guard,
            check=False,
        )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Verilog quoter execution failed: {stderr or 'unknown vvp error'}")

        if not self.output_path.exists():
            raise RuntimeError("Verilog quoter did not produce strategy_verilog_output.txt")

        text = self.output_path.read_text(encoding="utf-8").strip()
        parts = [part.strip() for part in text.split(",")]
        if len(parts) < 4:
            raise ValueError(f"Unexpected Verilog quote output: {text!r}")

        out_bid_ticks = int(parts[0])
        out_ask_ticks = int(parts[1])
        replace_hint = int(parts[2]) != 0
        cancel_hint = int(parts[3]) != 0
        return (
            out_bid_ticks * self.tick_size,
            out_ask_ticks * self.tick_size,
            replace_hint,
            cancel_hint,
        )


# ---------------------------------------------------------------------------
# RTL bridge: pnl_tracker.v
# ---------------------------------------------------------------------------
class RTLPnLTracker:
    """Drives pnl_tracker.v via iverilog/vvp for every fill and mark event."""

    SCALE = 100  # cents — prices multiplied by 100 before passing to RTL

    def __init__(self, repo_root: Path) -> None:
        self.root = repo_root
        self.exe  = repo_root / "sim" / "rtl_pnl_bridge_tb"
        self._build()
        # running totals returned from last RTL call
        self.total_realized   = 0.0
        self.total_unrealized = 0.0
        self.net_position     = 0.0
        self.daily_pnl        = 0.0
        # accumulate fills between mark calls
        self._pending_fill: Optional[dict] = None

    def _build(self) -> None:
        if not shutil.which("iverilog"):
            raise RuntimeError("iverilog not found — required for RTL PnL bridge")
        subprocess.run(
            ["iverilog", "-g2012", "-Winfloop", "-o", str(self.exe),
             str(self.root / "rtl" / "pnl_tracker.v"),
             str(self.root / "testbench" / "rtl_pnl_bridge_tb.v")],
            cwd=self.root, check=True, capture_output=True,
        )

    def on_fill(self, side: str, price: float, volume: float, mark_price: float) -> None:
        inp = self.root / "rtl_pnl_input.txt"
        out = self.root / "rtl_pnl_output.txt"
        fill_valid  = 1
        fill_side   = 1 if side == "SELL" else 0
        fill_price  = max(1, int(round(price  * self.SCALE)))
        fill_volume = max(1, int(round(volume * self.SCALE)))
        mark_p      = max(1, int(round(mark_price * self.SCALE)))
        inp.write_text(f"{fill_valid},{fill_side},{fill_price},{fill_volume},{mark_p}\n")
        subprocess.run(["vvp", str(self.exe)], cwd=self.root,
                       capture_output=True, timeout=5, check=False)
        self._read_output(out)

    def on_mark(self, mark_price: float) -> None:
        inp = self.root / "rtl_pnl_input.txt"
        out = self.root / "rtl_pnl_output.txt"
        mark_p = max(1, int(round(mark_price * self.SCALE)))
        inp.write_text(f"0,0,0,0,{mark_p}\n")
        subprocess.run(["vvp", str(self.exe)], cwd=self.root,
                       capture_output=True, timeout=5, check=False)
        self._read_output(out)

    def _read_output(self, out: Path) -> None:
        if not out.exists():
            return
        try:
            parts = out.read_text().strip().split(",")
            s = self.SCALE * self.SCALE  # price*volume scale
            self.total_realized   = int(parts[0]) / s
            self.total_unrealized = int(parts[1]) / s
            self.net_position     = int(parts[2]) / self.SCALE
            self.daily_pnl        = int(parts[3]) / s
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RTL bridge: risk_manager.v
# ---------------------------------------------------------------------------
class RTLRiskManager:
    """Drives risk_manager.v via iverilog/vvp for every order."""

    SCALE = 100

    def __init__(self, repo_root: Path,
                 max_position: float = 10.0,
                 max_order_size: float = 1.0,
                 max_drawdown: float = 5000.0) -> None:
        self.root = repo_root
        self.exe  = repo_root / "sim" / "rtl_risk_bridge_tb"
        self.max_position  = max_position
        self.max_order_size = max_order_size
        self.max_drawdown  = max_drawdown
        self._build()
        self.approved_cnt = 0
        self.rejected_cnt = 0
        self.kill_active  = False
        self.last_reason  = 0

    def _build(self) -> None:
        if not shutil.which("iverilog"):
            raise RuntimeError("iverilog not found — required for RTL risk bridge")
        subprocess.run(
            ["iverilog", "-g2012", "-Winfloop", "-o", str(self.exe),
             str(self.root / "rtl" / "risk_manager.v"),
             str(self.root / "testbench" / "rtl_risk_bridge_tb.v")],
            cwd=self.root, check=True, capture_output=True,
        )

    def check(self, side: str, volume: float, price: float,
              net_position: float, daily_pnl: float) -> bool:
        inp = self.root / "rtl_risk_input.txt"
        out = self.root / "rtl_risk_output.txt"
        s = self.SCALE
        inp.write_text(
            f"1,{1 if side=='SELL' else 0},"
            f"{max(1,int(round(volume*s)))},"
            f"{max(1,int(round(price*s)))},"
            f"{int(round(net_position*s))},"
            f"{int(round(daily_pnl*s))},"
            f"{max(1,int(round(self.max_position*s)))},"
            f"{max(1,int(round(self.max_order_size*s)))},"
            f"{max(1,int(round(self.max_drawdown*s)))},"
            f"0\n"
        )
        subprocess.run(["vvp", str(self.exe)], cwd=self.root,
                       capture_output=True, timeout=5, check=False)
        if not out.exists():
            return True  # fail open if RTL unavailable
        try:
            parts = out.read_text().strip().split(",")
            approved         = int(parts[0]) != 0
            self.last_reason = int(parts[1])
            self.kill_active = int(parts[2]) != 0
            self.approved_cnt= int(parts[3])
            self.rejected_cnt= int(parts[4])
            return approved
        except Exception:
            return True


# ---------------------------------------------------------------------------
# RTL bridge: latency_monitor.v
# ---------------------------------------------------------------------------
class RTLLatencyMonitor:
    """Drives latency_monitor.v to record tick-to-exec latency in RTL cycles."""

    NS_PER_CYCLE = 10  # 100 MHz clock (10ns period)

    def __init__(self, repo_root: Path, sla_us: float = 500.0) -> None:
        self.root    = repo_root
        self.exe     = repo_root / "sim" / "rtl_latency_bridge_tb"
        self.sla_cycles = max(1, int(sla_us * 1000 / self.NS_PER_CYCLE))
        self._build()
        self.last_latency_ns  = 0.0
        self.min_latency_ns   = float("inf")
        self.max_latency_ns   = 0.0
        self.mean_latency_ns  = 0.0
        self.sla_breach_cnt   = 0
        self.sample_cnt       = 0
        self.any_sla_breach   = False

    def _build(self) -> None:
        if not shutil.which("iverilog"):
            raise RuntimeError("iverilog not found — required for RTL latency bridge")
        subprocess.run(
            ["iverilog", "-g2012", "-Winfloop", "-o", str(self.exe),
             str(self.root / "rtl" / "latency_monitor.v"),
             str(self.root / "testbench" / "rtl_latency_bridge_tb.v")],
            cwd=self.root, check=True, capture_output=True,
        )

    def record(self, channel: int, latency_us: float) -> None:
        """Submit a measured latency (in µs) to the RTL monitor on given channel."""
        cycles = max(1, int(latency_us * 1000 / self.NS_PER_CYCLE))
        inp = self.root / "rtl_latency_input.txt"
        out = self.root / "rtl_latency_output.txt"
        inp.write_text(f"{channel},{cycles},{self.sla_cycles}\n")
        subprocess.run(["vvp", str(self.exe)], cwd=self.root,
                       capture_output=True, timeout=5, check=False)
        if not out.exists():
            return
        try:
            parts = out.read_text().strip().split(",")
            self.last_latency_ns  = int(parts[0]) * self.NS_PER_CYCLE
            min_c = int(parts[1])
            self.min_latency_ns   = min_c * self.NS_PER_CYCLE if min_c < 0xFFFFFFFF else float("inf")
            self.max_latency_ns   = int(parts[2]) * self.NS_PER_CYCLE
            self.mean_latency_ns  = int(parts[3]) * self.NS_PER_CYCLE
            self.sla_breach_cnt   = int(parts[4])
            self.sample_cnt       = int(parts[5])
            self.any_sla_breach   = int(parts[6]) != 0
        except Exception:
            pass


class StrategySim:
    def __init__(self, latency_us: int, spread_threshold_bps: float, window: int) -> None:
        self.latency_us = latency_us
        self.spread_threshold_bps = spread_threshold_bps
        self.window = window
        self.mid_history: Deque[float] = deque(maxlen=window)
        self.pending_orders: Deque[OrderEvent] = deque()
        self.next_order_id = 1

    def observe(self, tick: MarketTick) -> Optional[OrderEvent]:
        self.mid_history.append(tick.mid)
        if len(self.mid_history) < 8:
            return None

        avg_mid = sum(self.mid_history) / len(self.mid_history)
        deviation_bps = ((tick.mid - avg_mid) / avg_mid) * 10_000.0
        spread_bps = ((tick.ask - tick.bid) / tick.mid) * 10_000.0 if tick.mid > 0 else 0.0

        # Simple ultra-low-latency style signal: lean against fast dislocations
        if abs(deviation_bps) >= self.spread_threshold_bps or spread_bps >= self.spread_threshold_bps:
            side = "BUY" if deviation_bps < 0 else "SELL"
            order = OrderEvent(
                order_id=self.next_order_id,
                side=side,
                source_ts=tick.source_ts,
                recv_ts=tick.recv_ts,
                signal_ts=tick.ts,
                exec_ts=tick.ts + (self.latency_us / 1_000_000.0),
                price=tick.mid,
            )
            self.next_order_id += 1
            self.pending_orders.append(order)
            return order

        return None

    def due_executions(self, now_ts: float) -> Iterable[ExecEvent]:
        while self.pending_orders and self.pending_orders[0].exec_ts <= now_ts:
            order = self.pending_orders.popleft()
            yield ExecEvent(
                order_id=order.order_id,
                side=order.side,
                source_ts=order.source_ts,
                recv_ts=order.recv_ts,
                signal_ts=order.signal_ts,
                exec_ts=order.exec_ts,
                price=order.price,
                latency_us=(order.exec_ts - order.signal_ts) * 1_000_000.0,
                quantity=1.0,
                liquidity="taker",
                fee_paid=0.0,
            )


@dataclass
class _MakerQuote:
    order_id: int
    side: str
    price: float
    quantity: float
    remaining: float
    queue_ahead: float
    placed_ts: float
    last_replace_ts: float


class MakerSim:
    def __init__(
        self,
        base_spread_bps: float,
        skew_bps_per_unit: float,
        order_size: float,
        max_inventory: float,
        replace_bps: float,
        quote_ttl_ms: float,
        queue_join_ratio: float,
        maker_fee_bps: float,
        taker_fee_bps: float,
        taker_hedge: bool,
        verilog_quoter: Optional[Callable[[MarketTick, float, float], Tuple[float, float, bool, bool]]] = None,
    ) -> None:
        self.base_spread_bps = max(base_spread_bps, 0.1)
        self.skew_bps_per_unit = max(skew_bps_per_unit, 0.0)
        self.order_size = max(order_size, 0.0001)
        self.max_inventory = max(max_inventory, self.order_size)
        self.replace_bps = max(replace_bps, 0.01)
        self.quote_ttl_ms = max(quote_ttl_ms, 20.0)
        self.queue_join_ratio = min(max(queue_join_ratio, 0.0), 1.0)
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.taker_hedge = taker_hedge
        self.verilog_quoter = verilog_quoter

        self.next_order_id = 1
        self.bid_quote: Optional[_MakerQuote] = None
        self.ask_quote: Optional[_MakerQuote] = None
        self.inventory = 0.0
        self.cash = 0.0
        self.fees_paid = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.partial_fills = 0
        self.maker_fills = 0
        self.taker_fills = 0
        self.replaces = 0
        self.cancels = 0
        self.new_orders = 0
        self.prev_bid = 0.0
        self.prev_ask = 0.0
        self.prev_bid_qty = 0.0
        self.prev_ask_qty = 0.0
        self.force_replace = False
        self.force_cancel = False
        self.mid_history: Deque[float] = deque(maxlen=240)

    @staticmethod
    def _bps_diff(a: float, b: float, mid: float) -> float:
        if mid <= 0:
            return 0.0
        return abs(a - b) / mid * 10_000.0

    def _target_quotes(self, tick: MarketTick) -> tuple[float, float]:
        self.mid_history.append(tick.mid)
        volatility = _annualized_volatility(self.mid_history)
        if self.verilog_quoter is not None:
            try:
                bid, ask, replace_hint, cancel_hint = self.verilog_quoter(tick, self.inventory, volatility)
                self.force_replace = replace_hint
                self.force_cancel = cancel_hint
                if ask <= bid:
                    ask = bid + max(tick.mid * 0.00005, 0.01)
                return bid, ask
            except Exception as exc:
                print(f"Verilog quoter fallback to python logic: {exc}")
                self.force_replace = False
                self.force_cancel = False

        self.force_replace = False
        self.force_cancel = False

        mid = tick.mid
        half_spread = mid * (self.base_spread_bps / 10_000.0) / 2.0
        skew = mid * (self.skew_bps_per_unit / 10_000.0) * self.inventory
        reservation = mid - skew

        target_bid = min(tick.bid, reservation - half_spread)
        target_ask = max(tick.ask, reservation + half_spread)
        if target_ask <= target_bid:
            target_ask = target_bid + max(mid * 0.00005, 0.01)
        return target_bid, target_ask

    def _new_quote(self, side: str, price: float, qty: float, now_ts: float, touch_qty: float) -> _MakerQuote:
        quote = _MakerQuote(
            order_id=self.next_order_id,
            side=side,
            price=price,
            quantity=qty,
            remaining=qty,
            queue_ahead=max(touch_qty, 0.0) * self.queue_join_ratio,
            placed_ts=now_ts,
            last_replace_ts=now_ts,
        )
        self.next_order_id += 1
        self.new_orders += 1
        return quote

    def _replace_needed(self, quote: _MakerQuote, target: float, now_ts: float, mid: float) -> bool:
        age_ms = (now_ts - quote.last_replace_ts) * 1000.0
        if age_ms >= self.quote_ttl_ms:
            return True
        return self._bps_diff(quote.price, target, mid) >= self.replace_bps

    def _apply_fill(
        self,
        quote: _MakerQuote,
        fill_qty: float,
        fill_price: float,
        tick: MarketTick,
        liquidity: str,
    ) -> Optional[ExecEvent]:
        if fill_qty <= 0:
            return None
        fill_qty = min(fill_qty, quote.remaining)
        quote.remaining -= fill_qty

        fee_bps = self.maker_fee_bps if liquidity == "maker" else self.taker_fee_bps
        notional = fill_price * fill_qty
        fee_paid = notional * (fee_bps / 10_000.0)
        self.cash -= fee_paid
        self.fees_paid += fee_paid

        prev_inventory = self.inventory

        if quote.side == "BUY":
            if prev_inventory < 0:
                close_qty = min(fill_qty, -prev_inventory)
                self.realized_pnl += (self.avg_entry_price - fill_price) * close_qty
                remainder = fill_qty - close_qty
                if remainder > 0:
                    self.avg_entry_price = fill_price
            else:
                total_qty = prev_inventory + fill_qty
                if total_qty > 0:
                    self.avg_entry_price = (
                        (self.avg_entry_price * prev_inventory) + (fill_price * fill_qty)
                    ) / total_qty
            self.inventory += fill_qty
            self.cash -= notional
        else:
            if prev_inventory > 0:
                close_qty = min(fill_qty, prev_inventory)
                self.realized_pnl += (fill_price - self.avg_entry_price) * close_qty
                remainder = fill_qty - close_qty
                if remainder > 0:
                    self.avg_entry_price = fill_price
            else:
                total_qty = (-prev_inventory) + fill_qty
                if total_qty > 0:
                    self.avg_entry_price = (
                        (self.avg_entry_price * (-prev_inventory)) + (fill_price * fill_qty)
                    ) / total_qty
            self.inventory -= fill_qty
            self.cash += notional

        if abs(self.inventory) <= 1e-12:
            self.inventory = 0.0
            self.avg_entry_price = 0.0

        if liquidity == "maker":
            self.maker_fills += 1
        else:
            self.taker_fills += 1

        if quote.remaining > 1e-12 and fill_qty < quote.quantity:
            self.partial_fills += 1

        return ExecEvent(
            order_id=quote.order_id,
            side=quote.side,
            source_ts=tick.source_ts,
            recv_ts=tick.recv_ts,
            signal_ts=tick.ts,
            exec_ts=tick.ts,
            price=fill_price,
            latency_us=max((tick.ts - tick.recv_ts) * 1_000_000.0, 0.0),
            quantity=fill_qty,
            liquidity=liquidity,
            fee_paid=fee_paid,
        )

    def _passive_fill_qty(self, quote: _MakerQuote, tick: MarketTick) -> float:
        if quote.side == "BUY":
            if abs(quote.price - tick.bid) > 1e-10:
                return 0.0
            depletion = max(self.prev_bid_qty - float(getattr(tick, "bid_qty", 0.0) or 0.0), 0.0)
        else:
            if abs(quote.price - tick.ask) > 1e-10:
                return 0.0
            depletion = max(self.prev_ask_qty - float(getattr(tick, "ask_qty", 0.0) or 0.0), 0.0)

        if depletion <= 0:
            return 0.0
        if quote.queue_ahead >= depletion:
            quote.queue_ahead -= depletion
            return 0.0

        available = depletion - quote.queue_ahead
        quote.queue_ahead = 0.0
        return min(available, quote.remaining)

    def _crossed(self, quote: _MakerQuote, tick: MarketTick) -> bool:
        if quote.side == "BUY":
            return tick.ask <= quote.price
        return tick.bid >= quote.price

    def on_tick(self, tick: MarketTick) -> tuple[list[OrderEvent], list[ExecEvent]]:
        order_events: list[OrderEvent] = []
        exec_events: list[ExecEvent] = []

        now_ts = tick.ts
        bid_qty = float(getattr(tick, "bid_qty", 0.0) or 0.0)
        ask_qty = float(getattr(tick, "ask_qty", 0.0) or 0.0)

        target_bid, target_ask = self._target_quotes(tick)

        if self.force_cancel:
            if self.bid_quote is not None:
                self.cancels += 1
                self.bid_quote = None
            if self.ask_quote is not None:
                self.cancels += 1
                self.ask_quote = None
            self.prev_bid = tick.bid
            self.prev_ask = tick.ask
            self.prev_bid_qty = bid_qty
            self.prev_ask_qty = ask_qty
            return order_events, exec_events

        if self.bid_quote is None:
            self.bid_quote = self._new_quote("BUY", target_bid, self.order_size, now_ts, bid_qty)
            order_events.append(
                OrderEvent(
                    order_id=self.bid_quote.order_id,
                    side="BUY",
                    source_ts=tick.source_ts,
                    recv_ts=tick.recv_ts,
                    signal_ts=now_ts,
                    exec_ts=now_ts,
                    price=self.bid_quote.price,
                    quantity=self.bid_quote.quantity,
                )
            )
        elif self._replace_needed(self.bid_quote, target_bid, now_ts, tick.mid) or self.force_replace:
            self.cancels += 1
            self.replaces += 1
            self.bid_quote = self._new_quote("BUY", target_bid, self.order_size, now_ts, bid_qty)
            order_events.append(
                OrderEvent(
                    order_id=self.bid_quote.order_id,
                    side="BUY",
                    source_ts=tick.source_ts,
                    recv_ts=tick.recv_ts,
                    signal_ts=now_ts,
                    exec_ts=now_ts,
                    price=self.bid_quote.price,
                    quantity=self.bid_quote.quantity,
                )
            )

        if self.ask_quote is None:
            self.ask_quote = self._new_quote("SELL", target_ask, self.order_size, now_ts, ask_qty)
            order_events.append(
                OrderEvent(
                    order_id=self.ask_quote.order_id,
                    side="SELL",
                    source_ts=tick.source_ts,
                    recv_ts=tick.recv_ts,
                    signal_ts=now_ts,
                    exec_ts=now_ts,
                    price=self.ask_quote.price,
                    quantity=self.ask_quote.quantity,
                )
            )
        elif self._replace_needed(self.ask_quote, target_ask, now_ts, tick.mid) or self.force_replace:
            self.cancels += 1
            self.replaces += 1
            self.ask_quote = self._new_quote("SELL", target_ask, self.order_size, now_ts, ask_qty)
            order_events.append(
                OrderEvent(
                    order_id=self.ask_quote.order_id,
                    side="SELL",
                    source_ts=tick.source_ts,
                    recv_ts=tick.recv_ts,
                    signal_ts=now_ts,
                    exec_ts=now_ts,
                    price=self.ask_quote.price,
                    quantity=self.ask_quote.quantity,
                )
            )

        if self.bid_quote is not None:
            fill_qty = self._passive_fill_qty(self.bid_quote, tick)
            if fill_qty > 0:
                filled = self._apply_fill(self.bid_quote, fill_qty, self.bid_quote.price, tick, liquidity="maker")
                if filled is not None:
                    exec_events.append(filled)
            if self.bid_quote is not None and self._crossed(self.bid_quote, tick):
                filled = self._apply_fill(self.bid_quote, self.bid_quote.remaining, self.bid_quote.price, tick, liquidity="maker")
                if filled is not None:
                    exec_events.append(filled)
            if self.bid_quote is not None and self.bid_quote.remaining <= 1e-12:
                self.bid_quote = None

        if self.ask_quote is not None:
            fill_qty = self._passive_fill_qty(self.ask_quote, tick)
            if fill_qty > 0:
                filled = self._apply_fill(self.ask_quote, fill_qty, self.ask_quote.price, tick, liquidity="maker")
                if filled is not None:
                    exec_events.append(filled)
            if self.ask_quote is not None and self._crossed(self.ask_quote, tick):
                filled = self._apply_fill(self.ask_quote, self.ask_quote.remaining, self.ask_quote.price, tick, liquidity="maker")
                if filled is not None:
                    exec_events.append(filled)
            if self.ask_quote is not None and self.ask_quote.remaining <= 1e-12:
                self.ask_quote = None

        if self.taker_hedge:
            if self.inventory > self.max_inventory:
                hedge_qty = min(self.inventory - self.max_inventory, self.order_size)
                taker_quote = _MakerQuote(-1, "SELL", tick.bid, hedge_qty, hedge_qty, 0.0, now_ts, now_ts)
                filled = self._apply_fill(taker_quote, hedge_qty, tick.bid, tick, liquidity="taker")
                if filled is not None:
                    exec_events.append(filled)
            elif self.inventory < -self.max_inventory:
                hedge_qty = min(-self.max_inventory - self.inventory, self.order_size)
                taker_quote = _MakerQuote(-2, "BUY", tick.ask, hedge_qty, hedge_qty, 0.0, now_ts, now_ts)
                filled = self._apply_fill(taker_quote, hedge_qty, tick.ask, tick, liquidity="taker")
                if filled is not None:
                    exec_events.append(filled)

        self.prev_bid = tick.bid
        self.prev_ask = tick.ask
        self.prev_bid_qty = bid_qty
        self.prev_ask_qty = ask_qty
        return order_events, exec_events

    def metrics(self, mark_price: float) -> Dict[str, float]:
        mtm = self.cash + self.inventory * mark_price
        if self.inventory > 0:
            unrealized = (mark_price - self.avg_entry_price) * self.inventory
        elif self.inventory < 0:
            unrealized = (self.avg_entry_price - mark_price) * (-self.inventory)
        else:
            unrealized = 0.0
        realized_net = self.realized_pnl - self.fees_paid
        total_pnl = realized_net + unrealized
        return {
            "inventory": self.inventory,
            "cash": self.cash,
            "fees_paid": self.fees_paid,
            "mtm": mtm,
            "realized_pnl": realized_net,
            "unrealized_pnl": unrealized,
            "total_pnl": total_pnl,
            "new_orders": float(self.new_orders),
            "cancels": float(self.cancels),
            "replaces": float(self.replaces),
            "partial_fills": float(self.partial_fills),
            "maker_fills": float(self.maker_fills),
            "taker_fills": float(self.taker_fills),
        }


class HJBEngine:
    def __init__(
        self,
        pump: "EventPump",
        latency_guard: float = 60.0,
        fill_model: str = "simulated",
        fill_probability: float = 0.35,
    ) -> None:
        self.pump = pump
        self.repo_root = _repo_root()
        self.market_input = self.repo_root / "market_input.txt"
        self.strategy_output = self.repo_root / "strategy_output.txt"
        self.hjb_exe = self.repo_root / "sim" / "hjb_calculator_tb"
        self.queue: "queue.Queue[Optional[MarketTick]]" = queue.Queue(maxsize=10_000)
        self.mid_history: Deque[float] = deque(maxlen=240)
        self.latency_guard = latency_guard
        self.fill_model = fill_model
        self.fill_probability = max(0.0, min(fill_probability, 1.0))
        self.quote_id = 1
        self.fill_side_toggle = 0
        self.dropped_ticks = 0
        self.max_queue_depth = 0
        self._ensure_built()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _ensure_built(self) -> None:
        if self.hjb_exe.exists():
            return
        sim_dir = self.repo_root / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-Wall",
                "-Winfloop",
                "-o",
                str(self.hjb_exe),
                str(self.repo_root / "rtl" / "hjb_calculator.v"),
                str(self.repo_root / "testbench" / "hjb_calculator_tb.v"),
            ],
            cwd=self.repo_root,
            check=True,
        )

    def submit_tick(self, tick: MarketTick) -> None:
        try:
            self.queue.put_nowait(tick)
            self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())
        except queue.Full:
            self.dropped_ticks += 1
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(tick)
                self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())
            except queue.Full:
                self.dropped_ticks += 1
                pass

    def metrics(self) -> Dict[str, int]:
        return {
            "depth": self.queue.qsize(),
            "dropped": self.dropped_ticks,
            "max_depth": self.max_queue_depth,
        }

    def stop(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            tick = self.queue.get()
            if tick is None:
                return

            self.mid_history.append(tick.mid)
            volatility = _annualized_volatility(self.mid_history)
            sim_start_ts = time.time()

            self.market_input.write_text(f"{tick.mid:.8f},0,{volatility:.6f}\n", encoding="utf-8")
            result = subprocess.run(
                [str(self.hjb_exe)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.latency_guard,
                check=False,
            )
            sim_end_ts = time.time()

            latency_cycles = _parse_latency_cycles(result.stdout)
            latency_ns = latency_cycles * 4.0
            bid, ask = _parse_hjb_quote_output(self.strategy_output)

            quote_event = {
                "kind": "quote",
                "quote_id": self.quote_id,
                "source_ts": tick.source_ts,
                "recv_ts": tick.recv_ts,
                "sim_start_ts": sim_start_ts,
                "sim_end_ts": sim_end_ts,
                "ts": sim_end_ts,
                "symbol": tick.symbol,
                "mid": tick.mid,
                "bid": bid,
                "ask": ask,
                "latency_cycles": latency_cycles,
                "latency_ns": latency_ns,
                "volatility": volatility,
                "source": "hjb-rtl",
            }
            self.pump.put(quote_event)

            fill_side: Optional[str] = None
            fill_price = 0.0
            if self.fill_model == "strict":
                if tick.ask <= bid:
                    fill_side = "BUY"
                    fill_price = bid
                elif tick.bid >= ask:
                    fill_side = "SELL"
                    fill_price = ask
            else:
                # Simulated maker fill model: emit fills probabilistically so the
                # dashboard can visualize executions even when strict crossing is rare.
                if random.random() <= self.fill_probability:
                    fill_side = "BUY" if self.fill_side_toggle % 2 == 0 else "SELL"
                    fill_price = tick.ask if fill_side == "BUY" else tick.bid
                    self.fill_side_toggle += 1

            if fill_side is not None:
                exec_event = {
                    "kind": "execution",
                    "order_id": self.quote_id,
                    "side": fill_side,
                    "source_ts": tick.source_ts,
                    "recv_ts": tick.recv_ts,
                    "signal_ts": sim_start_ts,
                    "exec_ts": sim_end_ts,
                    "ts": sim_end_ts,
                    "symbol": tick.symbol,
                    "price": fill_price,
                    "latency_us": latency_ns / 1000.0,
                    "execution_model": self.fill_model,
                    "source": "hjb-fill",
                }
                self.pump.put(exec_event)

            self.quote_id += 1


class HJBCFFIEngine:
    def __init__(
        self,
        pump: "EventPump",
        config_path: Path,
        inventory: int = 0,
        time_index: int = 0,
        fill_model: str = "simulated",
        fill_probability: float = 0.35,
    ) -> None:
        self.pump = pump
        self.repo_root = _repo_root()
        self.queue: "queue.Queue[Optional[MarketTick]]" = queue.Queue(maxsize=10_000)
        self.max_queue_depth = 0
        self.dropped_ticks = 0
        self.quote_id = 1
        self.fill_model = fill_model
        self.fill_probability = max(0.0, min(fill_probability, 1.0))
        self.fill_side_toggle = 0
        self.inventory = inventory
        self.time_index = max(time_index, 0)
        self.mid_history: Deque[float] = deque(maxlen=240)

        backend_dir = self.repo_root / "backends"
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        from hjb_cffi import HJBSolver, SolverParams

        self._solver = HJBSolver()
        params = SolverParams.from_json(str(config_path))
        self._solver.initialize(params)
        self._solver.solve()

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit_tick(self, tick: MarketTick) -> None:
        try:
            self.queue.put_nowait(tick)
            self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())
        except queue.Full:
            self.dropped_ticks += 1
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(tick)
                self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())
            except queue.Full:
                self.dropped_ticks += 1

    def metrics(self) -> Dict[str, int]:
        return {
            "depth": self.queue.qsize(),
            "dropped": self.dropped_ticks,
            "max_depth": self.max_queue_depth,
        }

    def stop(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            tick = self.queue.get()
            if tick is None:
                return

            self.mid_history.append(tick.mid)
            volatility = _annualized_volatility(self.mid_history)
            sim_start_ts = time.time()
            quote = self._solver.get_quotes(tick.mid, self.inventory, self.time_index)
            sim_end_ts = time.time()

            compute_us = max((sim_end_ts - sim_start_ts) * 1_000_000.0, 0.0)
            latency_ns = compute_us * 1000.0
            latency_cycles = int(latency_ns / 4.0)

            quote_event = {
                "kind": "quote",
                "quote_id": self.quote_id,
                "source_ts": tick.source_ts,
                "recv_ts": tick.recv_ts,
                "sim_start_ts": sim_start_ts,
                "sim_end_ts": sim_end_ts,
                "ts": sim_end_ts,
                "symbol": tick.symbol,
                "mid": tick.mid,
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "latency_cycles": latency_cycles,
                "latency_ns": latency_ns,
                "volatility": volatility,
                "source": "hjb-cffi",
            }
            self.pump.put(quote_event)

            fill_side: Optional[str] = None
            fill_price = 0.0
            if self.fill_model == "strict":
                if tick.ask <= quote.bid_price:
                    fill_side = "BUY"
                    fill_price = quote.bid_price
                elif tick.bid >= quote.ask_price:
                    fill_side = "SELL"
                    fill_price = quote.ask_price
            else:
                if random.random() <= self.fill_probability:
                    fill_side = "BUY" if self.fill_side_toggle % 2 == 0 else "SELL"
                    fill_price = tick.ask if fill_side == "BUY" else tick.bid
                    self.fill_side_toggle += 1

            if fill_side is not None:
                exec_event = {
                    "kind": "execution",
                    "order_id": self.quote_id,
                    "side": fill_side,
                    "source_ts": tick.source_ts,
                    "recv_ts": tick.recv_ts,
                    "signal_ts": sim_start_ts,
                    "exec_ts": sim_end_ts,
                    "ts": sim_end_ts,
                    "symbol": tick.symbol,
                    "price": fill_price,
                    "latency_us": compute_us,
                    "execution_model": self.fill_model,
                    "source": "hjb-cffi-fill",
                }
                self.pump.put(exec_event)

            self.quote_id += 1


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_DARK   = "#0b1220"
BG_PANEL  = "#111827"
BG_CARD   = "#1a2235"
FG_TEXT   = "#e5e7eb"
C_MID     = "#00bcd4"
C_BID     = "#4caf50"
C_ASK     = "#ff9800"
C_QBID    = "#8e44ad"
C_QASK    = "#c0392b"
C_LAT_SRC = "#3498db"
C_LAT_DSH = "#9b59b6"
C_LAT_EXC = "#e67e22"
C_LAT_E2E = "#c0392b"
C_LAT_HJB = "#16a085"
C_PNL_REA = "#2ecc71"
C_PNL_UNR = "#f1c40f"
C_PNL_TOT = "#1abc9c"
C_SPR     = "#3498db"
C_IMB     = "#9b59b6"


class Dashboard(QtWidgets.QMainWindow):
    def __init__(
        self,
        title: str,
        max_points: int,
        max_orders: int,
        backend: str,
        quote_max_deviation_pct: float,
    ) -> None:
        super().__init__()
        self.title = title
        self.max_points = max_points
        self.max_orders = max_orders
        self.backend = backend
        self.quote_max_deviation_pct = max(0.01, quote_max_deviation_pct)

        # ---- data buffers ----
        self.times: Deque[float] = deque(maxlen=max_points)
        self.mids: Deque[float] = deque(maxlen=max_points)
        self.bids: Deque[float] = deque(maxlen=max_points)
        self.asks: Deque[float] = deque(maxlen=max_points)
        self.order_times: Deque[float] = deque(maxlen=max_orders)
        self.order_prices: Deque[float] = deque(maxlen=max_orders)
        self.order_sides: Deque[str] = deque(maxlen=max_orders)
        self.exec_times: Deque[float] = deque(maxlen=max_orders)
        self.exec_prices: Deque[float] = deque(maxlen=max_orders)
        self.exec_sides: Deque[str] = deque(maxlen=max_orders)
        self.latencies_us: Deque[float] = deque(maxlen=max_orders)
        self.source_lags_us: Deque[float] = deque(maxlen=max_points)
        self.dashboard_lags_us: Deque[float] = deque(maxlen=max_points)
        self.queue_lags_us: Deque[float] = deque(maxlen=max_orders)
        self.end_to_end_us: Deque[float] = deque(maxlen=max_orders)
        self.quote_times: Deque[float] = deque(maxlen=max_points)
        self.quote_bids: Deque[float] = deque(maxlen=max_points)
        self.quote_asks: Deque[float] = deque(maxlen=max_points)
        self.compute_lags_us: Deque[float] = deque(maxlen=max_points)
        self.latency_cycles: Deque[int] = deque(maxlen=max_points)
        self.spread_bps: Deque[float] = deque(maxlen=max_points)
        self.imbalance_pct: Deque[float] = deque(maxlen=max_points)
        self.maker_realized_series: Deque[float] = deque(maxlen=max_points)
        self.maker_unrealized_series: Deque[float] = deque(maxlen=max_points)
        self.maker_total_series: Deque[float] = deque(maxlen=max_points)
        self.maker_inventory_series: Deque[float] = deque(maxlen=max_points)

        # ---- scalar state ----
        self.total_ticks = 0
        self.total_orders = 0
        self.total_execs = 0
        self.last_symbol = "-"
        self.last_source = "-"
        self.last_latency = 0.0
        self.last_event_wall = time.time()
        self.pump_depth = 0
        self.pump_dropped = 0
        self.pump_max_depth = 0
        self.hjb_depth = 0
        self.hjb_dropped = 0
        self.hjb_max_depth = 0
        self.maker_inventory = 0.0
        self.maker_mtm = 0.0
        self.maker_fees = 0.0
        self.maker_realized = 0.0
        self.maker_unrealized = 0.0
        self.maker_total_pnl = 0.0
        self._cost_basis = 0.0
        self.maker_cancels = 0
        self.maker_replaces = 0
        self.maker_partial_fills = 0
        self.maker_maker_fills = 0
        self.maker_taker_fills = 0

        # ---- font: prefer "Average" ----
        db = QtGui.QFontDatabase()
        families = list(db.families())
        avg = [f for f in families if "average" in f.lower()]
        font_family = avg[0] if avg else "DejaVu Sans"
        QtWidgets.QApplication.instance().setFont(QtGui.QFont(font_family, 10))

        # ---- pyqtgraph global config ----
        pg.setConfigOptions(antialias=True, foreground=FG_TEXT, background=BG_DARK)

        # ---- window ----
        self.setWindowTitle(title)
        self.resize(1600, 960)
        self.setStyleSheet(f"background-color: {BG_DARK}; color: {FG_TEXT};")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # title bar
        lbl = QtWidgets.QLabel(title)
        lbl.setFont(QtGui.QFont(font_family, 15, QtGui.QFont.Bold))
        lbl.setStyleSheet("color: #f8fafc; padding: 4px 0;")
        root.addWidget(lbl)

        def _pw(title_text: str) -> pg.PlotWidget:
            w = pg.PlotWidget()
            w.setBackground(BG_PANEL)
            w.showGrid(x=True, y=True, alpha=0.18)
            w.getAxis("left").setTextPen(FG_TEXT)
            w.getAxis("bottom").setTextPen(FG_TEXT)
            w.setTitle(
                f'<span style="color:{FG_TEXT};font-family:{font_family};'
                f'font-size:11pt;font-weight:600;">{title_text}</span>'
            )
            return w

        price_w = _pw("Price Feed")
        lat_w   = _pw("Latency")
        pnl_w   = _pw("PnL")
        micro_w = _pw("Microstructure")

        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top.addWidget(price_w)
        top.addWidget(lat_w)
        top.setSizes([900, 500])

        bot = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bot.addWidget(pnl_w)
        bot.addWidget(micro_w)
        bot.setSizes([700, 700])

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vsplit.addWidget(top)
        vsplit.addWidget(bot)
        vsplit.setSizes([580, 320])
        root.addWidget(vsplit, stretch=1)

        # status strip
        self._status = QtWidgets.QLabel("Initialising…")
        mono = QtGui.QFont("Monospace", 9)
        mono.setStyleHint(QtGui.QFont.Monospace)
        self._status.setFont(mono)
        self._status.setStyleSheet(
            f"background:{BG_CARD};color:{FG_TEXT};padding:4px 8px;"
            f"border-top:1px solid #1f2937;"
        )
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        # ---- price curves ----
        self._c_mid  = price_w.plot(pen=pg.mkPen(C_MID,  width=2),   name="Mid")
        self._c_bid  = price_w.plot(pen=pg.mkPen(C_BID,  width=1),   name="Bid")
        self._c_ask  = price_w.plot(pen=pg.mkPen(C_ASK,  width=1),   name="Ask")
        self._c_qbid = price_w.plot(pen=pg.mkPen(C_QBID, width=1.5, style=QtCore.Qt.DashLine), name="HJB Bid")
        self._c_qask = price_w.plot(pen=pg.mkPen(C_QASK, width=1.5, style=QtCore.Qt.DashLine), name="HJB Ask")
        self._s_orders = pg.ScatterPlotItem(size=9,  symbol="t", pen=pg.mkPen(None))
        self._s_execs  = pg.ScatterPlotItem(size=12, symbol="o", pen=pg.mkPen("w", width=0.5))
        price_w.addItem(self._s_orders)
        price_w.addItem(self._s_execs)
        price_w.addLegend(offset=(10, 10))

        # ---- latency curves ----
        self._l_src = lat_w.plot(pen=pg.mkPen(C_LAT_SRC, width=1.4), name="Source lag")
        self._l_dsh = lat_w.plot(pen=pg.mkPen(C_LAT_DSH, width=1.2), name="Dashboard lag")
        self._l_exc = lat_w.plot(pen=pg.mkPen(C_LAT_EXC, width=1.2), name="Exec latency")
        self._l_e2e = lat_w.plot(pen=pg.mkPen(C_LAT_E2E, width=1.4, style=QtCore.Qt.DashLine), name="End-to-end")
        self._l_hjb = lat_w.plot(pen=pg.mkPen(C_LAT_HJB, width=1.3), name="HJB compute")
        lat_w.addLegend(offset=(10, 10))

        # ---- PnL curves ----
        self._p_rea = pnl_w.plot(pen=pg.mkPen(C_PNL_REA, width=1.4), name="Realized")
        self._p_unr = pnl_w.plot(pen=pg.mkPen(C_PNL_UNR, width=1.2), name="Unrealized")
        self._p_tot = pnl_w.plot(pen=pg.mkPen(C_PNL_TOT, width=1.8), name="Total")
        pnl_w.addLegend(offset=(10, 10))

        # ---- microstructure curves ----
        self._m_spr  = micro_w.plot(pen=pg.mkPen(C_SPR, width=1.3), name="Spread (bps)")
        self._m_imb  = micro_w.plot(pen=pg.mkPen(C_IMB, width=1.1), name="Imbalance (%)")
        self._m_zero = micro_w.plot(pen=pg.mkPen("#95a5a6", width=0.8, style=QtCore.Qt.DashLine))
        micro_w.addLegend(offset=(10, 10))
        self._fill_lbl = pg.TextItem("Fill ratio: 0.0%", color=FG_TEXT, anchor=(0, 0))
        self._fill_lbl.setFont(QtGui.QFont(font_family, 9))
        micro_w.addItem(self._fill_lbl)

        self.show()

    # ------------------------------------------------------------------
    def push_tick(self, tick: MarketTick) -> None:
        self.times.append(tick.ts)
        self.mids.append(tick.mid)
        self.bids.append(tick.bid)
        self.asks.append(tick.ask)
        self.source_lags_us.append(tick.source_lag_us)
        self.dashboard_lags_us.append(tick.dashboard_lag_us)
        spread_bps = ((tick.ask - tick.bid) / tick.mid) * 10_000.0 if tick.mid > 0 else 0.0
        self.spread_bps.append(max(spread_bps, 0.0))
        total_qty = tick.bid_qty + tick.ask_qty
        if total_qty > 0:
            imbalance = ((tick.bid_qty - tick.ask_qty) / total_qty) * 100.0
        elif len(self.mids) >= 2:
            # bookticker has no qty — use price momentum as proxy (-100..+100)
            prev = self.mids[-2]
            imbalance = max(-100.0, min(100.0, (tick.mid - prev) / max(prev, 1e-9) * 1_000_000.0))
        else:
            imbalance = 0.0
        self.imbalance_pct.append(imbalance)
        self.total_ticks += 1
        self.last_symbol = tick.symbol
        self.last_source = tick.source
        self.last_event_wall = time.time()

    def push_order(self, order: OrderEvent) -> None:
        self.order_times.append(order.signal_ts)
        self.order_prices.append(order.price)
        self.order_sides.append(order.side)
        self.queue_lags_us.append(order.queue_lag_us)
        self.total_orders += 1

    def push_execution(self, exec_event: ExecEvent) -> None:
        self.exec_times.append(exec_event.exec_ts)
        self.exec_prices.append(exec_event.price)
        self.exec_sides.append(exec_event.side)
        self.latencies_us.append(exec_event.latency_us)
        self.end_to_end_us.append(exec_event.end_to_end_us)
        self.total_execs += 1
        self.last_latency = exec_event.latency_us
        self.last_source = f"{exec_event.liquidity}-fill"
        self.last_event_wall = time.time()

        # Accumulate simulated PnL for all backends (simple FIFO model)
        qty = exec_event.quantity
        price = exec_event.price
        if exec_event.side == "BUY":
            self.maker_inventory += qty
            self._cost_basis = (
                (self._cost_basis * (self.maker_inventory - qty) + price * qty)
                / self.maker_inventory
                if self.maker_inventory > 0 else price
            )
        else:
            realized = (price - self._cost_basis) * qty
            self.maker_realized += realized
        mark = self.mids[-1] if self.mids else price
        self.maker_unrealized = (mark - self._cost_basis) * self.maker_inventory
        self.maker_total_pnl = self.maker_realized + self.maker_unrealized
        self.maker_realized_series.append(self.maker_realized)
        self.maker_unrealized_series.append(self.maker_unrealized)
        self.maker_total_series.append(self.maker_total_pnl)

    def push_quote(self, quote_event: QuoteEvent) -> None:
        if not self._quote_is_sane(quote_event):
            return
        self.quote_times.append(quote_event.sim_end_ts)
        self.quote_bids.append(quote_event.bid)
        self.quote_asks.append(quote_event.ask)
        self.compute_lags_us.append(quote_event.compute_lag_us)
        self.latency_cycles.append(quote_event.latency_cycles)
        self.last_symbol = f"Q{quote_event.quote_id}"
        self.last_source = "hjb-rtl"
        self.last_event_wall = time.time()

    def _quote_is_sane(self, quote_event: QuoteEvent) -> bool:
        if quote_event.mid <= 0 or quote_event.bid <= 0 or quote_event.ask <= 0:
            return False
        bid_dev = abs((quote_event.bid - quote_event.mid) / quote_event.mid)
        ask_dev = abs((quote_event.ask - quote_event.mid) / quote_event.mid)
        return bid_dev <= self.quote_max_deviation_pct and ask_dev <= self.quote_max_deviation_pct

    def set_telemetry(self, pump_depth, pump_dropped, pump_max_depth,
                      hjb_depth, hjb_dropped, hjb_max_depth) -> None:
        self.pump_depth = pump_depth
        self.pump_dropped = pump_dropped
        self.pump_max_depth = pump_max_depth
        self.hjb_depth = hjb_depth
        self.hjb_dropped = hjb_dropped
        self.hjb_max_depth = hjb_max_depth

    def set_maker_metrics(self, metrics: Dict[str, float]) -> None:
        self.maker_inventory  = float(metrics.get("inventory", 0.0))
        self.maker_mtm        = float(metrics.get("mtm", 0.0))
        self.maker_fees       = float(metrics.get("fees_paid", 0.0))
        self.maker_realized   = float(metrics.get("realized_pnl", 0.0))
        self.maker_unrealized = float(metrics.get("unrealized_pnl", 0.0))
        self.maker_total_pnl  = float(metrics.get("total_pnl", 0.0))
        self.maker_realized_series.append(self.maker_realized)
        self.maker_unrealized_series.append(self.maker_unrealized)
        self.maker_total_series.append(self.maker_total_pnl)
        self.maker_inventory_series.append(self.maker_inventory)
        self.maker_cancels       = int(metrics.get("cancels", 0))
        self.maker_replaces      = int(metrics.get("replaces", 0))
        self.maker_partial_fills = int(metrics.get("partial_fills", 0))
        self.maker_maker_fills   = int(metrics.get("maker_fills", 0))
        self.maker_taker_fills   = int(metrics.get("taker_fills", 0))

    def render(self) -> None:
        if not self.times:
            return

        x = list(self.times)
        self._c_mid.setData(x, list(self.mids))
        self._c_bid.setData(x, list(self.bids))
        self._c_ask.setData(x, list(self.asks))
        if self.quote_times:
            self._c_qbid.setData(list(self.quote_times), list(self.quote_bids))
            self._c_qask.setData(list(self.quote_times), list(self.quote_asks))

        if self.order_times:
            self._s_orders.setData(
                x=list(self.order_times), y=list(self.order_prices),
                brush=[pg.mkBrush("#2ecc71") if s == "BUY" else pg.mkBrush("#e74c3c")
                       for s in self.order_sides],
            )
        if self.exec_times:
            self._s_execs.setData(
                x=list(self.exec_times), y=list(self.exec_prices),
                brush=[pg.mkBrush("#2ecc71") if s == "BUY" else pg.mkBrush("#e74c3c")
                       for s in self.exec_sides],
            )

        def _set(curve, data):
            if data: curve.setData(data)
            else:    curve.clear()

        _set(self._l_src, list(self.source_lags_us))
        _set(self._l_dsh, list(self.dashboard_lags_us))
        _set(self._l_exc, list(self.latencies_us))
        _set(self._l_e2e, list(self.end_to_end_us))
        _set(self._l_hjb, list(self.compute_lags_us))
        _set(self._p_rea, list(self.maker_realized_series))
        _set(self._p_unr, list(self.maker_unrealized_series))
        _set(self._p_tot, list(self.maker_total_series))
        _set(self._m_spr, list(self.spread_bps))
        _set(self._m_imb, list(self.imbalance_pct))

        spr = list(self.spread_bps)
        if spr:
            self._m_zero.setData([0, len(spr) - 1], [0.0, 0.0])
            self._fill_lbl.setPos(0, max(spr))
        fill_ratio = (self.total_execs / self.total_orders * 100.0) if self.total_orders > 0 else 0.0
        self._fill_lbl.setText(f"Fill ratio: {fill_ratio:.1f}%")

        age_ms = (time.time() - self.last_event_wall) * 1000.0
        compute_lbl = ("REAL (HDL HJB)" if self.backend == "hjb"
                       else "MARKET MAKER SIM" if self.backend == "maker" else "SIMULATED")
        parts = [
            f"sym:{self.last_symbol}  src:{self.last_source}  compute:{compute_lbl}",
            (f"src-lag:{self.source_lags_us[-1]:,.1f}µs  "
             f"dash-lag:{self.dashboard_lags_us[-1]:,.1f}µs  "
             f"exec-lat:{self.last_latency:,.1f}µs") if self.source_lags_us else "lag:-",
            (f"pump q:{self.pump_depth}(max {self.pump_max_depth}) drop:{self.pump_dropped}  "
             f"ticks:{self.total_ticks}  orders:{self.total_orders}  execs:{self.total_execs}  "
             f"idle:{age_ms:.0f}ms"),
        ]
        if self.backend == "maker":
            parts.append(
                f"inv:{self.maker_inventory:,.4f}  mtm:{self.maker_mtm:,.2f}  "
                f"fees:{self.maker_fees:,.4f}  "
                f"realized:{self.maker_realized:,.2f}  unrealized:{self.maker_unrealized:,.2f}  "
                f"total:{self.maker_total_pnl:,.2f}  "
                f"cxl/rpl:{self.maker_cancels}/{self.maker_replaces}  "
                f"fills maker/taker:{self.maker_maker_fills}/{self.maker_taker_fills}"
            )
        self._status.setText("    |    ".join(parts))


class EventPump:
    def __init__(self) -> None:
        self.queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=10_000)
        self.stop_requested = threading.Event()
        self.dropped_events = 0
        self.max_depth = 0

    def put(self, event: Optional[Dict[str, Any]]) -> None:
        if self.stop_requested.is_set():
            return
        try:
            self.queue.put_nowait(event)
            self.max_depth = max(self.max_depth, self.queue.qsize())
        except queue.Full:
            self.dropped_events += 1
            # Drop old data rather than blocking the market feed.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(event)
                self.max_depth = max(self.max_depth, self.queue.qsize())
            except queue.Full:
                self.dropped_events += 1
                pass

    def get(self, timeout: float = 0.01) -> Optional[Dict[str, Any]]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def metrics(self) -> Dict[str, int]:
        return {
            "depth": self.queue.qsize(),
            "dropped": self.dropped_events,
            "max_depth": self.max_depth,
        }


class EventLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.file = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("a", encoding="utf-8")

    def log(self, event: Dict[str, Any]) -> None:
        if self.file is None:
            return
        self.file.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def parse_live_binance_message(symbol: str, stream: str, message: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None

    payload = data
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
            "source": "binance-bookTicker",
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
            "ts": utc_ts(),
            "symbol": symbol.upper(),
            "mid": price,
            "bid": price - spread / 2.0,
            "ask": price + spread / 2.0,
            "source": "binance-trade",
        }

    return None


def replay_events(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
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

            if stream == "bookticker":
                bid = float(payload["b"])
                ask = float(payload["a"])
                mid = (bid + ask) / 2.0
                captured_ts = float(row.get("captured_at_ts", 0.0)) if row.get("captured_at_ts") else 0.0
                now_ts = time.time()
                yield {
                    "kind": "tick",
                    "source_ts": captured_ts if captured_ts > 0 else now_ts,
                    "recv_ts": now_ts,
                    "ts": time.time(),
                    "symbol": str(row.get("symbol", "")) or "BINANCE",
                    "mid": mid,
                    "bid": bid,
                    "ask": ask,
                    "source": "replay-bookTicker",
                }
            elif stream == "trade":
                price = float(payload["p"])
                spread = max(price * 0.0005, 0.01)
                captured_ts = float(row.get("captured_at_ts", 0.0)) if row.get("captured_at_ts") else 0.0
                now_ts = time.time()
                yield {
                    "kind": "tick",
                    "source_ts": captured_ts if captured_ts > 0 else now_ts,
                    "recv_ts": now_ts,
                    "ts": time.time(),
                    "symbol": str(row.get("symbol", "")) or "BINANCE",
                    "mid": price,
                    "bid": price - spread / 2.0,
                    "ask": price + spread / 2.0,
                    "source": "replay-trade",
                }


def start_live_feed(pump: EventPump, symbol: str, stream: str) -> WebSocketApp:
    url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@{'bookTicker' if stream == 'bookticker' else 'trade'}"

    def on_message(_: WebSocketApp, message: str) -> None:
        parsed = parse_live_binance_message(symbol=symbol, stream=stream, message=message)
        if parsed is not None:
            pump.put(parsed)

    def on_error(_: WebSocketApp, error: Any) -> None:
        print(f"WebSocket error: {error}")

    def on_close(_: WebSocketApp, code: Any, reason: Any) -> None:
        print(f"WebSocket closed: code={code} reason={reason}")
        pump.put(None)

    app = WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)

    def runner() -> None:
        while not pump.stop_requested.is_set():
            app.run_forever(ping_interval=20, ping_timeout=10)
            if not pump.stop_requested.is_set():
                time.sleep(1.0)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return app


def start_replay_feed(pump: EventPump, input_path: Path, replay_speed: float) -> threading.Thread:
    def runner() -> None:
        previous_ts: Optional[float] = None
        for event in replay_events(input_path):
            if pump.stop_requested.is_set():
                break
            current_ts = event["ts"]
            if previous_ts is not None and replay_speed > 0:
                delay = max(current_ts - previous_ts, 0.0) / replay_speed
                time.sleep(min(delay, 0.25))
            previous_ts = current_ts
            pump.put(event)
        pump.put(None)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


def main() -> int:
    parser = argparse.ArgumentParser(description="Live trading/order-execution visualization")
    parser.add_argument("--mode", choices=["live", "replay"], default="live", help="Data source mode")
    parser.add_argument("--backend", choices=["synthetic", "hjb", "maker"], default="synthetic", help="Trading backend")
    parser.add_argument("--symbol", default="btcusdt", help="Binance symbol when in live mode")
    parser.add_argument("--stream", choices=["bookticker", "trade", "depth", "depth20"], default="bookticker")
    parser.add_argument("--depth-update-ms", type=int, default=100)
    parser.add_argument("--depth-snapshot-limit", type=int, default=1000)
    parser.add_argument("--input", default="data/binance_capture.ndjson")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--latency-us", type=int, default=35)
    parser.add_argument("--spread-threshold-bps", type=float, default=2.0)
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--max-orders", type=int, default=80)
    parser.add_argument("--quote-max-deviation-pct", type=float, default=0.25)
    parser.add_argument("--hjb-fill-model", choices=["strict", "simulated"], default="simulated")
    parser.add_argument("--hjb-fill-probability", type=float, default=0.35)
    parser.add_argument("--hjb-engine", choices=["rtl", "cffi"], default="cffi")
    parser.add_argument("--hjb-config", default="backends/demo_config.json")
    parser.add_argument("--hjb-inventory", type=int, default=0)
    parser.add_argument("--hjb-time-index", type=int, default=0)
    parser.add_argument("--event-log", default="")
    parser.add_argument("--maker-base-spread-bps", type=float, default=2.0)
    parser.add_argument("--maker-skew-bps-per-unit", type=float, default=0.4)
    parser.add_argument("--maker-order-size", type=float, default=0.01)
    parser.add_argument("--maker-max-inventory", type=float, default=0.08)
    parser.add_argument("--maker-replace-bps", type=float, default=0.8)
    parser.add_argument("--maker-ttl-ms", type=float, default=400.0)
    parser.add_argument("--maker-queue-join-ratio", type=float, default=0.45)
    parser.add_argument("--maker-fee-bps", type=float, default=1.0)
    parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser.add_argument("--enable-taker-hedge", action="store_true")
    parser.add_argument("--maker-use-verilog-quoter", action="store_true")
    parser.add_argument("--maker-tick-size", type=float, default=0.01)
    parser.add_argument("--maker-verilog-latency-guard", type=float, default=2.0)
    args = parser.parse_args()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    pump = EventPump()
    repo = _repo_root()

    # ---- RTL bridges (built once at startup) ----
    rtl_pnl  = RTLPnLTracker(repo)
    rtl_risk = RTLRiskManager(repo)
    rtl_lat  = RTLLatencyMonitor(repo)
    print("RTL bridges compiled: pnl_tracker, risk_manager, latency_monitor")
    dashboard = Dashboard(
        title="VeriTrade — FPGA + HJB Ultra-Low-Latency Market Maker",
        max_points=args.window,
        max_orders=args.max_orders,
        backend=args.backend,
        quote_max_deviation_pct=args.quote_max_deviation_pct,
    )
    logger = EventLogger(Path(args.event_log) if args.event_log else None)
    strategy = StrategySim(
        latency_us=args.latency_us,
        spread_threshold_bps=args.spread_threshold_bps,
        window=args.window,
    )
    hjb_engine: Optional[HJBEngine] = None
    maker_engine: Optional[MakerSim] = None
    verilog_quoter: Optional[VerilogQuoteEngine] = None

    if args.backend == "hjb":
        if args.hjb_engine == "rtl":
            hjb_engine = HJBEngine(pump, fill_model=args.hjb_fill_model, fill_probability=args.hjb_fill_probability)
        else:
            hjb_config = Path(args.hjb_config)
            if not hjb_config.is_absolute():
                hjb_config = _repo_root() / hjb_config
            hjb_engine = HJBCFFIEngine(
                pump,
                config_path=hjb_config,
                inventory=args.hjb_inventory,
                time_index=args.hjb_time_index,
                fill_model=args.hjb_fill_model,
                fill_probability=args.hjb_fill_probability,
            )
    elif args.backend == "maker":
        if args.maker_use_verilog_quoter:
            verilog_quoter = VerilogQuoteEngine(
                tick_size=args.maker_tick_size,
                latency_guard=args.maker_verilog_latency_guard,
            )
        maker_engine = MakerSim(
            base_spread_bps=args.maker_base_spread_bps,
            skew_bps_per_unit=args.maker_skew_bps_per_unit,
            order_size=args.maker_order_size,
            max_inventory=args.maker_max_inventory,
            replace_bps=args.maker_replace_bps,
            quote_ttl_ms=args.maker_ttl_ms,
            queue_join_ratio=args.maker_queue_join_ratio,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            taker_hedge=args.enable_taker_hedge,
            verilog_quoter=verilog_quoter.quote if verilog_quoter is not None else None,
        )

    def _stop(*_: Any) -> None:
        pump.stop_requested.set()
        pump.put(None)
        if hjb_engine is not None:
            hjb_engine.stop()
        logger.close()
        app.quit()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if args.mode == "live":
        print(f"Starting live Binance feed for {args.symbol.upper()} ({args.stream})")
        if args.stream in {"depth", "depth20"}:
            start_depth_feed(pump, args.symbol, stream=args.stream,
                             update_speed_ms=args.depth_update_ms,
                             snapshot_limit=args.depth_snapshot_limit)
        else:
            start_live_feed(pump, args.symbol, args.stream)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Replay file not found: {input_path}")
            return 2
        print(f"Replaying captured data from {input_path}")
        start_replay_feed(pump, input_path, args.replay_speed)

    # ---- Qt timer drives the update loop at ~30 fps ----
    def _tick() -> None:
        while True:
            event = pump.get(timeout=0.001)
            if event is None:
                break

            logger.log(dict(event))
            source_ts = float(event.get("source_ts", event["ts"]))
            recv_ts   = float(event.get("recv_ts",   event["ts"]))
            kind      = str(event.get("kind", "tick"))

            if kind == "tick":
                tick = MarketTick(
                    source_ts=source_ts, recv_ts=recv_ts,
                    ts=float(event["ts"]), symbol=str(event["symbol"]),
                    mid=float(event["mid"]), bid=float(event["bid"]),
                    ask=float(event["ask"]),
                    bid_qty=float(event.get("bid_qty", 0.0)),
                    ask_qty=float(event.get("ask_qty", 0.0)),
                    source=str(event.get("source", "feed")),
                )
                dashboard.push_tick(tick)

                if args.backend == "synthetic":
                    order = strategy.observe(tick)
                    if order is not None:
                        # RTL risk check before accepting order
                        approved = rtl_risk.check(
                            side=order.side,
                            volume=order.quantity,
                            price=order.price,
                            net_position=rtl_pnl.net_position,
                            daily_pnl=rtl_pnl.daily_pnl,
                        )
                        if approved:
                            dashboard.push_order(order)
                    for exec_ev in strategy.due_executions(time.time()):
                        # RTL PnL update on fill
                        rtl_pnl.on_fill(
                            side=exec_ev.side,
                            price=exec_ev.price,
                            volume=exec_ev.quantity,
                            mark_price=tick.mid,
                        )
                        # RTL latency record (CH2 = tick-to-exec)
                        rtl_lat.record(channel=2, latency_us=exec_ev.latency_us)
                        dashboard.push_execution(exec_ev)
                    # RTL mark-to-market on every tick
                    rtl_pnl.on_mark(tick.mid)
                    # Push RTL PnL into dashboard series
                    dashboard.maker_realized_series.append(rtl_pnl.total_realized)
                    dashboard.maker_unrealized_series.append(rtl_pnl.total_unrealized)
                    dashboard.maker_total_series.append(rtl_pnl.daily_pnl)
                    dashboard.maker_realized   = rtl_pnl.total_realized
                    dashboard.maker_unrealized = rtl_pnl.total_unrealized
                    dashboard.maker_total_pnl  = rtl_pnl.daily_pnl
                    dashboard.maker_inventory  = rtl_pnl.net_position
                elif args.backend == "maker" and maker_engine is not None:
                    orders, execs = maker_engine.on_tick(tick)
                    for o in orders:
                        dashboard.push_order(o)
                    for e in execs:
                        dashboard.push_execution(e)
                elif hjb_engine is not None:
                    hjb_engine.submit_tick(tick)

            elif kind == "quote":
                quote = QuoteEvent(
                    quote_id=int(event["quote_id"]),
                    source_ts=float(event["source_ts"]),
                    recv_ts=float(event["recv_ts"]),
                    sim_start_ts=float(event["sim_start_ts"]),
                    sim_end_ts=float(event["sim_end_ts"]),
                    mid=float(event["mid"]),
                    bid=float(event["bid"]),
                    ask=float(event["ask"]),
                    latency_cycles=int(event["latency_cycles"]),
                    latency_ns=float(event["latency_ns"]),
                    volatility=float(event.get("volatility", 0.0)),
                )
                dashboard.push_quote(quote)
                dashboard.push_order(OrderEvent(
                    order_id=quote.quote_id * 2 - 1, side="BUY",
                    source_ts=quote.source_ts, recv_ts=quote.recv_ts,
                    signal_ts=quote.sim_start_ts, exec_ts=quote.sim_end_ts,
                    price=quote.bid,
                ))
                dashboard.push_order(OrderEvent(
                    order_id=quote.quote_id * 2, side="SELL",
                    source_ts=quote.source_ts, recv_ts=quote.recv_ts,
                    signal_ts=quote.sim_start_ts, exec_ts=quote.sim_end_ts,
                    price=quote.ask,
                ))

            elif kind == "execution":
                dashboard.push_execution(ExecEvent(
                    order_id=int(event["order_id"]),
                    side=str(event["side"]),
                    source_ts=float(event["source_ts"]),
                    recv_ts=float(event["recv_ts"]),
                    signal_ts=float(event["signal_ts"]),
                    exec_ts=float(event["exec_ts"]),
                    price=float(event["price"]),
                    latency_us=float(event["latency_us"]),
                    quantity=float(event.get("quantity", 1.0)),
                    liquidity=str(event.get("liquidity", "maker")),
                    fee_paid=float(event.get("fee_paid", 0.0)),
                ))

        pump_m = pump.metrics()
        hjb_m  = hjb_engine.metrics() if hjb_engine else {"depth": 0, "dropped": 0, "max_depth": 0}
        dashboard.set_telemetry(
            pump_depth=int(pump_m["depth"]),
            pump_dropped=int(pump_m["dropped"]),
            pump_max_depth=int(pump_m["max_depth"]),
            hjb_depth=int(hjb_m["depth"]),
            hjb_dropped=int(hjb_m["dropped"]),
            hjb_max_depth=int(hjb_m["max_depth"]),
        )
        if maker_engine is not None and dashboard.mids:
            dashboard.set_maker_metrics(maker_engine.metrics(mark_price=dashboard.mids[-1]))

        # Push RTL latency monitor stats into dashboard latency series
        if rtl_lat.sample_cnt > 0:
            dashboard.source_lags_us.append(rtl_lat.last_latency_ns / 1000.0)
            dashboard.last_latency = rtl_lat.last_latency_ns / 1000.0

        dashboard.render()

    timer = QtCore.QTimer()
    timer.timeout.connect(_tick)
    timer.start(33)   # ~30 fps

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

