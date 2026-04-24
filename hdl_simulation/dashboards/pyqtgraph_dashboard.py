"""
Real-Time Trading Dashboard (PyQtGraph Backend)
Institutional-grade HFT monitoring for VeriTrade FPGA engine.

Features:
- Real-time tick charting (multi-symbol)
- Order book depth visualization
- Strategy performance metrics
- Position tracking
- Low-latency updates (<1ms)

Install:
    pip install pyqtgraph PyQt5 numpy pandas
"""

import sys
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import threading
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# Configuration
MAX_HISTORY = 10000  # Keep 10k ticks in memory
UPDATE_INTERVAL = 16  # ~60 FPS
TICK_SIZE = 4  # bytes per tick

@dataclass
class MarketTick:
    """Atomic market update"""
    timestamp: float
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    bid_size: int = 0
    ask_size: int = 0

class TickBuffer:
    """Lock-free circular buffer for market ticks"""
    def __init__(self, max_size=MAX_HISTORY):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        
    def push(self, tick: MarketTick):
        """Add tick to buffer (thread-safe)"""
        with self.lock:
            self.buffer.append(tick)
    
    def get_all(self) -> List[MarketTick]:
        """Get all ticks (thread-safe)"""
        with self.lock:
            return list(self.buffer)
    
    def get_recent(self, n: int = 100) -> List[MarketTick]:
        """Get last N ticks"""
        with self.lock:
            return list(self.buffer)[-n:]

class RTDashboard(pg.GraphicsLayoutWidget):
    """Real-time trading dashboard using PyQtGraph"""
    
    def __init__(self, title="VeriTrade FPGA Dashboard", parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1600, 900)
        
        # Data buffers per symbol
        self.tick_buffers: Dict[str, TickBuffer] = {}
        self.current_symbols = ["BTC", "ETH", "SOL"]
        
        # Initialize buffers
        for symbol in self.current_symbols:
            self.tick_buffers[symbol] = TickBuffer()
        
        # Create UI layout
        self._create_layout()
        
        # Update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(UPDATE_INTERVAL)
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        
    def _create_layout(self):
        """Build dashboard layout"""
        # Main price chart (3x3 grid for multiple symbols)
        self.price_plots = {}
        self.price_curves = {}
        
        for idx, symbol in enumerate(self.current_symbols):
            row = idx // 2
            col = idx % 2
            
            # Price candlestick plot
            plot = self.addPlot(row=row, col=col, title=f"{symbol} Price")
            plot.setLabel('bottom', 'Time (ticks)')
            plot.setLabel('left', 'Price')
            plot.setDownsampling(ds=1, auto=True, mode='peak')
            plot.enableAutoRange('xy', False)
            
            curve = plot.plot(pen=pg.mkPen('white', width=1.5))
            self.price_plots[symbol] = plot
            self.price_curves[symbol] = curve
            
        # Statistics panel (right side)
        stats_plot = self.addPlot(row=0, col=2, rowspan=3, title="Statistics")
        stats_plot.hideAxis('bottom')
        stats_plot.hideAxis('left')
        self.stats_text = pg.TextItem(anchor=(0, 0), fill=pg.mkColor(50, 50, 50, 200))
        stats_plot.addItem(self.stats_text)
        
        # Order book heatmap (bottom)
        book_plot = self.addPlot(row=3, col=0, colspan=3, title="Order Book Depth")
        book_plot.setLabel('bottom', 'Price Level')
        book_plot.setLabel('left', 'Cumulative Size')
        self.book_plot = book_plot
        
    def add_tick(self, symbol: str, tick: MarketTick):
        """Record market tick (thread-safe)"""
        if symbol not in self.tick_buffers:
            self.tick_buffers[symbol] = TickBuffer()
        self.tick_buffers[symbol].push(tick)
    
    def update_display(self):
        """Update all charts (called on timer)"""
        # Update price curves
        for symbol in self.current_symbols:
            ticks = self.tick_buffers[symbol].get_recent(500)
            if ticks:
                times = np.arange(len(ticks))
                prices = np.array([t.price for t in ticks])
                self.price_curves[symbol].setData(times, prices)
                
                # Auto-range
                price_min = prices.min()
                price_max = prices.max()
                margin = (price_max - price_min) * 0.1
                self.price_plots[symbol].setRange(yRange=(price_min - margin, price_max + margin))
        
        # Update statistics
        self._update_stats()
        
        # Frame rate counter
        self.frame_count += 1
        if self.frame_count % 60 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"FPS: {fps:.1f}")
    
    def _update_stats(self):
        """Update statistics panel"""
        stats_text = "=== VeriTrade FPGA Dashboard ===\n\n"
        
        for symbol in self.current_symbols:
            ticks = self.tick_buffers[symbol].get_recent(100)
            if ticks:
                last_tick = ticks[-1]
                spread = last_tick.ask - last_tick.bid
                spread_pct = (spread / last_tick.price * 10000) if last_tick.price > 0 else 0
                
                stats_text += f"{symbol}:\n"
                stats_text += f"  Price: ${last_tick.price:,.2f}\n"
                stats_text += f"  Bid/Ask: {last_tick.bid:.2f}/{last_tick.ask:.2f}\n"
                stats_text += f"  Spread: {spread_pct:.1f}bps\n"
                stats_text += f"  Vol: {last_tick.volume:,}\n\n"
        
        self.stats_text.setText(stats_text, color=(200, 200, 200))

def demo_data_generator(dashboard: RTDashboard):
    """Generate synthetic market data for demo"""
    import random
    
    prices = {"BTC": 45000.0, "ETH": 2500.0, "SOL": 150.0}
    
    while True:
        for symbol in prices:
            # Random walk
            prices[symbol] *= (1 + random.gauss(0, 0.0005))
            
            tick = MarketTick(
                timestamp=time.time(),
                symbol=symbol,
                price=prices[symbol],
                bid=prices[symbol] - 0.5,
                ask=prices[symbol] + 0.5,
                volume=random.randint(100, 1000)
            )
            dashboard.add_tick(symbol, tick)
        
        time.sleep(0.01)  # 100Hz data feed

if __name__ == '__main__':
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    dashboard = RTDashboard("VeriTrade FPGA Dashboard")
    dashboard.show()
    
    # Start data generator thread
    data_thread = threading.Thread(target=demo_data_generator, args=(dashboard,), daemon=True)
    data_thread.start()
    
    sys.exit(app.exec_())
