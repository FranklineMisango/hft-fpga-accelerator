#!/usr/bin/env python3
"""
Real-time FPGA Trading System Dashboard
Interactive visualization for monitoring trading performance, latency, and system metrics.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import asyncio
import threading
import time
import json
from datetime import datetime, timedelta
from collections import deque
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_sim.fpga_core import FPGACore
from python_sim.market_data_simulator import MarketDataSimulator
from python_sim.strategies import ArbitrageStrategy, MarketMakingStrategy

class TradingDashboard:
    def __init__(self):
        self.max_points = 1000
        self.latency_history = deque(maxlen=self.max_points)
        self.throughput_history = deque(maxlen=self.max_points)
        self.price_history = deque(maxlen=self.max_points)
        self.volume_history = deque(maxlen=self.max_points)
        self.pnl_history = deque(maxlen=self.max_points)
        self.trades_history = deque(maxlen=self.max_points)
        self.risk_metrics = deque(maxlen=self.max_points)
        self.timestamps = deque(maxlen=self.max_points)
        
        # Initialize FPGA simulation components
        self.fpga_core = FPGACore(clock_freq=250e6, max_orders=1000)
        self.market_data_sim = MarketDataSimulator(tick_rate=1000)
        self.strategy = ArbitrageStrategy()
        
        # Simulation state
        self.running = False
        self.simulation_thread = None
        
    def start_simulation(self):
        """Start the background simulation thread"""
        if not self.running:
            self.running = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the simulation thread"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join()
    
    def _simulation_loop(self):
        """Main simulation loop running in background"""
        while self.running:
            try:
                # Generate market data
                market_data = self.market_data_sim.generate_tick()
                
                # Process through FPGA core
                self.fpga_core.process_market_data(market_data)
                
                # Execute strategy
                signal = self.strategy.process_market_data(market_data)
                
                # Update metrics
                current_time = datetime.now()
                self._update_metrics(market_data, signal, current_time)
                
                # Sleep to simulate real-time processing
                time.sleep(0.001)  # 1ms interval
                
            except Exception as e:
                st.error(f"Simulation error: {e}")
                break
    
    def _update_metrics(self, market_data, signal, timestamp):
        """Update dashboard metrics"""
        # Latency metrics
        processing_latency = self.fpga_core.get_latency_stats()['avg']
        self.latency_history.append(processing_latency * 1e9)  # Convert to ns
        
        # Throughput metrics
        throughput = self.fpga_core.get_throughput()
        self.throughput_history.append(throughput)
        
        # Price and volume
        self.price_history.append(market_data['price'])
        self.volume_history.append(market_data['volume'])
        
        # P&L calculation (simplified)
        pnl = self.fpga_core.get_pnl()
        self.pnl_history.append(pnl)
        
        # Trade count
        trade_count = self.fpga_core.get_trade_count()
        self.trades_history.append(trade_count)
        
        # Risk metrics
        risk_score = self.fpga_core.get_risk_score()
        self.risk_metrics.append(risk_score)
        
        # Timestamp
        self.timestamps.append(timestamp)

def create_latency_chart(dashboard):
    """Create real-time latency chart"""
    fig = go.Figure()
    
    if dashboard.latency_history:
        fig.add_trace(go.Scatter(
            x=list(dashboard.timestamps),
            y=list(dashboard.latency_history),
            mode='lines+markers',
            name='Latency (ns)',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title='Real-time Processing Latency',
        xaxis_title='Time',
        yaxis_title='Latency (ns)',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_throughput_chart(dashboard):
    """Create throughput monitoring chart"""
    fig = go.Figure()
    
    if dashboard.throughput_history:
        fig.add_trace(go.Scatter(
            x=list(dashboard.timestamps),
            y=list(dashboard.throughput_history),
            mode='lines',
            name='Throughput',
            fill='tozeroy',
            line=dict(color='#4ecdc4', width=2)
        ))
    
    fig.update_layout(
        title='System Throughput',
        xaxis_title='Time',
        yaxis_title='Orders/sec',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_price_volume_chart(dashboard):
    """Create combined price and volume chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price', 'Volume'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    if dashboard.price_history:
        fig.add_trace(
            go.Scatter(
                x=list(dashboard.timestamps),
                y=list(dashboard.price_history),
                mode='lines',
                name='Price',
                line=dict(color='#45b7d1', width=2)
            ),
            row=1, col=1
        )
    
    if dashboard.volume_history:
        fig.add_trace(
            go.Bar(
                x=list(dashboard.timestamps),
                y=list(dashboard.volume_history),
                name='Volume',
                marker_color='#96ceb4'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_pnl_chart(dashboard):
    """Create P&L tracking chart"""
    fig = go.Figure()
    
    if dashboard.pnl_history:
        cumulative_pnl = np.cumsum(list(dashboard.pnl_history))
        colors = ['green' if pnl >= 0 else 'red' for pnl in cumulative_pnl]
        
        fig.add_trace(go.Scatter(
            x=list(dashboard.timestamps),
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#f39c12', width=3),
            fill='tozeroy'
        ))
    
    fig.update_layout(
        title='Cumulative P&L',
        xaxis_title='Time',
        yaxis_title='P&L ($)',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_system_metrics(dashboard):
    """Create system metrics cards"""
    if not dashboard.latency_history:
        return None, None, None, None
    
    # Calculate metrics
    avg_latency = np.mean(list(dashboard.latency_history))
    p99_latency = np.percentile(list(dashboard.latency_history), 99)
    current_throughput = list(dashboard.throughput_history)[-1] if dashboard.throughput_history else 0
    total_trades = list(dashboard.trades_history)[-1] if dashboard.trades_history else 0
    
    return avg_latency, p99_latency, current_throughput, total_trades

def main():
    st.set_page_config(
        page_title="FPGA Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 FPGA Trading System Dashboard")
    st.markdown("Real-time monitoring of hardware-accelerated trading performance")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = TradingDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Controls")
        
        if st.button("Start Simulation", type="primary"):
            dashboard.start_simulation()
            st.success("Simulation started!")
        
        if st.button("Stop Simulation"):
            dashboard.stop_simulation()
            st.success("Simulation stopped!")
        
        st.header("System Configuration")
        
        # FPGA settings
        clock_freq = st.slider("Clock Frequency (MHz)", 100, 500, 250)
        max_orders = st.slider("Max Orders", 100, 5000, 1000)
        
        # Market data settings
        tick_rate = st.slider("Tick Rate (Hz)", 100, 10000, 1000)
        volatility = st.slider("Market Volatility", 0.1, 5.0, 1.0)
        
        # Strategy selection
        strategy_type = st.selectbox(
            "Trading Strategy",
            ["Arbitrage", "Market Making", "TWAP", "VWAP"]
        )
        
        st.header("Risk Controls")
        position_limit = st.number_input("Position Limit", value=1000)
        max_exposure = st.number_input("Max Exposure ($)", value=100000)
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # System metrics
    avg_latency, p99_latency, current_throughput, total_trades = create_system_metrics(dashboard)
    
    if avg_latency is not None:
        with col1:
            st.metric("Avg Latency", f"{avg_latency:.1f} ns", delta=None)
        
        with col2:
            st.metric("P99 Latency", f"{p99_latency:.1f} ns", delta=None)
        
        with col3:
            st.metric("Throughput", f"{current_throughput:.0f} ops/s", delta=None)
        
        with col4:
            st.metric("Total Trades", f"{total_trades:,}", delta=None)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        latency_chart = create_latency_chart(dashboard)
        st.plotly_chart(latency_chart, use_container_width=True)
        
        pnl_chart = create_pnl_chart(dashboard)
        st.plotly_chart(pnl_chart, use_container_width=True)
    
    with col2:
        throughput_chart = create_throughput_chart(dashboard)
        st.plotly_chart(throughput_chart, use_container_width=True)
        
        price_volume_chart = create_price_volume_chart(dashboard)
        st.plotly_chart(price_volume_chart, use_container_width=True)
    
    # Status and logs
    st.header("System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.subheader("FPGA Core Status")
        if dashboard.running:
            st.success("🟢 Running")
        else:
            st.error("🔴 Stopped")
        
        st.subheader("Market Data")
        if dashboard.market_data_sim:
            st.success("🟢 Connected")
        else:
            st.error("🔴 Disconnected")
    
    with status_col2:
        st.subheader("Strategy Engine")
        if dashboard.strategy:
            st.success("🟢 Active")
        else:
            st.error("🔴 Inactive")
        
        st.subheader("Risk Controls")
        if dashboard.risk_metrics:
            current_risk = list(dashboard.risk_metrics)[-1]
            if current_risk < 0.5:
                st.success("🟢 Normal")
            elif current_risk < 0.8:
                st.warning("🟡 Elevated")
            else:
                st.error("🔴 High Risk")
        else:
            st.info("📊 Monitoring")
    
    # Auto-refresh
    if dashboard.running:
        time.sleep(0.1)
        st.rerun()

if __name__ == "__main__":
    main()
