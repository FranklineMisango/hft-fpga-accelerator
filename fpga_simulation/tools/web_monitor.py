#!/usr/bin/env python3
"""
Web-based FPGA Trading System Monitor
Real-time monitoring interface for FPGA trading accelerator performance.
"""

import sys
import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_sim.fpga_core import FPGACore
from python_sim.market_data_simulator import MarketDataSimulator
from python_sim.strategies import ArbitrageStrategy, MarketMakingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebMonitor:
    """Web-based monitoring system for FPGA trading"""
    
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'fpga-trading-monitor-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize FPGA system
        self.fpga_core = FPGACore(clock_freq=250e6, max_orders=1000)
        self.market_data_sim = MarketDataSimulator(tick_rate=1000)
        self.strategy = ArbitrageStrategy()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.clients_connected = 0
        
        # Data storage
        self.recent_metrics = []
        self.max_metrics = 1000
        
        # Setup routes
        self._setup_routes()
        self._setup_websocket_events()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main monitoring dashboard"""
            return render_template('monitor.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get current system status"""
            return jsonify({
                'fpga_core': {
                    'running': self.monitoring_active,
                    'clock_freq_mhz': self.fpga_core.clock_freq / 1e6,
                    'max_orders': self.fpga_core.max_orders
                },
                'market_data': {
                    'connected': True,
                    'tick_rate': self.market_data_sim.tick_rate
                },
                'strategy': {
                    'type': type(self.strategy).__name__,
                    'active': self.monitoring_active
                },
                'monitoring': {
                    'active': self.monitoring_active,
                    'clients_connected': self.clients_connected,
                    'metrics_count': len(self.recent_metrics)
                }
            })
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get recent performance metrics"""
            return jsonify({
                'metrics': self.recent_metrics[-100:],  # Last 100 metrics
                'count': len(self.recent_metrics)
            })
        
        @self.app.route('/api/start', methods=['POST'])
        def start_monitoring():
            """Start monitoring"""
            if not self.monitoring_active:
                self.start_monitoring()
                return jsonify({'status': 'started'})
            return jsonify({'status': 'already_running'})
        
        @self.app.route('/api/stop', methods=['POST'])
        def stop_monitoring():
            """Stop monitoring"""
            if self.monitoring_active:
                self.stop_monitoring()
                return jsonify({'status': 'stopped'})
            return jsonify({'status': 'not_running'})
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset_metrics():
            """Reset metrics"""
            self.recent_metrics.clear()
            return jsonify({'status': 'reset'})
    
    def _setup_websocket_events(self):
        """Setup WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.clients_connected += 1
            logger.info(f"Client connected. Total: {self.clients_connected}")
            emit('status', {'message': 'Connected to FPGA monitor'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.clients_connected -= 1
            logger.info(f"Client disconnected. Total: {self.clients_connected}")
        
        @self.socketio.on('request_metrics')
        def handle_metrics_request():
            """Handle metrics request"""
            emit('metrics_update', {
                'metrics': self.recent_metrics[-50:],
                'timestamp': datetime.now().isoformat()
            })
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("FPGA monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("FPGA monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Generate market data
                market_data = self.market_data_sim.generate_tick()
                
                # Process with FPGA
                start_time = time.perf_counter()
                self.fpga_core.process_market_data(market_data)
                signal = self.strategy.process_market_data(market_data)
                end_time = time.perf_counter()
                
                # Calculate metrics
                processing_time_ns = (end_time - start_time) * 1e9
                
                # Create metrics record
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'latency_ns': processing_time_ns,
                    'throughput': self.fpga_core.get_throughput(),
                    'trade_count': self.fpga_core.get_trade_count(),
                    'pnl': self.fpga_core.get_pnl(),
                    'risk_score': self.fpga_core.get_risk_score(),
                    'price': market_data['price'],
                    'volume': market_data['volume'],
                    'signal': signal if signal else {}
                }
                
                # Store metrics
                self.recent_metrics.append(metrics)
                
                # Limit metrics storage
                if len(self.recent_metrics) > self.max_metrics:
                    self.recent_metrics.pop(0)
                
                # Broadcast to connected clients
                if self.clients_connected > 0:
                    self.socketio.emit('metrics_update', {
                        'metrics': [metrics],
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Sleep to control update rate
                time.sleep(0.01)  # 100 Hz update rate
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1)
    
    def run(self, debug=False):
        """Run the web monitoring server"""
        logger.info(f"Starting FPGA Trading Monitor at http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)

# Create monitoring HTML template
def create_monitor_template():
    """Create the monitoring HTML template"""
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FPGA Trading Monitor</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .btn-start {
            background: #28a745;
            color: white;
        }
        .btn-stop {
            background: #dc3545;
            color: white;
        }
        .btn-reset {
            background: #ffc107;
            color: black;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .status-panel {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
        }
        .status-good {
            color: #28a745;
        }
        .status-bad {
            color: #dc3545;
        }
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        }
        .connected {
            background: #28a745;
            color: white;
        }
        .disconnected {
            background: #dc3545;
            color: white;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Disconnected</div>
    
    <div class="container">
        <div class="header">
            <h1>🚀 FPGA Trading Monitor</h1>
            <p>Real-time hardware-accelerated trading performance</p>
        </div>
        
        <div class="controls">
            <button class="btn btn-start" onclick="startMonitoring()">Start Monitoring</button>
            <button class="btn btn-stop" onclick="stopMonitoring()">Stop Monitoring</button>
            <button class="btn btn-reset" onclick="resetMetrics()">Reset Metrics</button>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="latency">0.0</div>
                <div class="metric-label">Latency (ns)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="throughput">0</div>
                <div class="metric-label">Throughput (ops/s)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="trades">0</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="pnl">$0.00</div>
                <div class="metric-label">P&L</div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart">
                <div id="latencyChart"></div>
            </div>
            <div class="chart">
                <div id="throughputChart"></div>
            </div>
            <div class="chart">
                <div id="priceChart"></div>
            </div>
            <div class="chart">
                <div id="pnlChart"></div>
            </div>
        </div>
        
        <div class="status-panel">
            <h3>System Status</h3>
            <div class="status-item">
                <span>FPGA Core</span>
                <span id="fpgaStatus" class="status-bad">Stopped</span>
            </div>
            <div class="status-item">
                <span>Market Data</span>
                <span id="marketStatus" class="status-good">Connected</span>
            </div>
            <div class="status-item">
                <span>Strategy Engine</span>
                <span id="strategyStatus" class="status-good">Active</span>
            </div>
            <div class="status-item">
                <span>Risk Controls</span>
                <span id="riskStatus" class="status-good">Normal</span>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Data storage
        let metricsData = [];
        const maxDataPoints = 100;
        
        // Connection status
        socket.on('connect', function() {
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'connection-status connected';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
        });
        
        // Metrics updates
        socket.on('metrics_update', function(data) {
            if (data.metrics && data.metrics.length > 0) {
                metricsData = metricsData.concat(data.metrics);
                
                // Limit data points
                if (metricsData.length > maxDataPoints) {
                    metricsData = metricsData.slice(-maxDataPoints);
                }
                
                updateMetrics();
                updateCharts();
            }
        });
        
        // Update metric cards
        function updateMetrics() {
            if (metricsData.length === 0) return;
            
            const latest = metricsData[metricsData.length - 1];
            
            document.getElementById('latency').textContent = latest.latency_ns.toFixed(1);
            document.getElementById('throughput').textContent = latest.throughput.toFixed(0);
            document.getElementById('trades').textContent = latest.trade_count.toLocaleString();
            document.getElementById('pnl').textContent = '$' + latest.pnl.toFixed(2);
            
            // Update status
            document.getElementById('fpgaStatus').textContent = 'Running';
            document.getElementById('fpgaStatus').className = 'status-good';
            
            // Risk status
            const riskElement = document.getElementById('riskStatus');
            if (latest.risk_score < 0.5) {
                riskElement.textContent = 'Normal';
                riskElement.className = 'status-good';
            } else if (latest.risk_score < 0.8) {
                riskElement.textContent = 'Elevated';
                riskElement.className = 'status-bad';
            } else {
                riskElement.textContent = 'High Risk';
                riskElement.className = 'status-bad';
            }
        }
        
        // Update charts
        function updateCharts() {
            if (metricsData.length === 0) return;
            
            const timestamps = metricsData.map(m => m.timestamp);
            const latencies = metricsData.map(m => m.latency_ns);
            const throughputs = metricsData.map(m => m.throughput);
            const prices = metricsData.map(m => m.price);
            const pnls = metricsData.map(m => m.pnl);
            
            // Latency chart
            Plotly.newPlot('latencyChart', [{
                x: timestamps,
                y: latencies,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Latency',
                line: {color: 'rgb(255, 99, 132)'}
            }], {
                title: 'Processing Latency',
                xaxis: {title: 'Time'},
                yaxis: {title: 'Latency (ns)'},
                margin: {l: 60, r: 20, t: 40, b: 40},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0.1)',
                font: {color: 'white'}
            });
            
            // Throughput chart
            Plotly.newPlot('throughputChart', [{
                x: timestamps,
                y: throughputs,
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                name: 'Throughput',
                line: {color: 'rgb(54, 162, 235)'}
            }], {
                title: 'System Throughput',
                xaxis: {title: 'Time'},
                yaxis: {title: 'Ops/sec'},
                margin: {l: 60, r: 20, t: 40, b: 40},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0.1)',
                font: {color: 'white'}
            });
            
            // Price chart
            Plotly.newPlot('priceChart', [{
                x: timestamps,
                y: prices,
                type: 'scatter',
                mode: 'lines',
                name: 'Price',
                line: {color: 'rgb(255, 206, 86)'}
            }], {
                title: 'Market Price',
                xaxis: {title: 'Time'},
                yaxis: {title: 'Price ($)'},
                margin: {l: 60, r: 20, t: 40, b: 40},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0.1)',
                font: {color: 'white'}
            });
            
            // P&L chart
            const cumulativePnl = [];
            let sum = 0;
            for (let pnl of pnls) {
                sum += pnl;
                cumulativePnl.push(sum);
            }
            
            Plotly.newPlot('pnlChart', [{
                x: timestamps,
                y: cumulativePnl,
                type: 'scatter',
                mode: 'lines',
                name: 'Cumulative P&L',
                line: {color: 'rgb(75, 192, 192)'}
            }], {
                title: 'Cumulative P&L',
                xaxis: {title: 'Time'},
                yaxis: {title: 'P&L ($)'},
                margin: {l: 60, r: 20, t: 40, b: 40},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0.1)',
                font: {color: 'white'}
            });
        }
        
        // Control functions
        function startMonitoring() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => console.log('Started:', data));
        }
        
        function stopMonitoring() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => console.log('Stopped:', data));
        }
        
        function resetMetrics() {
            fetch('/api/reset', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    metricsData = [];
                    updateMetrics();
                    updateCharts();
                    console.log('Reset:', data);
                });
        }
        
        // Initialize charts
        updateCharts();
        
        // Request initial metrics
        socket.emit('request_metrics');
    </script>
</body>
</html>"""
    
    with open(os.path.join(template_dir, 'monitor.html'), 'w') as f:
        f.write(template_content)

def main():
    """Main function to run the web monitor"""
    print("🚀 Starting FPGA Trading Web Monitor")
    print("====================================")
    
    # Create template
    create_monitor_template()
    
    # Initialize and run monitor
    monitor = WebMonitor(host='127.0.0.1', port=5000)
    
    try:
        monitor.run(debug=False)
    except KeyboardInterrupt:
        print("\n⏹️  Monitor stopped by user")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
