#!/usr/bin/env python3
"""
Advanced FPGA Trading Performance Profiler
Comprehensive performance analysis and optimization tool for FPGA trading systems.
"""

import time
import numpy as np
import threading
import json
import csv
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_sim.fpga_core import FPGACore
from python_sim.market_data_simulator import MarketDataSimulator
from python_sim.strategies import ArbitrageStrategy, MarketMakingStrategy, TWAPStrategy

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    latency_ns: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    trade_count: int
    pnl: float
    risk_score: float
    error_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'latency_ns': self.latency_ns,
            'throughput': self.throughput,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'trade_count': self.trade_count,
            'pnl': self.pnl,
            'risk_score': self.risk_score,
            'error_count': self.error_count
        }

class PerformanceProfiler:
    """Advanced performance profiling and analysis"""
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.latency_buckets = defaultdict(int)
        self.throughput_stats = deque(maxlen=10000)
        self.error_log = []
        
        # Analysis parameters
        self.profiling_active = False
        self.profiling_thread = None
        self.start_time = None
        
        # FPGA system components
        self.fpga_core = None
        self.market_data_sim = None
        self.strategy = None
        
        # Performance thresholds
        self.latency_threshold_ns = 1000  # 1μs
        self.throughput_threshold = 100000  # 100k ops/sec
        self.memory_threshold_mb = 1024  # 1GB
        
    def initialize_system(self, clock_freq: float = 250e6, max_orders: int = 1000):
        """Initialize FPGA trading system"""
        self.fpga_core = FPGACore(clock_freq=clock_freq, max_orders=max_orders)
        self.market_data_sim = MarketDataSimulator(tick_rate=1000)
        self.strategy = ArbitrageStrategy()
        
        print(f"✅ FPGA system initialized:")
        print(f"   Clock frequency: {clock_freq/1e6:.1f} MHz")
        print(f"   Max orders: {max_orders}")
        
    def start_profiling(self, duration_seconds: Optional[int] = None):
        """Start performance profiling"""
        if self.profiling_active:
            print("⚠️  Profiling already active")
            return
        
        self.profiling_active = True
        self.start_time = datetime.now()
        self.metrics_history.clear()
        
        # Start profiling thread
        self.profiling_thread = threading.Thread(
            target=self._profiling_loop,
            args=(duration_seconds,)
        )
        self.profiling_thread.daemon = True
        self.profiling_thread.start()
        
        print(f"🚀 Performance profiling started")
        if duration_seconds:
            print(f"   Duration: {duration_seconds} seconds")
    
    def stop_profiling(self):
        """Stop performance profiling"""
        if not self.profiling_active:
            print("⚠️  Profiling not active")
            return
        
        self.profiling_active = False
        if self.profiling_thread:
            self.profiling_thread.join()
        
        duration = datetime.now() - self.start_time
        print(f"⏹️  Profiling stopped after {duration.total_seconds():.1f} seconds")
        
        # Generate report
        self._generate_report()
    
    def _profiling_loop(self, duration_seconds: Optional[int] = None):
        """Main profiling loop"""
        end_time = None
        if duration_seconds:
            end_time = datetime.now() + timedelta(seconds=duration_seconds)
        
        while self.profiling_active:
            if end_time and datetime.now() >= end_time:
                break
            
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Update statistics
                self._update_statistics(metrics)
                
                # Check for performance issues
                self._check_performance_alerts(metrics)
                
                # Sleep for next measurement
                time.sleep(0.001)  # 1ms interval
                
            except Exception as e:
                self.error_log.append({
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'type': 'profiling_error'
                })
        
        # Auto-stop if duration was specified
        if duration_seconds:
            self.profiling_active = False
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        if not self.fpga_core:
            raise RuntimeError("FPGA system not initialized")
        
        # Generate market data and process
        market_data = self.market_data_sim.generate_tick()
        
        # Measure processing latency
        start_time = time.perf_counter()
        self.fpga_core.process_market_data(market_data)
        signal = self.strategy.process_market_data(market_data)
        end_time = time.perf_counter()
        
        latency_ns = (end_time - start_time) * 1e9
        
        # Collect system metrics
        return PerformanceMetrics(
            timestamp=datetime.now(),
            latency_ns=latency_ns,
            throughput=self.fpga_core.get_throughput(),
            cpu_usage=self._get_cpu_usage(),
            memory_usage=self._get_memory_usage(),
            trade_count=self.fpga_core.get_trade_count(),
            pnl=self.fpga_core.get_pnl(),
            risk_score=self.fpga_core.get_risk_score(),
            error_count=len(self.error_log)
        )
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage (simplified)"""
        # In a real implementation, you'd use psutil or similar
        return np.random.uniform(10, 30)
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB (simplified)"""
        # In a real implementation, you'd use psutil or similar
        return np.random.uniform(100, 500)
    
    def _update_statistics(self, metrics: PerformanceMetrics):
        """Update running statistics"""
        # Latency buckets
        if metrics.latency_ns < 100:
            self.latency_buckets['<100ns'] += 1
        elif metrics.latency_ns < 1000:
            self.latency_buckets['100-1000ns'] += 1
        elif metrics.latency_ns < 10000:
            self.latency_buckets['1-10μs'] += 1
        else:
            self.latency_buckets['>10μs'] += 1
        
        # Throughput stats
        self.throughput_stats.append(metrics.throughput)
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance issues and generate alerts"""
        alerts = []
        
        if metrics.latency_ns > self.latency_threshold_ns:
            alerts.append(f"High latency: {metrics.latency_ns:.1f} ns")
        
        if metrics.throughput < self.throughput_threshold:
            alerts.append(f"Low throughput: {metrics.throughput:.0f} ops/sec")
        
        if metrics.memory_usage > self.memory_threshold_mb:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f} MB")
        
        if metrics.risk_score > 0.8:
            alerts.append(f"High risk score: {metrics.risk_score:.2f}")
        
        if alerts:
            print(f"⚠️  Performance alerts at {metrics.timestamp.strftime('%H:%M:%S')}:")
            for alert in alerts:
                print(f"   • {alert}")
    
    def _generate_report(self):
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            print("⚠️  No metrics collected")
            return
        
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate summary statistics
        summary = self._generate_summary_stats()
        
        # Save detailed CSV report
        self._save_csv_report(report_time)
        
        # Save JSON report
        self._save_json_report(report_time, summary)
        
        # Generate plots
        self._generate_plots(report_time)
        
        # Print summary
        self._print_summary(summary)
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.metrics_history:
            return {}
        
        latencies = [m.latency_ns for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history]
        pnls = [m.pnl for m in self.metrics_history]
        
        return {
            'total_samples': len(self.metrics_history),
            'duration_seconds': (self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp).total_seconds(),
            'latency_stats': {
                'min_ns': min(latencies),
                'max_ns': max(latencies),
                'mean_ns': np.mean(latencies),
                'median_ns': np.median(latencies),
                'p95_ns': np.percentile(latencies, 95),
                'p99_ns': np.percentile(latencies, 99),
                'std_ns': np.std(latencies)
            },
            'throughput_stats': {
                'min_ops': min(throughputs),
                'max_ops': max(throughputs),
                'mean_ops': np.mean(throughputs),
                'median_ops': np.median(throughputs),
                'std_ops': np.std(throughputs)
            },
            'trading_stats': {
                'total_trades': self.metrics_history[-1].trade_count,
                'total_pnl': sum(pnls),
                'max_risk_score': max(m.risk_score for m in self.metrics_history),
                'total_errors': len(self.error_log)
            },
            'latency_distribution': dict(self.latency_buckets)
        }
    
    def _save_csv_report(self, report_time: str):
        """Save detailed CSV report"""
        csv_path = self.output_dir / f"performance_report_{report_time}.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'latency_ns', 'throughput', 'cpu_usage', 
                         'memory_usage', 'trade_count', 'pnl', 'risk_score', 'error_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metrics in self.metrics_history:
                writer.writerow(asdict(metrics))
        
        print(f"📊 Detailed CSV report saved: {csv_path}")
    
    def _save_json_report(self, report_time: str, summary: Dict[str, Any]):
        """Save JSON summary report"""
        json_path = self.output_dir / f"performance_summary_{report_time}.json"
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_id': report_time,
                'profiler_version': '1.0.0'
            },
            'system_configuration': {
                'clock_frequency_mhz': self.fpga_core.clock_freq / 1e6 if self.fpga_core else None,
                'max_orders': self.fpga_core.max_orders if self.fpga_core else None,
                'strategy_type': type(self.strategy).__name__ if self.strategy else None
            },
            'performance_summary': summary,
            'error_log': [
                {
                    'timestamp': error['timestamp'].isoformat(),
                    'error': error['error'],
                    'type': error['type']
                }
                for error in self.error_log
            ]
        }
        
        with open(json_path, 'w') as jsonfile:
            json.dump(report_data, jsonfile, indent=2)
        
        print(f"📋 JSON summary saved: {json_path}")
    
    def _generate_plots(self, report_time: str):
        """Generate performance visualization plots"""
        if not self.metrics_history:
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Extract data for plotting
        timestamps = [m.timestamp for m in self.metrics_history]
        latencies = [m.latency_ns for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history]
        pnls = [m.pnl for m in self.metrics_history]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FPGA Trading System Performance Analysis', fontsize=16)
        
        # Latency over time
        axes[0, 0].plot(timestamps, latencies, color='red', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Processing Latency Over Time')
        axes[0, 0].set_ylabel('Latency (ns)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Latency histogram
        axes[0, 1].hist(latencies, bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Latency Distribution')
        axes[0, 1].set_xlabel('Latency (ns)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput over time
        axes[1, 0].plot(timestamps, throughputs, color='blue', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('System Throughput Over Time')
        axes[1, 0].set_ylabel('Throughput (ops/sec)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        axes[1, 1].plot(timestamps, cumulative_pnl, color='green', linewidth=2)
        axes[1, 1].set_title('Cumulative P&L')
        axes[1, 1].set_ylabel('P&L ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"performance_plots_{report_time}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance plots saved: {plot_path}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print performance summary to console"""
        print("\n" + "="*60)
        print("🎯 FPGA TRADING SYSTEM PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"📊 Total Samples: {summary['total_samples']:,}")
        print(f"⏱️  Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"📈 Total Trades: {summary['trading_stats']['total_trades']:,}")
        print(f"💰 Total P&L: ${summary['trading_stats']['total_pnl']:.2f}")
        
        print(f"\n🚀 LATENCY STATISTICS:")
        lat_stats = summary['latency_stats']
        print(f"   Average: {lat_stats['mean_ns']:.1f} ns")
        print(f"   Median:  {lat_stats['median_ns']:.1f} ns")
        print(f"   P95:     {lat_stats['p95_ns']:.1f} ns")
        print(f"   P99:     {lat_stats['p99_ns']:.1f} ns")
        print(f"   Max:     {lat_stats['max_ns']:.1f} ns")
        
        print(f"\n⚡ THROUGHPUT STATISTICS:")
        thr_stats = summary['throughput_stats']
        print(f"   Average: {thr_stats['mean_ops']:.0f} ops/sec")
        print(f"   Peak:    {thr_stats['max_ops']:.0f} ops/sec")
        
        print(f"\n📋 LATENCY DISTRIBUTION:")
        for bucket, count in summary['latency_distribution'].items():
            percentage = (count / summary['total_samples']) * 100
            print(f"   {bucket}: {count:,} ({percentage:.1f}%)")
        
        if summary['trading_stats']['total_errors'] > 0:
            print(f"\n⚠️  ERRORS: {summary['trading_stats']['total_errors']}")
        
        print("="*60)

def main():
    """Main function for running performance profiling"""
    print("🚀 FPGA Trading Performance Profiler")
    print("=====================================")
    
    # Initialize profiler
    profiler = PerformanceProfiler()
    
    # Initialize FPGA system
    profiler.initialize_system(clock_freq=250e6, max_orders=1000)
    
    try:
        # Start profiling
        profiler.start_profiling(duration_seconds=30)  # 30 second test
        
        # Wait for completion
        while profiler.profiling_active:
            time.sleep(1)
        
        print("✅ Profiling completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Profiling interrupted by user")
        profiler.stop_profiling()
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        profiler.stop_profiling()

if __name__ == "__main__":
    main()
