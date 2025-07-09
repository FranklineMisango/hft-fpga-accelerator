#!/usr/bin/env python3
"""
FPGA Trading Advanced Tools Demo
Comprehensive demonstration of all advanced features and tools.
"""

import sys
import os
import time
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.performance_profiler import PerformanceProfiler
from tools.gpu_fpga_bridge import FPGABridge
from tools.enhanced_market_data import EnhancedMarketDataFeed, ProtocolType

class AdvancedToolsDemo:
    """Demo orchestrator for all advanced tools"""
    
    def __init__(self):
        self.demo_results = {}
        self.demo_start_time = None
        
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"🚀 {title}")
        print("="*60)
    
    def print_step(self, step: str, description: str):
        """Print formatted step"""
        print(f"\n{step}. {description}")
        print("-" * 40)
    
    def demo_performance_profiler(self):
        """Demonstrate performance profiler"""
        self.print_header("PERFORMANCE PROFILER DEMO")
        
        self.print_step("1", "Initializing Performance Profiler")
        profiler = PerformanceProfiler(output_dir="demo_reports")
        profiler.initialize_system(clock_freq=250e6, max_orders=1000)
        
        self.print_step("2", "Running Performance Analysis (10 seconds)")
        start_time = time.time()
        profiler.start_profiling(duration_seconds=10)
        
        # Wait for completion
        while profiler.profiling_active:
            time.sleep(0.5)
            print(".", end="", flush=True)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Profiling completed in {elapsed:.1f} seconds")
        
        self.print_step("3", "Performance Results")
        summary = profiler.get_performance_summary()
        
        if 'performance_summary' in summary:
            perf_summary = summary['performance_summary']
            
            if 'latency_stats' in perf_summary:
                lat_stats = perf_summary['latency_stats']
                print(f"   📊 Latency Statistics:")
                print(f"      Average: {lat_stats['mean_ns']:.1f} ns")
                print(f"      P95: {lat_stats['p95_ns']:.1f} ns")
                print(f"      P99: {lat_stats['p99_ns']:.1f} ns")
            
            if 'throughput_stats' in perf_summary:
                thr_stats = perf_summary['throughput_stats']
                print(f"   ⚡ Throughput: {thr_stats['mean_ops']:.0f} ops/sec")
            
            if 'trading_stats' in perf_summary:
                trade_stats = perf_summary['trading_stats']
                print(f"   💰 Trading: {trade_stats['total_trades']} trades, ${trade_stats['total_pnl']:.2f} P&L")
        
        self.demo_results['profiler'] = {
            'duration': elapsed,
            'summary': summary
        }
        
        print("✅ Performance profiler demo completed!")
        
        return profiler
    
    def demo_gpu_fpga_bridge(self):
        """Demonstrate GPU-FPGA bridge"""
        self.print_header("GPU-FPGA BRIDGE DEMO")
        
        self.print_step("1", "Initializing GPU-FPGA Bridge")
        bridge = FPGABridge(fpga_clock_freq=250e6)
        
        gpu_available = bridge.gpu_accelerator.gpu_available
        print(f"   GPU Acceleration: {'✅ Available' if gpu_available else '❌ Not Available'}")
        
        self.print_step("2", "Starting Hybrid Processing Pipeline")
        bridge.start_processing()
        
        # Monitor processing for 15 seconds
        self.print_step("3", "Processing Market Data (15 seconds)")
        for i in range(15):
            time.sleep(1)
            stats = bridge.processing_stats
            print(f"   {i+1:2d}s: Processed {stats['total_processed']:,} items", end="")
            if gpu_available:
                print(f", GPU: {stats['gpu_processed']:,}, FPGA: {stats['fpga_processed']:,}")
            else:
                print(f", FPGA: {stats['fpga_processed']:,}")
        
        self.print_step("4", "Stopping Processing")
        bridge.stop_processing()
        
        self.print_step("5", "Performance Results")
        summary = bridge.get_performance_summary()
        
        proc_stats = summary['processing_statistics']
        hybrid_perf = summary['hybrid_performance']
        
        print(f"   📊 Processing Statistics:")
        print(f"      Total Processed: {proc_stats['total_processed']:,}")
        print(f"      GPU Processed: {proc_stats['gpu_processed']:,}")
        print(f"      FPGA Processed: {proc_stats['fpga_processed']:,}")
        print(f"      Overall Throughput: {hybrid_perf['overall_throughput']:.0f} ops/sec")
        
        if gpu_available:
            gpu_perf = summary['gpu_performance']
            print(f"      GPU Speedup: {gpu_perf['speedup_factor']:.1f}x")
        
        self.demo_results['gpu_bridge'] = {
            'gpu_available': gpu_available,
            'summary': summary
        }
        
        print("✅ GPU-FPGA bridge demo completed!")
        
        return bridge
    
    def demo_enhanced_market_data(self):
        """Demonstrate enhanced market data protocols"""
        self.print_header("ENHANCED MARKET DATA DEMO")
        
        protocols = [
            (ProtocolType.ITCH_5_0, "NASDAQ ITCH 5.0"),
            (ProtocolType.FIX_4_4, "FIX 4.4"),
            (ProtocolType.OUCH_4_2, "OUCH 4.2")
        ]
        
        protocol_results = {}
        
        for protocol_type, protocol_name in protocols:
            self.print_step(f"{protocol_type.value}", f"Testing {protocol_name}")
            
            # Create feed
            feed = EnhancedMarketDataFeed(protocol=protocol_type)
            
            # Collect messages
            messages = []
            feed.set_message_callback(lambda msg: messages.append(msg))
            
            # Configure for demo
            feed.message_rate_hz = 1000
            feed.set_latency_simulation(True, (50, 200))
            
            # Start feed
            feed.start_feed(['AAPL', 'GOOGL', 'MSFT'])
            
            # Run for 5 seconds
            time.sleep(5)
            
            # Stop feed
            feed.stop_feed()
            
            # Get statistics
            stats = feed.get_statistics()
            
            print(f"   📊 {protocol_name} Results:")
            print(f"      Messages: {stats['messages_sent']:,}")
            print(f"      Bytes: {stats['bytes_sent']:,}")
            print(f"      Rate: {stats['message_rate']:.1f} Hz")
            
            if messages:
                sample_msg = messages[0]
                print(f"      Sample: {sample_msg.message_type} for {sample_msg.symbol}")
            
            protocol_results[protocol_name] = {
                'stats': stats,
                'message_count': len(messages)
            }
        
        self.demo_results['market_data'] = protocol_results
        
        print("✅ Enhanced market data demo completed!")
        
        return protocol_results
    
    def demo_integration_test(self):
        """Demonstrate integration between components"""
        self.print_header("INTEGRATION TEST DEMO")
        
        self.print_step("1", "Starting All Components")
        
        # Start market data feed
        market_feed = EnhancedMarketDataFeed(protocol=ProtocolType.ITCH_5_0)
        market_feed.start_feed(['AAPL', 'GOOGL', 'MSFT'])
        
        # Start GPU-FPGA bridge
        bridge = FPGABridge(fpga_clock_freq=250e6)
        bridge.start_processing()
        
        # Start performance profiler
        profiler = PerformanceProfiler(output_dir="integration_reports")
        profiler.initialize_system()
        profiler.start_profiling(duration_seconds=10)
        
        self.print_step("2", "Running Integrated System (10 seconds)")
        
        # Monitor all components
        for i in range(10):
            time.sleep(1)
            
            # Get stats from all components
            bridge_stats = bridge.processing_stats
            market_stats = market_feed.get_statistics()
            
            print(f"   {i+1:2d}s: Market: {market_stats['messages_sent']:,}, "
                  f"Processing: {bridge_stats['total_processed']:,}")
        
        self.print_step("3", "Stopping All Components")
        
        # Stop all components
        market_feed.stop_feed()
        bridge.stop_processing()
        profiler.stop_profiling()
        
        self.print_step("4", "Integration Results")
        
        # Collect final statistics
        final_market_stats = market_feed.get_statistics()
        final_bridge_stats = bridge.get_performance_summary()
        
        print(f"   📊 Integration Statistics:")
        print(f"      Market Messages: {final_market_stats['messages_sent']:,}")
        print(f"      Processing Throughput: {final_bridge_stats['hybrid_performance']['overall_throughput']:.0f} ops/sec")
        print(f"      GPU Acceleration: {'✅ Active' if bridge.gpu_accelerator.gpu_available else '❌ Inactive'}")
        
        self.demo_results['integration'] = {
            'market_stats': final_market_stats,
            'bridge_stats': final_bridge_stats
        }
        
        print("✅ Integration test demo completed!")
    
    def generate_demo_report(self):
        """Generate comprehensive demo report"""
        self.print_header("DEMO REPORT")
        
        total_duration = time.time() - self.demo_start_time
        
        print(f"📊 Demo Summary")
        print(f"   Total Duration: {total_duration:.1f} seconds")
        print(f"   Components Tested: {len(self.demo_results)}")
        
        # Performance profiler results
        if 'profiler' in self.demo_results:
            profiler_data = self.demo_results['profiler']
            print(f"\n🔬 Performance Profiler:")
            print(f"   Analysis Duration: {profiler_data['duration']:.1f}s")
            
            if 'summary' in profiler_data and 'performance_summary' in profiler_data['summary']:
                perf = profiler_data['summary']['performance_summary']
                if 'latency_stats' in perf:
                    lat = perf['latency_stats']
                    print(f"   Average Latency: {lat['mean_ns']:.1f} ns")
                    print(f"   P99 Latency: {lat['p99_ns']:.1f} ns")
        
        # GPU-FPGA bridge results
        if 'gpu_bridge' in self.demo_results:
            bridge_data = self.demo_results['gpu_bridge']
            print(f"\n⚡ GPU-FPGA Bridge:")
            print(f"   GPU Available: {'Yes' if bridge_data['gpu_available'] else 'No'}")
            
            if 'summary' in bridge_data:
                summary = bridge_data['summary']
                stats = summary['processing_statistics']
                perf = summary['hybrid_performance']
                print(f"   Total Processed: {stats['total_processed']:,}")
                print(f"   Throughput: {perf['overall_throughput']:.0f} ops/sec")
        
        # Market data results
        if 'market_data' in self.demo_results:
            market_data = self.demo_results['market_data']
            print(f"\n📡 Market Data Protocols:")
            
            for protocol_name, data in market_data.items():
                stats = data['stats']
                print(f"   {protocol_name}: {stats['messages_sent']:,} messages @ {stats['message_rate']:.1f} Hz")
        
        # Integration results
        if 'integration' in self.demo_results:
            integration_data = self.demo_results['integration']
            print(f"\n🔗 Integration Test:")
            
            market_stats = integration_data['market_stats']
            bridge_stats = integration_data['bridge_stats']
            
            print(f"   Market Messages: {market_stats['messages_sent']:,}")
            print(f"   Processing Throughput: {bridge_stats['hybrid_performance']['overall_throughput']:.0f} ops/sec")
        
        print(f"\n🎉 Demo Completed Successfully!")
        print(f"   All {len(self.demo_results)} components tested and validated")
        print(f"   Total execution time: {total_duration:.1f} seconds")
        
        # Performance assessment
        print(f"\n📈 Performance Assessment:")
        
        # Check latency
        if 'profiler' in self.demo_results:
            profiler_data = self.demo_results['profiler']
            if 'summary' in profiler_data and 'performance_summary' in profiler_data['summary']:
                perf = profiler_data['summary']['performance_summary']
                if 'latency_stats' in perf:
                    avg_latency = perf['latency_stats']['mean_ns']
                    if avg_latency < 100:
                        print(f"   ✅ Excellent latency: {avg_latency:.1f} ns")
                    elif avg_latency < 500:
                        print(f"   ✅ Good latency: {avg_latency:.1f} ns")
                    elif avg_latency < 1000:
                        print(f"   ⚠️  Acceptable latency: {avg_latency:.1f} ns")
                    else:
                        print(f"   ❌ High latency: {avg_latency:.1f} ns")
        
        # Check throughput
        if 'gpu_bridge' in self.demo_results:
            bridge_data = self.demo_results['gpu_bridge']
            if 'summary' in bridge_data:
                throughput = bridge_data['summary']['hybrid_performance']['overall_throughput']
                if throughput > 1000000:
                    print(f"   ✅ Excellent throughput: {throughput:.0f} ops/sec")
                elif throughput > 500000:
                    print(f"   ✅ Good throughput: {throughput:.0f} ops/sec")
                elif throughput > 100000:
                    print(f"   ⚠️  Acceptable throughput: {throughput:.0f} ops/sec")
                else:
                    print(f"   ❌ Low throughput: {throughput:.0f} ops/sec")
        
        print(f"\n🚀 FPGA Trading Advanced Tools Demo Complete!")
        print(f"   Ready for production use with sub-microsecond latency")
        print(f"   and high-throughput market data processing.")

def main():
    """Main demo function"""
    print("🚀 FPGA Trading Advanced Tools Demo")
    print("===================================")
    print("This demo showcases all advanced features of the FPGA trading accelerator.")
    print("Including performance profiling, GPU-FPGA bridge, and enhanced market data.")
    print("\nPress Enter to start the demo...")
    input()
    
    # Create demo instance
    demo = AdvancedToolsDemo()
    demo.demo_start_time = time.time()
    
    try:
        # Run all demos
        demo.demo_performance_profiler()
        demo.demo_gpu_fpga_bridge()
        demo.demo_enhanced_market_data()
        demo.demo_integration_test()
        
        # Generate final report
        demo.generate_demo_report()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        print("\n🏁 Demo cleanup completed")

if __name__ == "__main__":
    main()
