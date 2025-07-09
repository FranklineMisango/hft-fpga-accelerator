#!/usr/bin/env python3
"""
Advanced Tools Test Suite
Comprehensive testing for FPGA trading accelerator advanced tools.
"""

import sys
import os
import time
import unittest
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tools to test
from tools.performance_profiler import PerformanceProfiler, PerformanceMetrics
from tools.gpu_fpga_bridge import FPGABridge, GPUAccelerator, ProcessingResult
from tools.enhanced_market_data import (
    EnhancedMarketDataFeed, 
    ITCHProtocolSimulator, 
    FIXProtocolSimulator,
    OUCHProtocolSimulator,
    ProtocolType
)

class TestPerformanceProfiler(unittest.TestCase):
    """Test suite for performance profiler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.profiler = PerformanceProfiler(output_dir="test_output")
        self.profiler.initialize_system(clock_freq=250e6, max_orders=1000)
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'profiler'):
            self.profiler.stop_profiling()
        # Clean up test output directory
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
    
    def test_initialization(self):
        """Test profiler initialization"""
        self.assertIsNotNone(self.profiler.fpga_core)
        self.assertIsNotNone(self.profiler.market_data_sim)
        self.assertIsNotNone(self.profiler.strategy)
        self.assertFalse(self.profiler.profiling_active)
        self.assertEqual(len(self.profiler.metrics_history), 0)
    
    def test_start_stop_profiling(self):
        """Test starting and stopping profiling"""
        # Start profiling
        self.profiler.start_profiling(duration_seconds=1)
        self.assertTrue(self.profiler.profiling_active)
        
        # Wait for profiling to complete
        time.sleep(1.5)
        
        # Check that profiling stopped
        self.assertFalse(self.profiler.profiling_active)
        
        # Check that metrics were collected
        self.assertGreater(len(self.profiler.metrics_history), 0)
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        # Mock the FPGA core to return predictable values
        self.profiler.fpga_core.get_throughput = Mock(return_value=1000)
        self.profiler.fpga_core.get_trade_count = Mock(return_value=100)
        self.profiler.fpga_core.get_pnl = Mock(return_value=50.0)
        self.profiler.fpga_core.get_risk_score = Mock(return_value=0.3)
        
        # Collect metrics
        metrics = self.profiler._collect_metrics()
        
        # Verify metrics
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.throughput, 1000)
        self.assertEqual(metrics.trade_count, 100)
        self.assertEqual(metrics.pnl, 50.0)
        self.assertEqual(metrics.risk_score, 0.3)
    
    def test_report_generation(self):
        """Test report generation"""
        # Generate some test data
        test_metrics = [
            PerformanceMetrics(
                timestamp=datetime.now(),
                latency_ns=100 + i * 10,
                throughput=1000 + i * 100,
                cpu_usage=20.0,
                memory_usage=100.0,
                trade_count=i * 10,
                pnl=i * 5.0,
                risk_score=0.1 + i * 0.01,
                error_count=0
            )
            for i in range(10)
        ]
        
        self.profiler.metrics_history = test_metrics
        
        # Generate report
        self.profiler._generate_report()
        
        # Check that output files were created
        self.assertTrue(os.path.exists("test_output"))
        
        # Check for CSV report
        csv_files = [f for f in os.listdir("test_output") if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 0)
        
        # Check for JSON report
        json_files = [f for f in os.listdir("test_output") if f.endswith('.json')]
        self.assertGreater(len(json_files), 0)

class TestGPUFPGABridge(unittest.TestCase):
    """Test suite for GPU-FPGA bridge"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bridge = FPGABridge(fpga_clock_freq=250e6)
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'bridge'):
            self.bridge.stop_processing()
    
    def test_initialization(self):
        """Test bridge initialization"""
        self.assertIsNotNone(self.bridge.fpga_core)
        self.assertIsNotNone(self.bridge.market_data_sim)
        self.assertIsNotNone(self.bridge.strategy)
        self.assertIsNotNone(self.bridge.gpu_accelerator)
        self.assertFalse(self.bridge.running)
    
    def test_gpu_accelerator(self):
        """Test GPU accelerator functionality"""
        gpu_accelerator = GPUAccelerator()
        
        # Test market data processing
        test_data = [
            {'price': 100.0 + i, 'volume': 1000 + i * 100}
            for i in range(10)
        ]
        
        result = gpu_accelerator.process_market_data_batch(test_data)
        
        # Check result structure
        self.assertIsInstance(result, ProcessingResult)
        self.assertIn('processor_type', result.__dict__)
        
        # If GPU is available, check processing
        if gpu_accelerator.gpu_available:
            self.assertTrue(result.success)
            self.assertIn('sma_20', result.processed_data)
            self.assertIn('volatility', result.processed_data)
            self.assertIn('vwap', result.processed_data)
    
    def test_processing_pipeline(self):
        """Test processing pipeline"""
        # Start processing
        self.bridge.start_processing()
        self.assertTrue(self.bridge.running)
        
        # Let it run for a short time
        time.sleep(2)
        
        # Stop processing
        self.bridge.stop_processing()
        self.assertFalse(self.bridge.running)
        
        # Check statistics
        stats = self.bridge.get_performance_summary()
        self.assertIn('processing_statistics', stats)
        self.assertIn('gpu_performance', stats)
        self.assertIn('fpga_performance', stats)
        self.assertIn('hybrid_performance', stats)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Mock some processing statistics
        self.bridge.processing_stats = {
            'gpu_processed': 1000,
            'fpga_processed': 10000,
            'gpu_time_ns': 1000000,
            'fpga_time_ns': 5000000,
            'total_processed': 10000
        }
        
        summary = self.bridge.get_performance_summary()
        
        # Check summary structure
        self.assertIn('processing_statistics', summary)
        self.assertIn('gpu_performance', summary)
        self.assertIn('fpga_performance', summary)
        self.assertIn('hybrid_performance', summary)
        
        # Check calculations
        self.assertEqual(summary['processing_statistics']['total_processed'], 10000)
        self.assertGreater(summary['hybrid_performance']['overall_throughput'], 0)

class TestEnhancedMarketData(unittest.TestCase):
    """Test suite for enhanced market data feed"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feed = EnhancedMarketDataFeed(protocol=ProtocolType.ITCH_5_0)
        self.received_messages = []
        
        # Set up message callback
        def message_callback(message):
            self.received_messages.append(message)
        
        self.feed.set_message_callback(message_callback)
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'feed'):
            self.feed.stop_feed()
    
    def test_itch_protocol_simulator(self):
        """Test ITCH protocol simulator"""
        simulator = ITCHProtocolSimulator()
        
        # Test message generation
        message = simulator.generate_message('AAPL')
        
        # Check message structure
        self.assertEqual(message.symbol, 'AAPL')
        self.assertEqual(message.protocol, ProtocolType.ITCH_5_0)
        self.assertIsNotNone(message.raw_data)
        self.assertIsNotNone(message.parsed_data)
        self.assertGreater(len(message.raw_data), 0)
    
    def test_fix_protocol_simulator(self):
        """Test FIX protocol simulator"""
        simulator = FIXProtocolSimulator()
        
        # Test message generation
        message = simulator.generate_message('GOOGL')
        
        # Check message structure
        self.assertEqual(message.symbol, 'GOOGL')
        self.assertEqual(message.protocol, ProtocolType.FIX_4_4)
        self.assertIn('bid_price', message.parsed_data)
        self.assertIn('ask_price', message.parsed_data)
        self.assertIn('spread', message.parsed_data)
    
    def test_ouch_protocol_simulator(self):
        """Test OUCH protocol simulator"""
        simulator = OUCHProtocolSimulator()
        
        # Test message generation
        message = simulator.generate_message('MSFT')
        
        # Check message structure
        self.assertEqual(message.symbol, 'MSFT')
        self.assertEqual(message.protocol, ProtocolType.OUCH_4_2)
        self.assertIn('order_token', message.parsed_data)
        self.assertIn('order_state', message.parsed_data)
    
    def test_market_data_feed(self):
        """Test market data feed functionality"""
        # Start feed
        self.feed.start_feed(['AAPL', 'GOOGL'])
        self.assertTrue(self.feed.running)
        
        # Let it run for a short time
        time.sleep(1)
        
        # Stop feed
        self.feed.stop_feed()
        self.assertFalse(self.feed.running)
        
        # Check that messages were received
        self.assertGreater(len(self.received_messages), 0)
        
        # Check message content
        first_message = self.received_messages[0]
        self.assertEqual(first_message.protocol, ProtocolType.ITCH_5_0)
        self.assertIn(first_message.symbol, ['AAPL', 'GOOGL'])
    
    def test_feed_statistics(self):
        """Test feed statistics collection"""
        # Start feed
        self.feed.start_feed(['AAPL'])
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Stop feed
        self.feed.stop_feed()
        
        # Check statistics
        stats = self.feed.get_statistics()
        self.assertIn('messages_sent', stats)
        self.assertIn('bytes_sent', stats)
        self.assertIn('message_rate', stats)
        self.assertGreater(stats['messages_sent'], 0)
        self.assertGreater(stats['bytes_sent'], 0)
    
    def test_burst_mode(self):
        """Test burst mode functionality"""
        # Enable burst mode
        self.feed.set_burst_mode(True)
        self.assertTrue(self.feed.burst_mode)
        
        # Start feed
        self.feed.start_feed(['AAPL'])
        
        # Run briefly
        time.sleep(0.2)
        
        # Stop feed
        self.feed.stop_feed()
        
        # Check that more messages were generated in burst mode
        stats = self.feed.get_statistics()
        self.assertGreater(stats['message_rate'], 100)  # Should be much higher in burst mode
    
    def test_latency_simulation(self):
        """Test latency simulation"""
        # Enable latency simulation
        self.feed.set_latency_simulation(True, (100, 200))
        self.assertTrue(self.feed.latency_simulation)
        self.assertEqual(self.feed.latency_range_ns, (100, 200))
        
        # Test with disabled latency simulation
        self.feed.set_latency_simulation(False)
        self.assertFalse(self.feed.latency_simulation)

class TestIntegration(unittest.TestCase):
    """Integration tests for all components"""
    
    def test_profiler_with_enhanced_market_data(self):
        """Test performance profiler with enhanced market data"""
        profiler = PerformanceProfiler(output_dir="test_integration")
        profiler.initialize_system()
        
        # Mock market data simulator with enhanced data
        enhanced_feed = EnhancedMarketDataFeed(protocol=ProtocolType.ITCH_5_0)
        
        try:
            # Start enhanced feed
            enhanced_feed.start_feed(['AAPL'])
            
            # Start profiling
            profiler.start_profiling(duration_seconds=2)
            
            # Wait for completion
            time.sleep(2.5)
            
            # Check results
            self.assertFalse(profiler.profiling_active)
            self.assertGreater(len(profiler.metrics_history), 0)
            
        finally:
            enhanced_feed.stop_feed()
            profiler.stop_profiling()
            if os.path.exists("test_integration"):
                shutil.rmtree("test_integration")
    
    def test_gpu_bridge_with_market_data(self):
        """Test GPU-FPGA bridge with market data protocols"""
        bridge = FPGABridge()
        
        try:
            # Start processing
            bridge.start_processing()
            
            # Run for a short time
            time.sleep(2)
            
            # Stop processing
            bridge.stop_processing()
            
            # Check performance
            summary = bridge.get_performance_summary()
            self.assertGreater(summary['processing_statistics']['total_processed'], 0)
            
        finally:
            bridge.stop_processing()

def run_performance_tests():
    """Run performance-specific tests"""
    print("🚀 Running Performance Tests")
    print("============================")
    
    # Test 1: Latency performance
    print("Test 1: Latency Performance")
    profiler = PerformanceProfiler()
    profiler.initialize_system(clock_freq=250e6)
    
    start_time = time.time()
    profiler.start_profiling(duration_seconds=5)
    
    # Wait for completion
    while profiler.profiling_active:
        time.sleep(0.1)
    
    summary = profiler.get_performance_summary()
    avg_latency = summary['performance_summary']['latency_stats']['mean_ns']
    
    print(f"   Average latency: {avg_latency:.1f} ns")
    
    # Check performance threshold
    if avg_latency > 1000:  # 1μs threshold
        print("   ❌ FAIL: Latency too high")
        return False
    else:
        print("   ✅ PASS: Latency within bounds")
    
    # Test 2: Throughput performance
    print("Test 2: Throughput Performance")
    bridge = FPGABridge()
    bridge.start_processing()
    
    time.sleep(5)
    bridge.stop_processing()
    
    summary = bridge.get_performance_summary()
    throughput = summary['hybrid_performance']['overall_throughput']
    
    print(f"   Throughput: {throughput:.0f} ops/sec")
    
    # Check throughput threshold
    if throughput < 10000:  # 10k ops/sec threshold
        print("   ❌ FAIL: Throughput too low")
        return False
    else:
        print("   ✅ PASS: Throughput sufficient")
    
    print("✅ All performance tests passed!")
    return True

def run_stress_tests():
    """Run stress tests for stability"""
    print("\n🔥 Running Stress Tests")
    print("======================")
    
    # Test 1: Extended profiling
    print("Test 1: Extended Profiling (30 seconds)")
    profiler = PerformanceProfiler()
    profiler.initialize_system()
    
    try:
        profiler.start_profiling(duration_seconds=30)
        
        # Monitor for errors
        error_count = 0
        for i in range(30):
            time.sleep(1)
            if len(profiler.error_log) > error_count:
                print(f"   Error detected at {i+1}s")
                error_count = len(profiler.error_log)
        
        if error_count == 0:
            print("   ✅ PASS: No errors during extended profiling")
        else:
            print(f"   ❌ FAIL: {error_count} errors detected")
            return False
    
    finally:
        profiler.stop_profiling()
    
    # Test 2: High-frequency market data
    print("Test 2: High-frequency Market Data")
    feed = EnhancedMarketDataFeed(protocol=ProtocolType.ITCH_5_0)
    feed.message_rate_hz = 10000  # 10k messages/sec
    
    messages_received = []
    feed.set_message_callback(lambda msg: messages_received.append(msg))
    
    try:
        feed.start_feed(['AAPL', 'GOOGL', 'MSFT'])
        time.sleep(10)
        feed.stop_feed()
        
        stats = feed.get_statistics()
        if stats['messages_sent'] > 50000:  # Expect at least 50k messages
            print(f"   ✅ PASS: {stats['messages_sent']} messages processed")
        else:
            print(f"   ❌ FAIL: Only {stats['messages_sent']} messages processed")
            return False
    
    finally:
        feed.stop_feed()
    
    print("✅ All stress tests passed!")
    return True

def main():
    """Main test runner"""
    print("🧪 FPGA Trading Advanced Tools Test Suite")
    print("=========================================")
    
    # Run unit tests
    print("\n1. Unit Tests")
    print("=============")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPerformanceProfiler))
    test_suite.addTest(unittest.makeSuite(TestGPUFPGABridge))
    test_suite.addTest(unittest.makeSuite(TestEnhancedMarketData))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if not result.wasSuccessful():
        print(f"\n❌ Unit tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        return False
    
    print("\n✅ All unit tests passed!")
    
    # Run performance tests
    print("\n2. Performance Tests")
    print("===================")
    
    if not run_performance_tests():
        return False
    
    # Run stress tests
    print("\n3. Stress Tests")
    print("===============")
    
    if not run_stress_tests():
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("====================")
    print("✅ Unit tests: PASSED")
    print("✅ Performance tests: PASSED")
    print("✅ Stress tests: PASSED")
    print("\nThe FPGA trading advanced tools are working correctly!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
