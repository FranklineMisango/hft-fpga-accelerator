#!/usr/bin/env python3
"""
GPU-FPGA Bridge for High-Performance Trading
Integration layer between GPU acceleration and FPGA simulation components.
"""

import sys
import os
import time
import numpy as np
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import asyncio

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'CuDA_cuDNN'))

from python_sim.fpga_core import FPGACore
from python_sim.market_data_simulator import MarketDataSimulator
from python_sim.strategies import ArbitrageStrategy, MarketMakingStrategy

# Try to import GPU acceleration components
try:
    import cupy as cp
    from numba import cuda, njit
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU acceleration not available, falling back to CPU")

@dataclass
class ProcessingResult:
    """Result from GPU or FPGA processing"""
    processed_data: Dict[str, Any]
    processing_time_ns: float
    processor_type: str
    success: bool
    error: Optional[str] = None

class GPUAccelerator:
    """GPU-based market data processing acceleration"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        if self.gpu_available:
            self.device = cp.cuda.Device()
            self.stream = cp.cuda.Stream()
            
    def process_market_data_batch(self, market_data_batch: List[Dict]) -> ProcessingResult:
        """Process a batch of market data using GPU acceleration"""
        if not self.gpu_available:
            return ProcessingResult(
                processed_data={},
                processing_time_ns=0,
                processor_type="CPU_FALLBACK",
                success=False,
                error="GPU not available"
            )
        
        try:
            start_time = time.perf_counter()
            
            # Convert to GPU arrays
            prices = cp.array([data['price'] for data in market_data_batch])
            volumes = cp.array([data['volume'] for data in market_data_batch])
            
            # GPU-accelerated calculations
            with self.stream:
                # Technical indicators
                sma_20 = self._calculate_sma_gpu(prices, 20)
                volatility = self._calculate_volatility_gpu(prices)
                vwap = self._calculate_vwap_gpu(prices, volumes)
                
                # Pattern recognition
                momentum = self._calculate_momentum_gpu(prices)
                
                # Synchronize stream
                self.stream.synchronize()
            
            end_time = time.perf_counter()
            processing_time_ns = (end_time - start_time) * 1e9
            
            # Convert results back to CPU
            processed_data = {
                'sma_20': float(sma_20[-1]) if len(sma_20) > 0 else 0.0,
                'volatility': float(volatility),
                'vwap': float(vwap),
                'momentum': float(momentum),
                'batch_size': len(market_data_batch)
            }
            
            return ProcessingResult(
                processed_data=processed_data,
                processing_time_ns=processing_time_ns,
                processor_type="GPU_CUDA",
                success=True
            )
            
        except Exception as e:
            return ProcessingResult(
                processed_data={},
                processing_time_ns=0,
                processor_type="GPU_CUDA",
                success=False,
                error=str(e)
            )
    
    def _calculate_sma_gpu(self, prices: cp.ndarray, window: int) -> cp.ndarray:
        """Calculate Simple Moving Average on GPU"""
        if len(prices) < window:
            return cp.array([])
        
        # Use CuPy's convolve for efficient SMA calculation
        kernel = cp.ones(window) / window
        return cp.convolve(prices, kernel, mode='valid')
    
    def _calculate_volatility_gpu(self, prices: cp.ndarray) -> cp.ndarray:
        """Calculate price volatility on GPU"""
        if len(prices) < 2:
            return cp.array([0.0])
        
        returns = cp.diff(prices) / prices[:-1]
        return cp.std(returns)
    
    def _calculate_vwap_gpu(self, prices: cp.ndarray, volumes: cp.ndarray) -> cp.ndarray:
        """Calculate Volume-Weighted Average Price on GPU"""
        if len(prices) == 0:
            return cp.array([0.0])
        
        total_volume = cp.sum(volumes)
        if total_volume == 0:
            return cp.mean(prices)
        
        return cp.sum(prices * volumes) / total_volume
    
    def _calculate_momentum_gpu(self, prices: cp.ndarray) -> cp.ndarray:
        """Calculate price momentum on GPU"""
        if len(prices) < 2:
            return cp.array([0.0])
        
        return (prices[-1] - prices[0]) / prices[0]

class FPGABridge:
    """Bridge between GPU acceleration and FPGA simulation"""
    
    def __init__(self, fpga_clock_freq: float = 250e6):
        # Initialize FPGA components
        self.fpga_core = FPGACore(clock_freq=fpga_clock_freq, max_orders=1000)
        self.market_data_sim = MarketDataSimulator(tick_rate=1000)
        self.strategy = ArbitrageStrategy()
        
        # Initialize GPU accelerator
        self.gpu_accelerator = GPUAccelerator()
        
        # Configuration
        self.use_gpu_preprocessing = True
        self.batch_size = 100
        self.processing_stats = {
            'gpu_processed': 0,
            'fpga_processed': 0,
            'gpu_time_ns': 0,
            'fpga_time_ns': 0,
            'total_processed': 0
        }
        
        # Data buffers
        self.market_data_buffer = []
        self.processing_results = []
        
        # Threading
        self.processing_thread = None
        self.running = False
        
    def start_processing(self):
        """Start the hybrid GPU-FPGA processing pipeline"""
        if self.running:
            print("⚠️  Processing already running")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("🚀 GPU-FPGA processing pipeline started")
    
    def stop_processing(self):
        """Stop the processing pipeline"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        print("⏹️  Processing pipeline stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Generate market data
                market_data = self.market_data_sim.generate_tick()
                self.market_data_buffer.append(market_data)
                
                # Process batch when buffer is full
                if len(self.market_data_buffer) >= self.batch_size:
                    self._process_batch()
                
                # Small delay to prevent overwhelming
                time.sleep(0.001)
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                break
    
    def _process_batch(self):
        """Process a batch of market data using GPU+FPGA pipeline"""
        if not self.market_data_buffer:
            return
        
        batch = self.market_data_buffer.copy()
        self.market_data_buffer.clear()
        
        # Step 1: GPU preprocessing (if available)
        gpu_result = None
        if self.use_gpu_preprocessing and self.gpu_accelerator.gpu_available:
            gpu_result = self.gpu_accelerator.process_market_data_batch(batch)
            if gpu_result.success:
                self.processing_stats['gpu_processed'] += len(batch)
                self.processing_stats['gpu_time_ns'] += gpu_result.processing_time_ns
        
        # Step 2: FPGA processing for each tick
        fpga_results = []
        for market_data in batch:
            fpga_result = self._process_with_fpga(market_data, gpu_result)
            fpga_results.append(fpga_result)
        
        # Step 3: Combine results
        combined_result = self._combine_results(batch, gpu_result, fpga_results)
        self.processing_results.append(combined_result)
        
        # Update statistics
        self.processing_stats['total_processed'] += len(batch)
    
    def _process_with_fpga(self, market_data: Dict, gpu_result: Optional[ProcessingResult]) -> ProcessingResult:
        """Process individual market data tick with FPGA simulation"""
        try:
            start_time = time.perf_counter()
            
            # Enhance market data with GPU results if available
            enhanced_data = market_data.copy()
            if gpu_result and gpu_result.success:
                enhanced_data.update(gpu_result.processed_data)
            
            # FPGA processing
            self.fpga_core.process_market_data(enhanced_data)
            signal = self.strategy.process_market_data(enhanced_data)
            
            end_time = time.perf_counter()
            processing_time_ns = (end_time - start_time) * 1e9
            
            # Update FPGA statistics
            self.processing_stats['fpga_processed'] += 1
            self.processing_stats['fpga_time_ns'] += processing_time_ns
            
            processed_data = {
                'market_data': enhanced_data,
                'signal': signal,
                'fpga_metrics': {
                    'latency_ns': processing_time_ns,
                    'throughput': self.fpga_core.get_throughput(),
                    'trade_count': self.fpga_core.get_trade_count(),
                    'pnl': self.fpga_core.get_pnl()
                }
            }
            
            return ProcessingResult(
                processed_data=processed_data,
                processing_time_ns=processing_time_ns,
                processor_type="FPGA",
                success=True
            )
            
        except Exception as e:
            return ProcessingResult(
                processed_data={},
                processing_time_ns=0,
                processor_type="FPGA",
                success=False,
                error=str(e)
            )
    
    def _combine_results(self, batch: List[Dict], gpu_result: Optional[ProcessingResult], 
                        fpga_results: List[ProcessingResult]) -> Dict[str, Any]:
        """Combine GPU and FPGA processing results"""
        successful_fpga = [r for r in fpga_results if r.success]
        
        return {
            'timestamp': datetime.now(),
            'batch_size': len(batch),
            'gpu_processing': {
                'enabled': self.use_gpu_preprocessing,
                'success': gpu_result.success if gpu_result else False,
                'time_ns': gpu_result.processing_time_ns if gpu_result else 0,
                'data': gpu_result.processed_data if gpu_result and gpu_result.success else {}
            },
            'fpga_processing': {
                'successful_ticks': len(successful_fpga),
                'total_time_ns': sum(r.processing_time_ns for r in successful_fpga),
                'avg_time_ns': np.mean([r.processing_time_ns for r in successful_fpga]) if successful_fpga else 0,
                'results': [r.processed_data for r in successful_fpga]
            },
            'performance_metrics': {
                'total_processing_time_ns': (gpu_result.processing_time_ns if gpu_result else 0) + 
                                          sum(r.processing_time_ns for r in fpga_results),
                'gpu_fpga_ratio': (gpu_result.processing_time_ns if gpu_result else 0) / 
                                 max(sum(r.processing_time_ns for r in fpga_results), 1),
                'throughput_ops_per_sec': len(batch) / (sum(r.processing_time_ns for r in fpga_results) / 1e9) if fpga_results else 0
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_time_ns = self.processing_stats['gpu_time_ns'] + self.processing_stats['fpga_time_ns']
        
        return {
            'processing_statistics': self.processing_stats,
            'gpu_performance': {
                'enabled': self.gpu_accelerator.gpu_available,
                'processed_batches': self.processing_stats['gpu_processed'] // self.batch_size,
                'avg_batch_time_ns': self.processing_stats['gpu_time_ns'] / max(self.processing_stats['gpu_processed'], 1),
                'speedup_factor': self._calculate_gpu_speedup()
            },
            'fpga_performance': {
                'processed_ticks': self.processing_stats['fpga_processed'],
                'avg_tick_time_ns': self.processing_stats['fpga_time_ns'] / max(self.processing_stats['fpga_processed'], 1),
                'clock_frequency_mhz': self.fpga_core.clock_freq / 1e6
            },
            'hybrid_performance': {
                'total_processing_time_ns': total_time_ns,
                'gpu_percentage': (self.processing_stats['gpu_time_ns'] / max(total_time_ns, 1)) * 100,
                'fpga_percentage': (self.processing_stats['fpga_time_ns'] / max(total_time_ns, 1)) * 100,
                'overall_throughput': self.processing_stats['total_processed'] / (total_time_ns / 1e9) if total_time_ns > 0 else 0
            }
        }
    
    def _calculate_gpu_speedup(self) -> float:
        """Calculate GPU speedup factor compared to CPU processing"""
        if not self.gpu_accelerator.gpu_available or self.processing_stats['gpu_processed'] == 0:
            return 1.0
        
        # Estimate CPU processing time (simplified)
        estimated_cpu_time_ns = self.processing_stats['gpu_processed'] * 1000  # Assume 1μs per item on CPU
        actual_gpu_time_ns = self.processing_stats['gpu_time_ns']
        
        return estimated_cpu_time_ns / max(actual_gpu_time_ns, 1)
    
    def save_performance_report(self, filename: str = None):
        """Save performance report to file"""
        if filename is None:
            filename = f"gpu_fpga_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'gpu_available': self.gpu_accelerator.gpu_available,
                'fpga_clock_freq_mhz': self.fpga_core.clock_freq / 1e6,
                'batch_size': self.batch_size
            },
            'performance_summary': self.get_performance_summary(),
            'recent_results': self.processing_results[-10:] if len(self.processing_results) > 10 else self.processing_results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Performance report saved: {filename}")

def main():
    """Main function demonstrating GPU-FPGA bridge"""
    print("🚀 GPU-FPGA Trading Bridge Demo")
    print("================================")
    
    # Initialize bridge
    bridge = FPGABridge(fpga_clock_freq=250e6)
    
    try:
        # Start processing
        bridge.start_processing()
        
        # Run for demonstration
        print("🔄 Processing for 10 seconds...")
        time.sleep(10)
        
        # Stop and get results
        bridge.stop_processing()
        
        # Print performance summary
        summary = bridge.get_performance_summary()
        print("\n📊 PERFORMANCE SUMMARY:")
        print(f"Total processed: {summary['processing_statistics']['total_processed']:,}")
        print(f"GPU processed: {summary['processing_statistics']['gpu_processed']:,}")
        print(f"FPGA processed: {summary['processing_statistics']['fpga_processed']:,}")
        print(f"Overall throughput: {summary['hybrid_performance']['overall_throughput']:.0f} ops/sec")
        
        if summary['gpu_performance']['enabled']:
            print(f"GPU speedup: {summary['gpu_performance']['speedup_factor']:.1f}x")
        
        # Save report
        bridge.save_performance_report()
        
        print("✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        bridge.stop_processing()
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        bridge.stop_processing()

if __name__ == "__main__":
    main()
