#!/usr/bin/env python3
"""
FPGA Trading System Simulation Runner
Automated testing and analysis framework for HDL simulation

Features:
- Automated test execution
- Performance analysis
- Report generation
- Waveform analysis
- Regression testing
"""

import os
import sys
import subprocess
import time
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class SimulationRunner:
    """Main simulation runner class"""
    
    def __init__(self, work_dir: str = "sim"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.results = {}
        self.start_time = None
        
    def run_command(self, command: str, timeout: int = 300) -> Tuple[int, str, str]:
        """Execute a command with timeout"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=str(self.work_dir.parent)
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def run_iverilog_test(self, test_name: str, rtl_files: List[str], tb_file: str) -> Dict:
        """Run Icarus Verilog test"""
        print(f"Running Icarus Verilog test: {test_name}")
        
        # Compile
        rtl_list = " ".join(rtl_files)
        compile_cmd = f"iverilog -g2012 -Wall -o {self.work_dir}/{test_name} {rtl_list} {tb_file}"
        
        start_time = time.time()
        ret_code, stdout, stderr = self.run_command(compile_cmd)
        compile_time = time.time() - start_time
        
        if ret_code != 0:
            return {
                "status": "FAILED",
                "phase": "compile",
                "compile_time": compile_time,
                "error": stderr
            }
        
        # Run simulation
        run_cmd = f"cd {self.work_dir} && vvp {test_name}"
        start_time = time.time()
        ret_code, stdout, stderr = self.run_command(run_cmd)
        run_time = time.time() - start_time
        
        # Parse results
        passed = "PASSED" in stdout or "✓" in stdout
        failed = "FAILED" in stdout or "✗" in stdout
        
        return {
            "status": "PASSED" if passed and not failed else "FAILED",
            "phase": "simulation",
            "compile_time": compile_time,
            "run_time": run_time,
            "output": stdout,
            "error": stderr if stderr else None
        }
    
    def run_verilator_test(self, test_name: str, rtl_files: List[str], 
                          tb_file: str, cpp_file: Optional[str] = None) -> Dict:
        """Run Verilator test"""
        print(f"Running Verilator test: {test_name}")
        
        # Build command
        rtl_list = " ".join(rtl_files)
        verilator_cmd = f"verilator --cc --exe --build --trace -Wall -Wno-fatal"
        
        if cpp_file:
            verilator_cmd += f" {rtl_list} {tb_file} {cpp_file}"
        else:
            verilator_cmd += f" {rtl_list} {tb_file}"
        
        start_time = time.time()
        ret_code, stdout, stderr = self.run_command(verilator_cmd)
        build_time = time.time() - start_time
        
        if ret_code != 0:
            return {
                "status": "FAILED",
                "phase": "build",
                "build_time": build_time,
                "error": stderr
            }
        
        # Run simulation
        if cpp_file:
            run_cmd = f"./obj_dir/V{test_name}"
        else:
            run_cmd = f"./obj_dir/V{test_name.replace('_tb', '')}"
        
        start_time = time.time()
        ret_code, stdout, stderr = self.run_command(run_cmd)
        run_time = time.time() - start_time
        
        # Parse results
        passed = "PASSED" in stdout or "✓" in stdout
        failed = "FAILED" in stdout or "✗" in stdout
        
        return {
            "status": "PASSED" if passed and not failed else "FAILED",
            "phase": "simulation",
            "build_time": build_time,
            "run_time": run_time,
            "output": stdout,
            "error": stderr if stderr else None
        }
    
    def run_all_tests(self, simulator: str = "both") -> Dict:
        """Run all tests"""
        print("=" * 50)
        print("FPGA Trading System Simulation Suite")
        print("=" * 50)
        
        self.start_time = time.time()
        
        # Test configurations
        tests = [
            {
                "name": "market_data_tb",
                "rtl_files": ["rtl/market_data_processor.v"],
                "tb_file": "testbench/market_data_tb.v",
                "cpp_file": None
            },
            {
                "name": "order_manager_tb",
                "rtl_files": ["rtl/order_manager.v"],
                "tb_file": "testbench/order_manager_tb.v",
                "cpp_file": None
            },
            {
                "name": "trading_strategy_tb",
                "rtl_files": ["rtl/trading_strategy.v"],
                "tb_file": "testbench/trading_strategy_tb.v",
                "cpp_file": None
            },
            {
                "name": "fpga_trading_system_tb",
                "rtl_files": [
                    "rtl/market_data_processor.v",
                    "rtl/order_manager.v",
                    "rtl/trading_strategy.v"
                ],
                "tb_file": "testbench/fpga_trading_system_tb.v",
                "cpp_file": "cpp_testbench/fpga_trading_system_test.cpp"
            }
        ]
        
        # Run tests
        if simulator in ["iverilog", "both"]:
            print("\nRunning Icarus Verilog tests...")
            self.results["iverilog"] = {}
            for test in tests:
                if test["name"] != "fpga_trading_system_tb":  # Skip integration for basic tests
                    result = self.run_iverilog_test(
                        test["name"], 
                        test["rtl_files"], 
                        test["tb_file"]
                    )
                    self.results["iverilog"][test["name"]] = result
                    print(f"  {test['name']}: {result['status']}")
        
        if simulator in ["verilator", "both"]:
            print("\nRunning Verilator tests...")
            self.results["verilator"] = {}
            for test in tests:
                result = self.run_verilator_test(
                    test["name"], 
                    test["rtl_files"], 
                    test["tb_file"],
                    test["cpp_file"]
                )
                self.results["verilator"][test["name"]] = result
                print(f"  {test['name']}: {result['status']}")
        
        total_time = time.time() - self.start_time
        self.results["total_time"] = total_time
        
        return self.results
    
    def generate_report(self, output_file: str = "simulation_report.json"):
        """Generate detailed test report"""
        print("\nGenerating test report...")
        
        # Summary statistics
        summary = {
            "total_time": self.results.get("total_time", 0),
            "timestamp": time.time(),
            "simulators": {}
        }
        
        for simulator in ["iverilog", "verilator"]:
            if simulator in self.results:
                sim_results = self.results[simulator]
                passed = sum(1 for r in sim_results.values() if r["status"] == "PASSED")
                failed = sum(1 for r in sim_results.values() if r["status"] == "FAILED")
                total = passed + failed
                
                summary["simulators"][simulator] = {
                    "total_tests": total,
                    "passed": passed,
                    "failed": failed,
                    "pass_rate": (passed / total * 100) if total > 0 else 0
                }
        
        # Write report
        report = {
            "summary": summary,
            "detailed_results": self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {output_file}")
        return report
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("SIMULATION SUMMARY")
        print("=" * 50)
        
        total_time = self.results.get("total_time", 0)
        print(f"Total execution time: {total_time:.2f} seconds")
        
        for simulator in ["iverilog", "verilator"]:
            if simulator in self.results:
                print(f"\n{simulator.upper()} Results:")
                sim_results = self.results[simulator]
                
                for test_name, result in sim_results.items():
                    status = result["status"]
                    emoji = "✅" if status == "PASSED" else "❌"
                    print(f"  {emoji} {test_name}: {status}")
                    
                    if result.get("compile_time"):
                        print(f"    Compile time: {result['compile_time']:.3f}s")
                    if result.get("build_time"):
                        print(f"    Build time: {result['build_time']:.3f}s")
                    if result.get("run_time"):
                        print(f"    Run time: {result['run_time']:.3f}s")
                    
                    if result["status"] == "FAILED" and result.get("error"):
                        print(f"    Error: {result['error'][:100]}...")
        
        # Overall status
        all_passed = all(
            result["status"] == "PASSED" 
            for sim_results in self.results.values() 
            if isinstance(sim_results, dict)
            for result in sim_results.values()
        )
        
        print(f"\nOverall Status: {'🎉 ALL TESTS PASSED' if all_passed else '⚠️ SOME TESTS FAILED'}")
    
    def analyze_performance(self):
        """Analyze performance metrics from simulation output"""
        print("\n" + "=" * 50)
        print("PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Extract performance metrics from output
        for simulator in ["iverilog", "verilator"]:
            if simulator in self.results:
                print(f"\n{simulator.upper()} Performance:")
                
                for test_name, result in self.results[simulator].items():
                    if result["status"] == "PASSED" and result.get("output"):
                        output = result["output"]
                        
                        # Extract latency information
                        if "latency" in output.lower():
                            lines = output.split('\n')
                            for line in lines:
                                if "latency" in line.lower():
                                    print(f"  {test_name}: {line.strip()}")
                        
                        # Extract throughput information
                        if "throughput" in output.lower():
                            lines = output.split('\n')
                            for line in lines:
                                if "throughput" in line.lower():
                                    print(f"  {test_name}: {line.strip()}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="FPGA Trading System Simulation Runner")
    parser.add_argument("--simulator", choices=["iverilog", "verilator", "both"], 
                       default="both", help="Simulator to use")
    parser.add_argument("--work-dir", default="sim", help="Working directory")
    parser.add_argument("--report", default="simulation_report.json", 
                       help="Report output file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create runner
    runner = SimulationRunner(args.work_dir)
    
    try:
        # Run tests
        results = runner.run_all_tests(args.simulator)
        
        # Generate report
        runner.generate_report(args.report)
        
        # Print summary
        runner.print_summary()
        
        # Analyze performance
        runner.analyze_performance()
        
        # Exit with appropriate code
        all_passed = all(
            result["status"] == "PASSED" 
            for sim_results in results.values() 
            if isinstance(sim_results, dict)
            for result in sim_results.values()
        )
        
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
