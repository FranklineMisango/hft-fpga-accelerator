#!/usr/bin/env python3
"""
Setup script for FPGA Trading Simulation
"""

import os
import sys
import subprocess

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully!")

def setup_directories():
    """Create necessary directories"""
    directories = [
        "output",
        "logs",
        "data",
        "reports"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    """Main setup function"""
    print("Setting up FPGA Trading Simulation environment...")
    
    try:
        install_requirements()
        setup_directories()
        
        print("\n=== Setup Complete ===")
        print("You can now run the simulation with:")
        print("  python simulation_runner.py")
        print("\nOr run individual components:")
        print("  python fpga_core.py")
        print("  python market_data_simulator.py")
        print("  python strategies.py")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
