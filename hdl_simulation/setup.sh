#!/bin/bash

# HDL Simulation Environment Setup Script
# Automates installation and configuration of FPGA simulation tools

set -e

echo "==============================================="
echo "FPGA Trading Accelerator - HDL Simulation Setup"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if command -v apt-get &> /dev/null; then
        PKG_MANAGER="apt"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
    elif command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
    else
        print_error "Unsupported Linux package manager"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    PKG_MANAGER="brew"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS with package manager: $PKG_MANAGER"

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root is not recommended"
fi

# Install dependencies based on OS and package manager
install_dependencies() {
    print_status "Installing simulation dependencies..."
    
    case $PKG_MANAGER in
        "apt")
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                git \
                python3 \
                python3-pip \
                iverilog \
                gtkwave \
                verilator \
                make \
                cmake
            ;;
        "yum"|"dnf")
            sudo $PKG_MANAGER install -y \
                gcc \
                gcc-c++ \
                git \
                python3 \
                python3-pip \
                iverilog \
                gtkwave \
                verilator \
                make \
                cmake
            ;;
        "brew")
            if ! command -v brew &> /dev/null; then
                print_error "Homebrew not found. Please install from https://brew.sh"
                exit 1
            fi
            
            brew update
            brew install \
                icarus-verilog \
                gtkwave \
                verilator \
                python@3.9
            ;;
        *)
            print_error "Unsupported package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac
    
    print_status "System dependencies installed successfully"
}

# Install Python requirements
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if command -v pip3 &> /dev/null; then
        pip3 install --user -r requirements.txt
    elif command -v pip &> /dev/null; then
        pip install --user -r requirements.txt
    else
        print_error "pip not found. Please install Python pip"
        exit 1
    fi
    
    print_status "Python dependencies installed successfully"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check Icarus Verilog
    if command -v iverilog &> /dev/null; then
        VERSION=$(iverilog -V 2>&1 | head -n1)
        print_status "Icarus Verilog: $VERSION"
    else
        print_error "Icarus Verilog not found"
        return 1
    fi
    
    # Check GTKWave
    if command -v gtkwave &> /dev/null; then
        print_status "GTKWave: Available"
    else
        print_warning "GTKWave not found (waveform viewing will not work)"
    fi
    
    # Check Verilator
    if command -v verilator &> /dev/null; then
        VERSION=$(verilator --version 2>&1 | head -n1)
        print_status "Verilator: $VERSION"
    else
        print_error "Verilator not found"
        return 1
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        VERSION=$(python3 --version)
        print_status "Python: $VERSION"
    else
        print_error "Python 3 not found"
        return 1
    fi
    
    print_status "Installation verification completed"
}

# Create simulation directories
setup_directories() {
    print_status "Setting up simulation directories..."
    
    mkdir -p sim
    mkdir -p logs
    mkdir -p reports
    
    print_status "Directories created successfully"
}

# Run basic test
run_basic_test() {
    print_status "Running basic functionality test..."
    
    # Test Icarus Verilog
    echo "module test; initial begin \$display(\"Icarus Verilog test\"); \$finish; end endmodule" > test.v
    if iverilog -o test test.v && vvp test; then
        print_status "Icarus Verilog test: PASSED"
    else
        print_error "Icarus Verilog test: FAILED"
        return 1
    fi
    rm -f test test.v
    
    # Test Python imports
    if python3 -c "import numpy, matplotlib, pandas; print('Python modules test: PASSED')"; then
        print_status "Python modules test: PASSED"
    else
        print_error "Python modules test: FAILED"
        return 1
    fi
    
    print_status "Basic tests completed successfully"
}

# Main setup function
main() {
    print_status "Starting HDL simulation environment setup..."
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Install dependencies
    install_dependencies
    
    # Install Python requirements
    if [ -f "requirements.txt" ]; then
        install_python_deps
    else
        print_warning "requirements.txt not found, skipping Python dependencies"
    fi
    
    # Setup directories
    setup_directories
    
    # Verify installation
    verify_installation
    
    # Run basic test
    run_basic_test
    
    print_status "Setup completed successfully!"
    echo ""
    echo "==============================================="
    echo "Setup Summary:"
    echo "✅ System dependencies installed"
    echo "✅ Python dependencies installed"
    echo "✅ Simulation directories created"
    echo "✅ Installation verified"
    echo "✅ Basic tests passed"
    echo ""
    echo "Next steps:"
    echo "1. Run 'make help' to see available commands"
    echo "2. Run 'make all' to execute all simulations"
    echo "3. Run './run_simulation.py' for automated testing"
    echo "4. Check README.md for detailed usage instructions"
    echo "==============================================="
}

# Handle command line arguments
case "${1:-}" in
    "--help"|"-h")
        echo "HDL Simulation Environment Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --deps-only    Install dependencies only"
        echo "  --verify-only  Verify installation only"
        echo "  --test-only    Run basic tests only"
        echo ""
        exit 0
        ;;
    "--deps-only")
        install_dependencies
        install_python_deps
        ;;
    "--verify-only")
        verify_installation
        ;;
    "--test-only")
        run_basic_test
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
