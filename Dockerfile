# HDL Simulation Environment for FPGA Trading Accelerator
# Docker container for reproducible simulation environment

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    iverilog \
    gtkwave \
    verilator \
    make \
    cmake \
    gcc \
    g++ \
    clang \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for analysis
RUN pip3 install \
    numpy \
    matplotlib \
    pandas \
    jupyter \
    cocotb \
    pytest

# Create working directory
WORKDIR /workspace

# Copy simulation files
COPY . .

# Set up environment
ENV PATH="/workspace:${PATH}"

# Default command
CMD ["make", "help"]
