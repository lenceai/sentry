#!/bin/bash
# CUDA Toolkit Installation Script for Sentry Project
# Version: v1.0.0
#
# This script installs CUDA Toolkit 13.0 which is required for:
# - DeepSpeed (distributed training)
# - Mamba-SSM (state space models)
#
# Usage:
#   bash install_cuda_toolkit.sh

echo "============================================"
echo "CUDA Toolkit 13.0 Installation Script"
echo "============================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please do not run this script as root (no sudo)"
   exit 1
fi

# Detect Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
    echo "Detected OS: $OS $VER"
else
    echo "Cannot detect OS version"
    exit 1
fi

# Set repository based on Ubuntu version
if [[ "$VER" == "24.04" ]]; then
    # Ubuntu 24.04 - use 22.04 repos (compatible)
    REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
    echo "Note: Using Ubuntu 22.04 CUDA repository (compatible with 24.04)"
elif [[ "$VER" == "22.04" ]]; then
    REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
elif [[ "$VER" == "20.04" ]]; then
    REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64"
else
    echo "Unsupported Ubuntu version: $VER"
    echo "This script supports Ubuntu 20.04, 22.04, and 24.04"
    exit 1
fi

echo ""
echo "Step 1: Downloading CUDA keyring..."
wget -q $REPO_URL/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb

if [ $? -ne 0 ]; then
    echo "Failed to download CUDA keyring"
    exit 1
fi

echo "Step 2: Installing CUDA keyring..."
sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb

echo "Step 3: Updating package list..."
sudo apt-get update

echo "Step 4: Installing CUDA Toolkit 13.0..."
echo "This may take several minutes..."
sudo apt-get install -y cuda-toolkit-13-0

if [ $? -ne 0 ]; then
    echo "Failed to install CUDA Toolkit"
    exit 1
fi

# Clean up
rm /tmp/cuda-keyring_1.1-1_all.deb

echo ""
echo "============================================"
echo "CUDA Toolkit 13.0 installed successfully!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Add these lines to your ~/.bashrc:"
echo ""
echo "   export CUDA_HOME=/usr/local/cuda-13.0"
echo "   export PATH=\$CUDA_HOME/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "2. Reload your shell:"
echo "   source ~/.bashrc"
echo ""
echo "3. Verify installation:"
echo "   nvcc --version"
echo ""
echo "4. Re-run dependency installation:"
echo "   python setup.py --install-deps"
echo ""

