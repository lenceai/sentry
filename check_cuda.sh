#!/bin/bash
# Quick CUDA Status Check for Sentry Project
# Version: v1.0.0

echo "=========================================="
echo "CUDA Status Check"
echo "=========================================="
echo ""

echo "1. NVIDIA Driver & Runtime:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    CUDA_RUNTIME=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "   CUDA Runtime Version: $CUDA_RUNTIME"
    echo "   ✅ NVIDIA Driver installed"
else
    echo "   ❌ NVIDIA Driver NOT found"
fi

echo ""
echo "2. CUDA Toolkit (Development):"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "   ✅ CUDA Toolkit installed"
    echo "   Location: $(which nvcc)"
else
    echo "   ❌ CUDA Toolkit NOT installed (nvcc not found)"
    echo "   This is required for DeepSpeed and Mamba-SSM"
fi

echo ""
echo "3. Environment Variables:"
if [ -z "$CUDA_HOME" ]; then
    echo "   ❌ CUDA_HOME not set"
else
    echo "   ✅ CUDA_HOME=$CUDA_HOME"
fi

if [ -z "$LD_LIBRARY_PATH" ]; then
    echo "   ⚠️  LD_LIBRARY_PATH not set"
else
    echo "   ✅ LD_LIBRARY_PATH includes: $(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda | head -1)"
fi

echo ""
echo "4. PyTorch CUDA Status:"
if conda env list | grep -q "sentry"; then
    echo "   Testing PyTorch in 'sentry' environment..."
    conda run -n sentry python -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA device count: {torch.cuda.device_count()}'); print(f'   Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "   ✅ PyTorch working correctly"
    else
        echo "   ⚠️  PyTorch check failed (environment may need setup)"
    fi
else
    echo "   ⚠️  'sentry' conda environment not found"
    echo "   Run: python setup.py --create-env"
fi

echo ""
echo "5. Optional Packages:"
if conda env list | grep -q "sentry"; then
    DEEPSPEED=$(conda run -n sentry pip list 2>/dev/null | grep deepspeed || echo "not installed")
    MAMBA=$(conda run -n sentry pip list 2>/dev/null | grep mamba-ssm || echo "not installed")
    
    if [[ "$DEEPSPEED" != "not installed" ]]; then
        echo "   ✅ DeepSpeed: $DEEPSPEED"
    else
        echo "   ❌ DeepSpeed: not installed"
    fi
    
    if [[ "$MAMBA" != "not installed" ]]; then
        echo "   ✅ Mamba-SSM: $MAMBA"
    else
        echo "   ❌ Mamba-SSM: not installed"
    fi
fi

echo ""
echo "=========================================="
echo "Recommendations:"
echo "=========================================="

if ! command -v nvcc &> /dev/null; then
    echo "• Install CUDA Toolkit for DeepSpeed and Mamba-SSM:"
    echo "  ./install_cuda_toolkit.sh"
    echo ""
fi

if [ -z "$CUDA_HOME" ] && command -v nvcc &> /dev/null; then
    echo "• Add to ~/.bashrc:"
    echo "  export CUDA_HOME=/usr/local/cuda-13.0"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    echo ""
fi

echo "• See CUDA_SETUP.md for detailed instructions"
echo ""

