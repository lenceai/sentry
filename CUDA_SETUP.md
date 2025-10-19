# CUDA Setup Guide for Sentry Project

## Current Status

✅ **NVIDIA Driver**: Installed (version 580.65.06)  
✅ **CUDA Runtime**: Available (version 13.0)  
✅ **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)  
❌ **CUDA Toolkit**: NOT installed (required for DeepSpeed and Mamba-SSM)

## What's the Difference?

- **CUDA Runtime**: Allows running pre-compiled CUDA programs (PyTorch works fine)
- **CUDA Toolkit**: Includes development tools (nvcc compiler) needed to compile CUDA code from source

## What Works Without CUDA Toolkit

✅ PyTorch with GPU acceleration  
✅ Transformer models  
✅ Standard GPU training  
✅ All basic deep learning operations

## What Requires CUDA Toolkit

❌ DeepSpeed (distributed training optimization)  
❌ Mamba-SSM (state space models)  
❌ Other packages that compile CUDA code from source

## Installation Options

### Option 1: Install CUDA Toolkit (Recommended)

Run the provided installation script:

```bash
cd ~/sentry
./install_cuda_toolkit.sh
```

This will:
1. Download and install CUDA Toolkit 13.0
2. Set up the necessary repositories
3. Install the development tools

After installation:

1. Add to your `~/.bashrc`:
```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

2. Reload your shell:
```bash
source ~/.bashrc
```

3. Verify installation:
```bash
nvcc --version
```

4. Re-run dependency installation:
```bash
conda activate sentry
python setup.py --install-deps
```

### Option 2: Continue Without CUDA Toolkit

If you only need transformer models and don't need DeepSpeed or Mamba-SSM, you can continue without installing the CUDA Toolkit. Everything else will work fine.

## Manual Installation Commands

If you prefer to install manually:

```bash
# For Ubuntu 24.04 (uses 22.04 repository)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0

# For Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0

# For Ubuntu 20.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
```

## Troubleshooting

### Check CUDA Toolkit Installation
```bash
nvcc --version
which nvcc
echo $CUDA_HOME
```

### Verify GPU Access
```bash
nvidia-smi
conda activate sentry
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Check Installed Packages
```bash
conda activate sentry
pip list | grep -E "(torch|deepspeed|mamba)"
```

## Disk Space Requirements

CUDA Toolkit 13.0 requires approximately **3-4 GB** of disk space.

Check available space:
```bash
df -h /usr/local
```

## Questions?

If you encounter issues, check:
1. Ubuntu version compatibility (20.04, 22.04, or 24.04)
2. Available disk space
3. Internet connection for downloads
4. Sudo/admin privileges

## Notes for Ubuntu 24.04

Ubuntu 24.04 is newer than CUDA 13.0's official release, so we use the Ubuntu 22.04 repository which is compatible. This is a standard approach and works correctly.

