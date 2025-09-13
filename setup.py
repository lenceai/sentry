#!/usr/bin/env python3
"""
Setup Script for Amazing LLM Project
Version: v1.0.0

This script sets up the environment and dependencies for the Amazing LLM project.
It handles virtual environment creation, dependency installation, and initial configuration.

Usage:
    python setup.py --install-deps
    python setup.py --create-venv
    python setup.py --setup-all
"""

import argparse
import os
import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Setup manager for the Amazing LLM project"""
    
    def __init__(self, force_cuda_version=None):
        self.project_root = Path(__file__).parent
        self.conda_env_name = "LLM"
        self.requirements_file = self.project_root / "requirements.txt"
        self.cuda_version = force_cuda_version if force_cuda_version else self._detect_cuda_version()
    
    def _detect_cuda_version(self):
        """Detect CUDA version for PyTorch installation"""
        try:
            # Try to get CUDA version from nvidia-smi
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse CUDA version from nvidia-smi output
                import re
                cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
                    logger.info(f"Detected CUDA version: {cuda_version}")
                    
                    # Map to PyTorch CUDA versions
                    if cuda_version.startswith("13.0") or cuda_version.startswith("12.6"):
                        return "cu126"  # PyTorch 2.7.1 supports cu126 for CUDA 12.6+
                    elif cuda_version.startswith("12.4") or cuda_version.startswith("12.1") or cuda_version.startswith("12.2"):
                        return "cu121"
                    elif cuda_version.startswith("11.8"):
                        return "cu118"
                    elif cuda_version.startswith("11.7"):
                        return "cu117"
                    else:
                        logger.info(f"CUDA version {cuda_version} detected, using cu126 (latest compatible)")
                        return "cu126"
            
            logger.warning("Could not detect CUDA version, defaulting to cu118")
            return "cu118"
            
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, assuming CPU-only installation")
            return "cpu"
        except Exception as e:
            logger.warning(f"Error detecting CUDA version: {e}, defaulting to cu118")
            return "cu118"
    
    def _get_conda_command(self):
        """Get conda command path"""
        # First try to find conda in PATH
        try:
            result = subprocess.run(["which", "conda"], capture_output=True, text=True, check=True)
            conda_path = result.stdout.strip()
            if conda_path and os.path.exists(conda_path):
                logger.info(f"Found conda at: {conda_path}")
                return conda_path
        except subprocess.CalledProcessError:
            pass
        
        # Fallback to common conda installation paths
        conda_paths = [
            "/home/lence/miniconda3/bin/conda",
            "/home/lence/anaconda3/bin/conda", 
            "/opt/miniconda3/bin/conda",
            "/opt/anaconda3/bin/conda"
        ]
        
        for path in conda_paths:
            if os.path.exists(path):
                logger.info(f"Found conda at: {path}")
                return path
        
        raise FileNotFoundError("Conda not found in PATH or common installation directories")
    
    def _setup_cuda_environment(self):
        """Setup CUDA environment variables"""
        cuda_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/local/cuda-13.0",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-11.8"
        ]
        
        # Try to find CUDA installation
        cuda_home = None
        for path in cuda_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "bin", "nvcc")):
                cuda_home = path
                break
        
        if cuda_home:
            logger.info(f"Found CUDA at: {cuda_home}")
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["CUDA_PATH"] = cuda_home
            # Add CUDA to PATH
            cuda_bin = os.path.join(cuda_home, "bin")
            if cuda_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
            return cuda_home
        else:
            logger.warning("CUDA installation not found. DeepSpeed will be skipped.")
            return None
    
    def create_conda_environment(self):
        """Create conda environment"""
        logger.info(f"Creating conda environment: {self.conda_env_name}")
        
        # Get conda command path
        try:
            conda_cmd = self._get_conda_command()
        except FileNotFoundError as e:
            logger.error("Conda is not installed or not in PATH. Please install Anaconda or Miniconda first.")
            raise
        
        # Check if conda is available
        try:
            subprocess.run([conda_cmd, "--version"], check=True, capture_output=True)
            logger.info("Conda is available and working")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Conda is not working properly. Please check your conda installation.")
            raise
        
        # Check if environment already exists
        try:
            result = subprocess.run([conda_cmd, "env", "list"], capture_output=True, text=True, check=True)
            if self.conda_env_name in result.stdout:
                logger.info(f"Conda environment '{self.conda_env_name}' already exists")
                return
        except subprocess.CalledProcessError:
            pass
        
        # Accept conda TOS if needed
        logger.info("Accepting conda Terms of Service...")
        try:
            subprocess.run([conda_cmd, "tos", "accept", "--override-channels", 
                           "--channel", "https://repo.anaconda.com/pkgs/main"], 
                         check=False, capture_output=True)
            subprocess.run([conda_cmd, "tos", "accept", "--override-channels", 
                           "--channel", "https://repo.anaconda.com/pkgs/r"], 
                         check=False, capture_output=True)
        except Exception:
            pass  # TOS acceptance might not be needed or already done
        
        try:
            # Create conda environment with Python 3.11
            logger.info("Creating conda environment...")
            result = subprocess.run([conda_cmd, "create", "-n", self.conda_env_name, "python=3.11", "-y"], 
                                  check=True, capture_output=True, text=True)
            logger.info(f"Conda environment '{self.conda_env_name}' created successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create conda environment: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"STDERR: {e.stderr}")
            raise
    
    def install_dependencies(self):
        """Install project dependencies using pip in conda environment"""
        logger.info("Installing dependencies...")
        
        # Get conda command path
        try:
            conda_cmd = self._get_conda_command()
        except FileNotFoundError as e:
            logger.error("Conda is not available. Please check your conda installation.")
            raise
        
        # Check if conda environment exists
        try:
            result = subprocess.run([conda_cmd, "env", "list"], capture_output=True, text=True, check=True)
            if self.conda_env_name not in result.stdout:
                logger.error(f"Conda environment '{self.conda_env_name}' not found. Run --create-env first.")
                return
        except subprocess.CalledProcessError:
            logger.error("Failed to check conda environments")
            return
        
        try:
            # Use conda run to execute pip in the LLM environment
            # First upgrade pip
            logger.info("Upgrading pip...")
            subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "install", "--upgrade", "pip"], 
                         check=True)
            
            # Install PyTorch with appropriate CUDA support
            if self.cuda_version == "cpu":
                logger.info("Installing PyTorch 2.7.1 (CPU-only)...")
                subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "install", 
                              "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
                              "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
            else:
                logger.info(f"Installing PyTorch 2.7.1 with CUDA support ({self.cuda_version})...")
                subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "install", 
                              "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
                              "--index-url", f"https://download.pytorch.org/whl/{self.cuda_version}"], check=True)
            
            # Install remaining requirements using pip (excluding PyTorch)
            logger.info("Installing remaining dependencies...")
            requirements_no_pytorch = self.project_root / "requirements-no-pytorch.txt"
            subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "install", "-r", str(requirements_no_pytorch)], 
                         check=True)
            
            # Try to install optional CUDA-dependent packages
            cuda_home = self._setup_cuda_environment()
            
            # Try to install DeepSpeed
            logger.info("Attempting to install DeepSpeed...")
            if cuda_home:
                try:
                    # Set CUDA environment for DeepSpeed installation
                    env = os.environ.copy()
                    env["CUDA_HOME"] = cuda_home
                    env["CUDA_PATH"] = cuda_home
                    
                    subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "install", "deepspeed>=0.12.0"], 
                                 check=True, env=env)
                    logger.info("DeepSpeed installed successfully")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"DeepSpeed installation failed: {e}")
                    logger.warning("Continuing without DeepSpeed - you can install it manually later if needed")
            else:
                logger.warning("Skipping DeepSpeed installation - CUDA development tools not found")
            
            # Try to install Mamba-SSM and dependencies
            logger.info("Attempting to install Mamba-SSM...")
            if cuda_home:
                try:
                    # Set CUDA environment for Mamba installation
                    env = os.environ.copy()
                    env["CUDA_HOME"] = cuda_home
                    env["CUDA_PATH"] = cuda_home
                    
                    # Install numpy first to avoid the warning
                    subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "install", "numpy"], 
                                 check=True)
                    
                    # Install causal-conv1d first (dependency of mamba-ssm)
                    subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "install", "causal-conv1d>=1.0.0"], 
                                 check=True, env=env)
                    
                    # Install mamba-ssm
                    subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "install", "mamba-ssm>=1.2.0"], 
                                 check=True, env=env)
                    logger.info("Mamba-SSM installed successfully")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Mamba-SSM installation failed: {e}")
                    logger.warning("Continuing without Mamba-SSM - you can use transformer architecture instead")
            else:
                logger.warning("Skipping Mamba-SSM installation - CUDA development tools not found")
                logger.info("You can still use transformer architecture for training")
            
            logger.info("Dependencies installed successfully using pip in conda environment")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise
    
    def setup_directories(self):
        """Create necessary directories"""
        logger.info("Setting up project directories...")
        
        directories = [
            "data/cache",
            "data/fineweb",
            "data/fineweb-edu",
            "data/medical",
            "models/checkpoints",
            "models/transformer-base",
            "models/transformer-edu",
            "models/transformer-medical",
            "models/mamba-base",
            "models/mamba-edu",
            "models/mamba-medical",
            "logs",
            "configs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def setup_git_hooks(self):
        """Setup Git hooks for code quality"""
        logger.info("Setting up Git hooks...")
        
        git_hooks_dir = self.project_root / ".git" / "hooks"
        if not git_hooks_dir.exists():
            logger.warning("Git repository not found. Skipping Git hooks setup.")
            return
        
        # Pre-commit hook
        pre_commit_hook = git_hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# Pre-commit hook for Amazing LLM project

echo "Running pre-commit checks..."

# Check Python syntax
python -m py_compile $(find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*")

if [ $? -ne 0 ]; then
    echo "Python syntax errors found. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed!"
"""
        
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        
        os.chmod(pre_commit_hook, 0o755)
        logger.info("Git hooks setup completed")
    
    def create_launch_scripts(self):
        """Create convenient launch scripts using conda"""
        logger.info("Creating launch scripts...")
        
        # Determine script extension
        script_ext = ".bat" if os.name == 'nt' else ".sh"
        
        # Training script
        train_script = self.project_root / f"train{script_ext}"
        train_content = f"""@echo off
REM Amazing LLM Training Launcher
conda run -n {self.conda_env_name} python train.py %*
""" if os.name == 'nt' else f"""#!/bin/bash
# Amazing LLM Training Launcher
conda run -n {self.conda_env_name} python train.py "$@"
"""
        
        with open(train_script, 'w') as f:
            f.write(train_content)
        
        if os.name != 'nt':
            os.chmod(train_script, 0o755)
        
        # Inference script
        inference_script = self.project_root / f"inference{script_ext}"
        inference_content = f"""@echo off
REM Amazing LLM Inference Launcher
conda run -n {self.conda_env_name} python deployment/inference.py %*
""" if os.name == 'nt' else f"""#!/bin/bash
# Amazing LLM Inference Launcher
conda run -n {self.conda_env_name} python deployment/inference.py "$@"
"""
        
        with open(inference_script, 'w') as f:
            f.write(inference_content)
        
        if os.name != 'nt':
            os.chmod(inference_script, 0o755)
        
        # Activation helper script
        activate_script = self.project_root / f"activate_env{script_ext}"
        activate_content = f"""@echo off
REM Activate LLM conda environment
conda activate {self.conda_env_name}
""" if os.name == 'nt' else f"""#!/bin/bash
# Activate LLM conda environment
echo "Activating conda environment: {self.conda_env_name}"
echo "Run: conda activate {self.conda_env_name}"
"""
        
        with open(activate_script, 'w') as f:
            f.write(activate_content)
        
        if os.name != 'nt':
            os.chmod(activate_script, 0o755)
        
        logger.info("Launch scripts created")
    
    def verify_installation(self):
        """Verify that the installation is working"""
        logger.info("Verifying installation...")
        
        # Get conda command path
        try:
            conda_cmd = self._get_conda_command()
        except FileNotFoundError as e:
            logger.error("Conda is not available. Please check your conda installation.")
            return False
        
        # Check Python version in conda environment
        try:
            result = subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "python", "--version"], 
                                  capture_output=True, text=True, check=True)
            
            python_version = result.stdout.strip()
            logger.info(f"Python version in conda environment: {python_version}")
            
        except subprocess.CalledProcessError:
            logger.error(f"Failed to check Python version in conda environment '{self.conda_env_name}'")
            return False
        
        # Check if conda environment exists
        try:
            result = subprocess.run([conda_cmd, "env", "list"], capture_output=True, text=True, check=True)
            if self.conda_env_name not in result.stdout:
                logger.error(f"Conda environment '{self.conda_env_name}' not found")
                return False
        except subprocess.CalledProcessError:
            logger.error("Failed to check conda environments")
            return False
        
        # Check if requirements are installed
        try:
            result = subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "pip", "list"], 
                                  capture_output=True, text=True, check=True)
            
            if "torch" not in result.stdout:
                logger.error(f"PyTorch not found in conda environment '{self.conda_env_name}'")
                return False
            
            # Check PyTorch version
            import re
            torch_match = re.search(r'torch\s+(\d+\.\d+\.\d+)', result.stdout)
            if torch_match:
                torch_version = torch_match.group(1)
                logger.info(f"PyTorch version: {torch_version}")
                if not torch_version.startswith("2.7"):
                    logger.warning(f"Expected PyTorch 2.7.x, found {torch_version}")
            
            # Test CUDA availability
            logger.info("Testing CUDA availability...")
            cuda_test = subprocess.run([conda_cmd, "run", "-n", self.conda_env_name, "python", "-c", 
                                      "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"], 
                                     capture_output=True, text=True)
            
            if cuda_test.returncode == 0:
                logger.info(f"CUDA test result: {cuda_test.stdout.strip()}")
            else:
                logger.warning("CUDA test failed, but installation may still be valid for CPU usage")
            
            logger.info("Installation verification passed")
            return True
        except Exception as e:
            logger.error(f"Installation verification failed: {e}")
            return False
    
    def setup_all(self):
        """Run complete setup"""
        logger.info("Starting complete project setup...")
        
        try:
            self.setup_directories()
            self.create_conda_environment()
            self.install_dependencies()
            self.setup_git_hooks()
            self.create_launch_scripts()
            
            if self.verify_installation():
                logger.info("Project setup completed successfully!")
                logger.info("\nNext steps:")
                logger.info("1. Activate conda environment:")
                logger.info(f"   conda activate {self.conda_env_name}")
                logger.info("2. Download data: python train.py --pipeline data --download-fineweb")
                logger.info("3. Start training: python train.py --pipeline complete --architecture transformer")
                logger.info("\nAlternatively, you can use the provided scripts:")
                logger.info("   ./train.sh --pipeline complete --architecture transformer")
            else:
                logger.error("Setup verification failed. Please check the installation.")
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Setup Amazing LLM Project")
    parser.add_argument("--create-env", action="store_true", help="Create conda environment")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--setup-dirs", action="store_true", help="Setup directories")
    parser.add_argument("--setup-git", action="store_true", help="Setup Git hooks")
    parser.add_argument("--create-scripts", action="store_true", help="Create launch scripts")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--setup-all", action="store_true", help="Run complete setup")
    parser.add_argument("--cuda-version", choices=["cu118", "cu121", "cu126", "cpu"], 
                       help="Force specific CUDA version for PyTorch installation")
    
    args = parser.parse_args()
    
    setup = ProjectSetup(force_cuda_version=args.cuda_version)
    
    if args.setup_all:
        setup.setup_all()
    else:
        if args.setup_dirs:
            setup.setup_directories()
        if args.create_env:
            setup.create_conda_environment()
        if args.install_deps:
            setup.install_dependencies()
        if args.setup_git:
            setup.setup_git_hooks()
        if args.create_scripts:
            setup.create_launch_scripts()
        if args.verify:
            setup.verify_installation()

if __name__ == "__main__":
    main()
