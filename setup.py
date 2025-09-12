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
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
    
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        logger.info("Creating virtual environment...")
        
        if self.venv_path.exists():
            logger.info("Virtual environment already exists")
            return
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            logger.info(f"Virtual environment created at {self.venv_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            raise
    
    def install_dependencies(self):
        """Install project dependencies"""
        logger.info("Installing dependencies...")
        
        if not self.venv_path.exists():
            logger.error("Virtual environment not found. Run --create-venv first.")
            return
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / "Scripts" / "pip"
        else:  # Unix/Linux/macOS
            pip_path = self.venv_path / "bin" / "pip"
        
        try:
            # Upgrade pip first
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            subprocess.run([str(pip_path), "install", "-r", str(self.requirements_file)], check=True)
            
            logger.info("Dependencies installed successfully")
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
        """Create convenient launch scripts"""
        logger.info("Creating launch scripts...")
        
        # Determine Python path
        if os.name == 'nt':  # Windows
            python_path = self.venv_path / "Scripts" / "python"
            script_ext = ".bat"
        else:  # Unix/Linux/macOS
            python_path = self.venv_path / "bin" / "python"
            script_ext = ".sh"
        
        # Training script
        train_script = self.project_root / f"train{script_ext}"
        train_content = f"""@echo off
REM Amazing LLM Training Launcher
{str(python_path)} train.py %*
""" if os.name == 'nt' else f"""#!/bin/bash
# Amazing LLM Training Launcher
{str(python_path)} train.py "$@"
"""
        
        with open(train_script, 'w') as f:
            f.write(train_content)
        
        if os.name != 'nt':
            os.chmod(train_script, 0o755)
        
        # Inference script
        inference_script = self.project_root / f"inference{script_ext}"
        inference_content = f"""@echo off
REM Amazing LLM Inference Launcher
{str(python_path)} deployment/inference.py %*
""" if os.name == 'nt' else f"""#!/bin/bash
# Amazing LLM Inference Launcher
{str(python_path)} deployment/inference.py "$@"
"""
        
        with open(inference_script, 'w') as f:
            f.write(inference_content)
        
        if os.name != 'nt':
            os.chmod(inference_script, 0o755)
        
        logger.info("Launch scripts created")
    
    def verify_installation(self):
        """Verify that the installation is working"""
        logger.info("Verifying installation...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        # Check if virtual environment exists
        if not self.venv_path.exists():
            logger.error("Virtual environment not found")
            return False
        
        # Check if requirements are installed
        try:
            if os.name == 'nt':
                pip_path = self.venv_path / "Scripts" / "pip"
            else:
                pip_path = self.venv_path / "bin" / "pip"
            
            result = subprocess.run([str(pip_path), "list"], capture_output=True, text=True)
            if "torch" not in result.stdout:
                logger.error("PyTorch not found in virtual environment")
                return False
            
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
            self.create_virtual_environment()
            self.install_dependencies()
            self.setup_git_hooks()
            self.create_launch_scripts()
            
            if self.verify_installation():
                logger.info("Project setup completed successfully!")
                logger.info("\nNext steps:")
                logger.info("1. Activate virtual environment:")
                if os.name == 'nt':
                    logger.info("   venv\\Scripts\\activate")
                else:
                    logger.info("   source venv/bin/activate")
                logger.info("2. Download data: python train.py --pipeline data --download-fineweb")
                logger.info("3. Start training: python train.py --pipeline complete --architecture transformer")
            else:
                logger.error("Setup verification failed. Please check the installation.")
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Setup Amazing LLM Project")
    parser.add_argument("--create-venv", action="store_true", help="Create virtual environment")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--setup-dirs", action="store_true", help="Setup directories")
    parser.add_argument("--setup-git", action="store_true", help="Setup Git hooks")
    parser.add_argument("--create-scripts", action="store_true", help="Create launch scripts")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--setup-all", action="store_true", help="Run complete setup")
    
    args = parser.parse_args()
    
    setup = ProjectSetup()
    
    if args.setup_all:
        setup.setup_all()
    else:
        if args.setup_dirs:
            setup.setup_directories()
        if args.create_venv:
            setup.create_virtual_environment()
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
