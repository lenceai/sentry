#!/usr/bin/env python3
"""
Distributed Training Launcher
Version: v1.0.0

This script launches distributed training across multiple systems and GPUs.
It handles the coordination between the RTX 3080 TI and 3x RTX 3090 systems.

Usage:
    # Launch transformer training
    python training/launch_distributed.py --config configs/base_config.yaml --architecture transformer --stage base
    
    # Launch Mamba training
    python training/launch_distributed.py --config configs/mamba_config.yaml --architecture mamba --stage base
"""

import argparse
import os
import subprocess
import yaml
import logging
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedLauncher:
    """Launches distributed training across multiple systems"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # System configurations
        self.systems = [
            {
                "name": "system_1",
                "host": "192.168.1.100",  # RTX 3080 TI
                "gpu": "RTX_3080_TI",
                "rank": 0,
                "cuda_devices": "0"
            },
            {
                "name": "system_2", 
                "host": "192.168.1.101",  # RTX 3090
                "gpu": "RTX_3090",
                "rank": 1,
                "cuda_devices": "0"
            },
            {
                "name": "system_3",
                "host": "192.168.1.102",  # RTX 3090
                "gpu": "RTX_3090", 
                "rank": 2,
                "cuda_devices": "0"
            },
            {
                "name": "system_4",
                "host": "192.168.1.103",  # RTX 3090
                "gpu": "RTX_3090",
                "rank": 3,
                "cuda_devices": "0"
            }
        ]
    
    def create_launch_script(self, architecture: str, stage: str, dataset_path: str, output_dir: str) -> str:
        """Create launch script for distributed training"""
        
        if architecture == "transformer":
            train_script = "training/train_transformer.py"
        elif architecture == "mamba":
            train_script = "training/train_mamba.py"
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        script_content = f"""#!/bin/bash
# Distributed Training Launch Script
# Architecture: {architecture}
# Stage: {stage}

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# Master node (system_1)
if [ "$SLURM_PROCID" = "0" ]; then
    echo "Starting master node on system_1"
    python {train_script} \\
        --config {self.config_path} \\
        --dataset_path {dataset_path} \\
        --stage {stage} \\
        --output_dir {output_dir} \\
        --master_addr {self.systems[0]['host']} \\
        --master_port {self.config['distributed']['master_port']} \\
        --world_size {len(self.systems)} \\
        --rank 0
else
    echo "Starting worker node rank $SLURM_PROCID"
    python {train_script} \\
        --config {self.config_path} \\
        --dataset_path {dataset_path} \\
        --stage {stage} \\
        --output_dir {output_dir} \\
        --master_addr {self.systems[0]['host']} \\
        --master_port {self.config['distributed']['master_port']} \\
        --world_size {len(self.systems)} \\
        --rank $SLURM_PROCID
fi
"""
        
        script_path = f"launch_{architecture}_{stage}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def create_slurm_script(self, architecture: str, stage: str, dataset_path: str, output_dir: str) -> str:
        """Create SLURM job script for cluster execution"""
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=llm-{architecture}-{stage}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Create logs directory
mkdir -p logs

# Load modules
module load cuda/11.8
module load python/3.9

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# Launch distributed training
srun python training/launch_distributed.py \\
    --config {self.config_path} \\
    --architecture {architecture} \\
    --stage {stage} \\
    --dataset_path {dataset_path} \\
    --output_dir {output_dir}
"""
        
        slurm_path = f"slurm_{architecture}_{stage}.sh"
        with open(slurm_path, 'w') as f:
            f.write(slurm_content)
        
        return slurm_path
    
    def create_docker_compose(self, architecture: str, stage: str, dataset_path: str, output_dir: str) -> str:
        """Create Docker Compose file for containerized distributed training"""
        
        compose_content = f"""version: '3.8'

services:
  system1:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    command: python training/train_{architecture}.py --config {self.config_path} --dataset_path {dataset_path} --stage {stage} --output_dir {output_dir} --master_addr system1 --master_port {self.config['distributed']['master_port']} --world_size 4 --rank 0
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NCCL_DEBUG=INFO
    volumes:
      - .:/workspace
      - {dataset_path}:/data
      - {output_dir}:/output
    working_dir: /workspace
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  system2:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    command: python training/train_{architecture}.py --config {self.config_path} --dataset_path {dataset_path} --stage {stage} --output_dir {output_dir} --master_addr system1 --master_port {self.config['distributed']['master_port']} --world_size 4 --rank 1
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NCCL_DEBUG=INFO
    volumes:
      - .:/workspace
      - {dataset_path}:/data
      - {output_dir}:/output
    working_dir: /workspace
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  system3:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    command: python training/train_{architecture}.py --config {self.config_path} --dataset_path {dataset_path} --stage {stage} --output_dir {output_dir} --master_addr system1 --master_port {self.config['distributed']['master_port']} --world_size 4 --rank 2
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NCCL_DEBUG=INFO
    volumes:
      - .:/workspace
      - {dataset_path}:/data
      - {output_dir}:/output
    working_dir: /workspace
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  system4:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    command: python training/train_{architecture}.py --config {self.config_path} --dataset_path {dataset_path} --stage {stage} --output_dir {output_dir} --master_addr system1 --master_port {self.config['distributed']['master_port']} --world_size 4 --rank 3
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NCCL_DEBUG=INFO
    volumes:
      - .:/workspace
      - {dataset_path}:/data
      - {output_dir}:/output
    working_dir: /workspace
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
        
        compose_path = f"docker-compose-{architecture}-{stage}.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        return compose_path
    
    def launch_training(self, architecture: str, stage: str, dataset_path: str, output_dir: str, method: str = "manual"):
        """Launch distributed training using specified method"""
        
        logger.info(f"Launching {architecture} {stage} training using {method} method")
        
        if method == "manual":
            # Manual SSH-based launch
            self._launch_manual(architecture, stage, dataset_path, output_dir)
        elif method == "slurm":
            # SLURM cluster launch
            slurm_script = self.create_slurm_script(architecture, stage, dataset_path, output_dir)
            subprocess.run(["sbatch", slurm_script])
        elif method == "docker":
            # Docker Compose launch
            compose_file = self.create_docker_compose(architecture, stage, dataset_path, output_dir)
            subprocess.run(["docker-compose", "-f", compose_file, "up"])
        else:
            raise ValueError(f"Unknown launch method: {method}")
    
    def _launch_manual(self, architecture: str, stage: str, dataset_path: str, output_dir: str):
        """Launch training manually via SSH"""
        
        # Create launch script
        script_path = self.create_launch_script(architecture, stage, dataset_path, output_dir)
        
        # Launch on each system
        for i, system in enumerate(self.systems):
            if i == 0:
                # Master node - run locally
                logger.info(f"Starting master on {system['name']}")
                subprocess.Popen([f"./{script_path}"])
            else:
                # Worker nodes - run via SSH
                logger.info(f"Starting worker on {system['name']} ({system['host']})")
                ssh_cmd = [
                    "ssh", system['host'],
                    f"cd /path/to/project && ./{script_path}"
                ]
                subprocess.Popen(ssh_cmd)

def main():
    parser = argparse.ArgumentParser(description="Launch distributed training")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--architecture", choices=["transformer", "mamba"], required=True,
                       help="Model architecture")
    parser.add_argument("--stage", choices=["base", "finetune"], required=True,
                       help="Training stage")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--method", choices=["manual", "slurm", "docker"], default="manual",
                       help="Launch method")
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = DistributedLauncher(args.config)
    
    # Launch training
    launcher.launch_training(
        architecture=args.architecture,
        stage=args.stage,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        method=args.method
    )

if __name__ == "__main__":
    main()
