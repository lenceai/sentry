#!/usr/bin/env python3
"""
Main Training Script for Amazing LLM
Version: v1.0.0

This is the main entry point for training your amazing LLM from scratch.
It orchestrates the entire training pipeline from data preparation to model deployment.

Usage:
    # Complete training pipeline
    python train.py --pipeline complete --architecture transformer
    
    # Data preparation only
    python train.py --pipeline data --download-fineweb
    
    # Base training only
    python train.py --pipeline base --architecture transformer --data-path ./data/fineweb
    
    # Fine-tuning only
    python train.py --pipeline finetune --architecture transformer --data-path ./data/medical --base-model ./models/transformer-base
"""

import argparse
import os
import json
import logging
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmazingLLMTrainer:
    """Main orchestrator for the Amazing LLM training pipeline"""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def prepare_data(self, download_fineweb: bool = False, download_fineweb_edu: bool = False, 
                    custom_data_path: Optional[str] = None, domain: str = "medical", use_finepdfs: bool = True,
                    finepdfs_languages: Optional[List[str]] = None, finepdfs_max_samples: int = 100000):
        """Prepare all training data"""
        logger.info("Starting data preparation...")
        
        # Default: FinePDFs for base corpus
        if use_finepdfs:
            logger.info("Downloading and processing FinePDFs dataset (default)...")
            cmd = [
                "python", "scripts/datasets/download_finepdfs.py",
                "--output_dir", str(self.data_dir / "finepdfs"),
                "--max_samples", str(finepdfs_max_samples),
            ]
            if finepdfs_languages:
                cmd.extend(["--languages", *finepdfs_languages])
            subprocess.run(cmd, check=True)

        if download_fineweb:
            logger.info("Downloading and processing FineWeb dataset...")
            cmd = [
                "python", "data/download_fineweb.py",
                "--dataset", "fineweb",
                "--output_dir", str(self.data_dir / "fineweb"),
                "--max_samples", "1000000"
            ]
            subprocess.run(cmd, check=True)
        
        if download_fineweb_edu:
            logger.info("Downloading and processing FineWeb-Edu dataset...")
            cmd = [
                "python", "data/download_fineweb.py",
                "--dataset", "fineweb-edu",
                "--output_dir", str(self.data_dir / "fineweb-edu"),
                "--max_samples", "500000"
            ]
            subprocess.run(cmd, check=True)
        
        if custom_data_path:
            logger.info(f"Processing custom dataset from {custom_data_path}...")
            cmd = [
                "python", "data/custom_dataset.py",
                "--input_dir", custom_data_path,
                "--domain", domain,
                "--output_dir", str(self.data_dir / domain)
            ]
            subprocess.run(cmd, check=True)
        
        logger.info("Data preparation completed!")
    
    def train_base_model(self, architecture: str, data_path: str, output_dir: str):
        """Train base model on FineWeb data"""
        logger.info(f"Starting base training with {architecture} architecture...")
        
        if architecture == "transformer":
            config_file = "configs/base_config.yaml"
            train_script = "training/train_transformer.py"
        elif architecture == "mamba":
            config_file = "configs/mamba_config.yaml"
            train_script = "training/train_mamba.py"
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        cmd = [
            "python", train_script,
            "--config", config_file,
            "--dataset_path", data_path,
            "--stage", "base",
            "--output_dir", output_dir
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("Base training completed!")
    
    def finetune_model(self, architecture: str, base_model_path: str, 
                      custom_data_path: str, output_dir: str):
        """Fine-tune model on custom dataset"""
        logger.info(f"Starting fine-tuning with {architecture} architecture...")
        
        if architecture == "transformer":
            config_file = "configs/base_config.yaml"
            train_script = "training/train_transformer.py"
        elif architecture == "mamba":
            config_file = "configs/mamba_config.yaml"
            train_script = "training/train_mamba.py"
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Update config for fine-tuning
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Reduce learning rate for fine-tuning
        config['training']['learning_rate'] = config['training']['learning_rate'] * 0.1
        config['training']['max_steps'] = config['training']['max_steps'] // 10
        
        # Save updated config
        finetune_config = f"configs/{architecture}_finetune_config.yaml"
        with open(finetune_config, 'w') as f:
            yaml.dump(config, f)
        
        cmd = [
            "python", train_script,
            "--config", finetune_config,
            "--dataset_path", custom_data_path,
            "--stage", "finetune",
            "--output_dir", output_dir,
            "--resume_from", base_model_path
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("Fine-tuning completed!")
    
    def evaluate_model(self, model_path: str, architecture: str):
        """Evaluate trained model"""
        logger.info(f"Evaluating {architecture} model at {model_path}...")
        
        cmd = [
            "python", "evaluation/benchmark.py",
            "--model_path", model_path,
            "--architecture", architecture,
            "--benchmark", "all"
        ]
        
        subprocess.run(cmd, check=True)
        
        logger.info("Model evaluation completed!")
    
    def deploy_model(self, model_path: str, architecture: str, port: int = 8000):
        """Deploy model as API server"""
        logger.info(f"Deploying {architecture} model as API server...")
        
        cmd = [
            "python", "deployment/inference.py",
            "--model_path", model_path,
            "--architecture", architecture,
            "--mode", "api",
            "--port", str(port)
        ]
        
        logger.info(f"Starting API server on port {port}")
        logger.info("API will be available at http://localhost:8000")
        logger.info("Use Ctrl+C to stop the server")
        
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            logger.info("API server stopped by user")
    
    def run_complete_pipeline(self, architecture: str, custom_data_path: Optional[str] = None, 
                            domain: str = "medical"):
        """Run the complete training pipeline"""
        logger.info("Starting complete Amazing LLM training pipeline!")
        
        # Step 1: Data preparation
        self.prepare_data(
            use_finepdfs=True,
            download_fineweb=False,
            download_fineweb_edu=True,
            custom_data_path=custom_data_path,
            domain=domain
        )
        
        # Step 2: Base training on FineWeb
        base_output_dir = self.models_dir / f"{architecture}-base"
        # Train base on FinePDFs by default
        self.train_base_model(
            architecture=architecture,
            data_path=str(self.data_dir / "finepdfs"),
            output_dir=str(base_output_dir)
        )
        
        # Step 3: Educational training on FineWeb-Edu
        edu_output_dir = self.models_dir / f"{architecture}-edu"
        self.train_base_model(
            architecture=architecture,
            data_path=str(self.data_dir / "fineweb-edu"),
            output_dir=str(edu_output_dir)
        )
        
        # Step 4: Fine-tuning on custom data (if provided)
        if custom_data_path:
            finetune_output_dir = self.models_dir / f"{architecture}-{domain}"
            self.finetune_model(
                architecture=architecture,
                base_model_path=str(edu_output_dir),
                custom_data_path=str(self.data_dir / domain),
                output_dir=str(finetune_output_dir)
            )
            
            # Step 5: Evaluate final model
            self.evaluate_model(str(finetune_output_dir), architecture)
            
            # Step 6: Deploy model
            self.deploy_model(str(finetune_output_dir), architecture)
        else:
            # Evaluate base model
            self.evaluate_model(str(edu_output_dir), architecture)
            
            # Deploy base model
            self.deploy_model(str(edu_output_dir), architecture)
        
        logger.info("Complete training pipeline finished!")

def main():
    parser = argparse.ArgumentParser(description="Amazing LLM Training Pipeline")
    parser.add_argument("--pipeline", choices=["complete", "data", "base", "finetune", "evaluate", "deploy"],
                       required=True, help="Pipeline stage to run")
    parser.add_argument("--architecture", choices=["transformer", "mamba"], default="transformer",
                       help="Model architecture")
    parser.add_argument("--data-path", help="Path to dataset")
    parser.add_argument("--base-model", help="Path to base model for fine-tuning")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--custom-data", help="Path to custom dataset")
    parser.add_argument("--domain", default="medical", help="Domain for custom dataset")
    parser.add_argument("--download-fineweb", action="store_true", help="Download FineWeb dataset")
    parser.add_argument("--download-fineweb-edu", action="store_true", help="Download FineWeb-Edu dataset")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--config", default="configs/base_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AmazingLLMTrainer(args.config)
    
    if args.pipeline == "complete":
        trainer.run_complete_pipeline(
            architecture=args.architecture,
            custom_data_path=args.custom_data,
            domain=args.domain
        )
    
    elif args.pipeline == "data":
        trainer.prepare_data(
            download_fineweb=args.download_fineweb,
            download_fineweb_edu=args.download_fineweb_edu,
            custom_data_path=args.custom_data,
            domain=args.domain
        )
    
    elif args.pipeline == "base":
        if not args.data_path:
            raise ValueError("--data-path is required for base training")
        if not args.output_dir:
            raise ValueError("--output-dir is required for base training")
        
        trainer.train_base_model(
            architecture=args.architecture,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
    
    elif args.pipeline == "finetune":
        if not args.base_model:
            raise ValueError("--base-model is required for fine-tuning")
        if not args.data_path:
            raise ValueError("--data-path is required for fine-tuning")
        if not args.output_dir:
            raise ValueError("--output-dir is required for fine-tuning")
        
        trainer.finetune_model(
            architecture=args.architecture,
            base_model_path=args.base_model,
            custom_data_path=args.data_path,
            output_dir=args.output_dir
        )
    
    elif args.pipeline == "evaluate":
        if not args.data_path:
            raise ValueError("--data-path is required for evaluation")
        
        trainer.evaluate_model(args.data_path, args.architecture)
    
    elif args.pipeline == "deploy":
        if not args.data_path:
            raise ValueError("--data-path is required for deployment")
        
        trainer.deploy_model(args.data_path, args.architecture, args.port)

if __name__ == "__main__":
    main()
