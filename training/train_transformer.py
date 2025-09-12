#!/usr/bin/env python3
"""
Transformer-based LLM Training Script
Version: v1.0.0

This script trains transformer-based language models using distributed training
across multiple GPUs and systems. Supports both base training and fine-tuning.

Usage:
    # Base training on FineWeb
    python training/train_transformer.py --config configs/base_config.yaml --stage base
    
    # Fine-tuning on custom dataset
    python training/train_transformer.py --config configs/base_config.yaml --stage finetune --dataset_path ./data/medical
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerTrainer:
    """Trainer for transformer-based language models"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Set random seed
        set_seed(self.config.get('seed', 42))
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model'].get('tokenizer_name', 'gpt2')
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal language modeling
        )
    
    def _create_model(self):
        """Create transformer model"""
        model_config = self.config['model']
        
        config = AutoConfig.from_pretrained(
            model_config.get('base_model', 'gpt2'),
            vocab_size=model_config['vocab_size'],
            n_positions=model_config['max_seq_length'],
            n_ctx=model_config['max_seq_length'],
            n_embd=model_config['d_model'],
            n_layer=model_config['n_layers'],
            n_head=model_config['n_heads'],
            n_inner=model_config['d_ff'],
            activation_function="gelu_new",
            resid_pdrop=model_config['dropout'],
            embd_pdrop=model_config['dropout'],
            attn_pdrop=model_config['dropout'],
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        model = AutoModelForCausalLM.from_config(config)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['optimization']['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer"""
        training_config = self.config['training']
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        training_config = self.config['training']
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=training_config['warmup_steps'],
            num_training_steps=training_config['max_steps']
        )
        
        return scheduler
    
    def prepare_data(self, dataset_path: str, stage: str):
        """Prepare training data"""
        from datasets import load_from_disk
        
        # Load dataset
        if stage == "base":
            # Load FineWeb data
            dataset = load_from_disk(os.path.join(dataset_path, "processed"))
        elif stage == "finetune":
            # Load custom domain data
            dataset = load_from_disk(os.path.join(dataset_path, "processed"))
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.config['optimization']['dataloader_num_workers'],
            pin_memory=self.config['optimization']['pin_memory']
        )
        
        return dataloader
    
    def train(self, dataset_path: str, stage: str, output_dir: str):
        """Main training loop"""
        logger.info(f"Starting {stage} training")
        
        # Prepare data
        dataloader = self.prepare_data(dataset_path, stage)
        
        # Prepare model and optimizer with accelerator
        model, optimizer, dataloader, scheduler = self.accelerator.prepare(
            self.model, self.optimizer, dataloader, self.scheduler
        )
        
        # Initialize wandb if enabled
        if self.accelerator.is_main_process:
            wandb.init(
                project="amazing-llm",
                name=f"transformer-{stage}",
                config=self.config
            )
        
        # Training loop
        model.train()
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.config['training']['max_epochs']):
            for batch in dataloader:
                with self.accelerator.accumulate(model):
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Update parameters
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Logging
                    total_loss += loss.item()
                    global_step += 1
                    
                    if global_step % self.config['training']['logging_steps'] == 0:
                        avg_loss = total_loss / self.config['training']['logging_steps']
                        logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}")
                        
                        if self.accelerator.is_main_process:
                            wandb.log({
                                "loss": avg_loss,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "step": global_step
                            })
                        
                        total_loss = 0
                    
                    # Save checkpoint
                    if global_step % self.config['training']['save_steps'] == 0:
                        self.save_checkpoint(output_dir, global_step)
                    
                    # Early stopping
                    if global_step >= self.config['training']['max_steps']:
                        break
            
            if global_step >= self.config['training']['max_steps']:
                break
        
        # Save final model
        self.save_checkpoint(output_dir, "final")
        
        if self.accelerator.is_main_process:
            wandb.finish()
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, output_dir: str, step: str):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Save config
            with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train transformer-based LLM")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument("--stage", choices=["base", "finetune"], required=True,
                       help="Training stage")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--resume_from", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = TransformerTrainer(args.config)
    
    # Start training
    trainer.train(
        dataset_path=args.dataset_path,
        stage=args.stage,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
