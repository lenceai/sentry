#!/usr/bin/env python3
"""
Mamba Architecture Training Script
Version: v1.0.0

This script trains Mamba-based language models using the state-space architecture
for efficient long-sequence processing. Supports distributed training across multiple GPUs.

Usage:
    # Base training on FineWeb
    python training/train_mamba.py --config configs/mamba_config.yaml --stage base
    
    # Fine-tuning on custom dataset
    python training/train_mamba.py --config configs/mamba_config.yaml --stage finetune --dataset_path ./data/medical
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, TrainingArguments, Trainer
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

# Mamba imports
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MambaLMHeadModel(nn.Module):
    """Mamba Language Model with LM Head"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dt_rank=config.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            dt_init_floor=config.dt_init_floor,
            conv_bias=config.conv_bias,
            bias=config.bias,
            use_fast_path=config.use_fast_path,
            layer_idx=0,
        )
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(torch.randn(config.vocab_size, config.d_model) * 0.02)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Mamba forward pass
        hidden_states = self.backbone(input_ids)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }

class MambaTrainer:
    """Trainer for Mamba-based language models"""
    
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
    
    def _create_model(self):
        """Create Mamba model"""
        model_config = self.config['model']
        
        # Mamba configuration
        class MambaConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        config = MambaConfig(
            d_model=model_config['d_model'],
            d_state=model_config.get('d_state', 16),
            d_conv=model_config.get('d_conv', 4),
            expand=model_config.get('expand', 2),
            dt_rank=model_config.get('dt_rank', "auto"),
            dt_min=model_config.get('dt_min', 0.001),
            dt_max=model_config.get('dt_max', 0.1),
            dt_init=model_config.get('dt_init', "random"),
            dt_scale=model_config.get('dt_scale', 1.0),
            dt_init_floor=model_config.get('dt_init_floor', 1e-4),
            conv_bias=model_config.get('conv_bias', True),
            bias=model_config.get('bias', False),
            use_fast_path=model_config.get('use_fast_path', True),
            vocab_size=model_config['vocab_size'],
            max_seq_length=model_config['max_seq_length']
        )
        
        model = MambaLMHeadModel(config)
        
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
        from transformers import get_linear_schedule_with_warmup
        
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
        from torch.utils.data import DataLoader
        from transformers import DataCollatorForLanguageModeling
        
        # Load dataset
        if stage == "base":
            dataset = load_from_disk(os.path.join(dataset_path, "processed"))
        elif stage == "finetune":
            dataset = load_from_disk(os.path.join(dataset_path, "processed"))
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=data_collator,
            num_workers=self.config['optimization']['dataloader_num_workers'],
            pin_memory=self.config['optimization']['pin_memory']
        )
        
        return dataloader
    
    def train(self, dataset_path: str, stage: str, output_dir: str):
        """Main training loop"""
        logger.info(f"Starting Mamba {stage} training")
        
        # Prepare data
        dataloader = self.prepare_data(dataset_path, stage)
        
        # Prepare model and optimizer with accelerator
        model, optimizer, dataloader, scheduler = self.accelerator.prepare(
            self.model, self.optimizer, dataloader, self.scheduler
        )
        
        # Initialize wandb if enabled
        if self.accelerator.is_main_process:
            wandb.init(
                project="amazing-llm-mamba",
                name=f"mamba-{stage}",
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
                    loss = outputs['loss']
                    
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
        
        logger.info("Mamba training completed!")
    
    def save_checkpoint(self, output_dir: str, step: str):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), 
                      os.path.join(checkpoint_dir, "pytorch_model.bin"))
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Save config
            with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Mamba checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train Mamba-based LLM")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument("--stage", choices=["base", "finetune"], required=True,
                       help="Training stage")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--resume_from", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MambaTrainer(args.config)
    
    # Start training
    trainer.train(
        dataset_path=args.dataset_path,
        stage=args.stage,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
