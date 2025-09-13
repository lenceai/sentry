#!/usr/bin/env python3
"""
FineWeb Data Download and Processing Pipeline
Version: v1.0.0

This script downloads and processes FineWeb and FineWeb-Edu datasets
for training our custom LLM. It handles data cleaning, tokenization,
and preparation for both transformer and mamba architectures.

Usage:
    python data/download_fineweb.py --dataset fineweb --output_dir ./data/fineweb
    python data/download_fineweb.py --dataset fineweb-edu --output_dir ./data/fineweb-edu
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import webdataset as wds
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineWebProcessor:
    """Processes FineWeb datasets for LLM training"""
    
    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove very short or very long texts
        if len(text) < 50 or len(text) > 100000:
            return None
            
        return text.strip()
    
    def tokenize_text(self, text: str) -> Dict:
        """Tokenize text and return tokenized data"""
        if not text:
            return None
            
        # Tokenize
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "text_length": len(text)
        }
    
    def process_dataset(self, dataset_name: str, output_dir: str, max_samples: Optional[int] = None):
        """Process and save dataset"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        if dataset_name == "fineweb":
            dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train")
        elif dataset_name == "fineweb-edu":
            # Use SmolLM-Corpus FineWeb-Edu (deduplicated) - much more efficient!
            dataset = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train")
        elif dataset_name == "cosmopedia":
            # SmolLM synthetic educational content
            dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train")
        elif dataset_name == "python-edu":
            # SmolLM high-quality Python code
            dataset = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Processing {len(dataset)} samples")
        
        # Process data
        processed_data = []
        for i, sample in enumerate(tqdm(dataset, desc="Processing")):
            text = sample.get("text", "")
            cleaned_text = self.clean_text(text)
            
            if cleaned_text:
                tokenized = self.tokenize_text(cleaned_text)
                if tokenized:
                    processed_data.append({
                        "id": i,
                        "text": cleaned_text,
                        **tokenized
                    })
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as HuggingFace dataset
        processed_dataset = Dataset.from_list(processed_data)
        processed_dataset.save_to_disk(os.path.join(output_dir, "processed"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "total_samples": len(processed_data),
            "max_length": self.max_length,
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}")
        logger.info(f"Processed {len(processed_data)} samples")

def main():
    parser = argparse.ArgumentParser(description="Download and process FineWeb datasets")
    parser.add_argument("--dataset", choices=["fineweb", "fineweb-edu"], required=True,
                       help="Dataset to download")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer to use")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create processor
    processor = FineWebProcessor(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length
    )
    
    # Process dataset
    processor.process_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
