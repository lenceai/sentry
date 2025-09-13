#!/usr/bin/env python3
"""
SmolLM-Corpus Download Script
Version: v1.0.0

Download educational data from SmolLM-Corpus - much more efficient than FineWeb!
Based on: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

Usage:
    python data/download_smollm.py --subset fineweb-edu-dedup --max_samples 10000
    python data/download_smollm.py --subset cosmopedia-v2 --max_samples 5000
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Optional
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmolLMProcessor:
    """Process SmolLM-Corpus datasets efficiently"""
    
    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not text or len(text) < 50:
            return None
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Limit very long texts
        if len(text) > 50000:
            text = text[:50000]
            
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
    
    def process_dataset(self, subset: str, output_dir: str, max_samples: Optional[int] = None):
        """Process SmolLM-Corpus dataset"""
        logger.info(f"Loading SmolLM-Corpus subset: {subset}")
        
        # Available subsets
        valid_subsets = ["fineweb-edu-dedup", "cosmopedia-v2", "python-edu"]
        if subset not in valid_subsets:
            raise ValueError(f"Unknown subset: {subset}. Available: {valid_subsets}")
        
        try:
            # Load dataset in streaming mode to avoid huge downloads
            dataset = load_dataset("HuggingFaceTB/smollm-corpus", subset, split="train", streaming=True)
            logger.info(f"Successfully loaded {subset} in streaming mode")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            # Fallback to regular loading if streaming fails
            logger.info("Trying regular loading...")
            dataset = load_dataset("HuggingFaceTB/smollm-corpus", subset, split="train")
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Process samples
        processed_data = []
        processed_count = 0
        skipped_count = 0
        
        target_samples = max_samples if max_samples else 10000
        progress_bar = tqdm(total=target_samples, desc=f"Processing {subset}")
        
        for example in dataset:
            if processed_count >= target_samples:
                break
            
            # Extract text based on subset format
            if subset == "fineweb-edu-dedup":
                text = example.get("text", "")
            elif subset == "cosmopedia-v2":
                text = example.get("text", "")
            elif subset == "python-edu":
                # For python-edu, we need to download the actual content
                text = example.get("text", "") or "# Python code content"
            else:
                text = str(example)
            
            cleaned_text = self.clean_text(text)
            
            if cleaned_text:
                tokenized = self.tokenize_text(cleaned_text)
                if tokenized:
                    processed_data.append({
                        "id": processed_count,
                        "text": cleaned_text,
                        **tokenized
                    })
                    processed_count += 1
                    progress_bar.update(1)
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        
        progress_bar.close()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as HuggingFace dataset
        from datasets import Dataset
        processed_dataset = Dataset.from_list(processed_data)
        processed_dataset.save_to_disk(os.path.join(output_dir, "processed"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        # Save metadata
        metadata = {
            "dataset_name": f"smollm-corpus-{subset}",
            "total_samples": processed_count,
            "skipped_samples": skipped_count,
            "max_length": self.max_length,
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "source": "HuggingFaceTB/smollm-corpus"
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset processing complete!")
        logger.info(f"Processed: {processed_count:,} samples")
        logger.info(f"Skipped: {skipped_count:,} samples")
        logger.info(f"Saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download and process SmolLM-Corpus datasets")
    parser.add_argument("--subset", required=True, 
                       choices=["fineweb-edu-dedup", "cosmopedia-v2", "python-edu"],
                       help="SmolLM-Corpus subset to download")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer to use")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create processor
    processor = SmolLMProcessor(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length
    )
    
    # Process dataset
    processor.process_dataset(
        subset=args.subset,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
