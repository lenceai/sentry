#!/usr/bin/env python3
"""
FineWeb-Edu Streaming Download (Efficient)
Version: v1.0.0

This script downloads only the samples you need from FineWeb-Edu
without downloading the entire 32GB dataset first.

Usage:
    python data/download_fineweb_streaming.py --max_samples 10000 --output_dir ./data/fineweb-edu
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingFineWebProcessor:
    """Efficiently processes FineWeb-Edu with streaming"""
    
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
    
    def process_streaming_dataset(self, output_dir: str, max_samples: int = 10000):
        """Process dataset using streaming to avoid full download"""
        logger.info(f"Loading FineWeb-Edu in streaming mode (target: {max_samples:,} samples)")
        
        # Load dataset in streaming mode - this doesn't download everything!
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        
        logger.info("Processing samples from stream...")
        
        processed_data = []
        processed_count = 0
        skipped_count = 0
        
        # Process samples one by one from the stream
        progress_bar = tqdm(total=max_samples, desc="Processing samples")
        
        for example in dataset:
            if processed_count >= max_samples:
                break
                
            text = example.get("text", "")
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
                    
                    # Save in batches to avoid memory issues
                    if len(processed_data) >= 1000:
                        self._save_batch(processed_data, output_dir, processed_count)
                        processed_data = []
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        
        progress_bar.close()
        
        # Save final batch
        if processed_data:
            self._save_batch(processed_data, output_dir, processed_count, final=True)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        # Save metadata
        metadata = {
            "dataset_name": "fineweb-edu",
            "total_samples": processed_count,
            "skipped_samples": skipped_count,
            "max_length": self.max_length,
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "streaming_mode": True
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset processing complete!")
        logger.info(f"Processed: {processed_count:,} samples")
        logger.info(f"Skipped: {skipped_count:,} samples")
        logger.info(f"Saved to: {output_dir}")
    
    def _save_batch(self, data: List[Dict], output_dir: str, count: int, final: bool = False):
        """Save a batch of processed data"""
        from datasets import Dataset
        
        os.makedirs(output_dir, exist_ok=True)
        
        if final:
            # Save as final dataset
            dataset = Dataset.from_list(data)
            dataset.save_to_disk(os.path.join(output_dir, "processed"))
            logger.info(f"Final batch saved: {len(data)} samples")
        else:
            # Save as intermediate batch
            batch_dir = os.path.join(output_dir, f"batch_{count//1000}")
            dataset = Dataset.from_list(data)
            dataset.save_to_disk(batch_dir)
            logger.info(f"Batch saved: {len(data)} samples to {batch_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu efficiently with streaming")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples to process")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer to use")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create processor
    processor = StreamingFineWebProcessor(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length
    )
    
    # Process dataset
    processor.process_streaming_dataset(
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
