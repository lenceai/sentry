#!/usr/bin/env python3
"""
Custom Dataset Processing for Domain-Specific Fine-tuning
Version: v1.0.0

This script handles custom datasets for domain-specific fine-tuning,
such as autoimmune disease research papers, medical literature, etc.

Usage:
    python data/custom_dataset.py --input_dir ./research_papers --domain medical --output_dir ./data/medical
"""

import argparse
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDatasetProcessor:
    """Processes custom datasets for domain-specific fine-tuning"""
    
    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def process_text_file(self, file_path: str) -> str:
        """Process a single text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean text
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        content = content.strip()
        
        return content
    
    def process_pdf_text(self, text: str) -> str:
        """Process text extracted from PDF"""
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[^\w\s.,!?;:()\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def process_json_data(self, data: Union[Dict, List]) -> List[str]:
        """Process JSON data structure"""
        texts = []
        
        if isinstance(data, dict):
            # Extract text from common fields
            text_fields = ['text', 'content', 'abstract', 'body', 'description']
            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    texts.append(data[field])
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    texts.extend(self.process_json_data(item))
                elif isinstance(item, str):
                    texts.append(item)
        
        return texts
    
    def create_training_pairs(self, texts: List[str], domain: str) -> List[Dict]:
        """Create training pairs for domain-specific fine-tuning"""
        training_data = []
        
        for i, text in enumerate(texts):
            if len(text) < 100:  # Skip very short texts
                continue
                
            # Create different types of training examples based on domain
            if domain == "medical":
                # Create Q&A pairs for medical literature
                training_data.append({
                    "instruction": "Analyze this medical text and provide key insights:",
                    "input": text[:500],  # First 500 chars as context
                    "output": text[500:1000],  # Next 500 chars as response
                    "domain": domain
                })
            elif domain == "research":
                # Create abstract summarization tasks
                training_data.append({
                    "instruction": "Summarize the key findings of this research:",
                    "input": text,
                    "output": text[:200] + "...",  # Truncated summary
                    "domain": domain
                })
            else:
                # Generic instruction following
                training_data.append({
                    "instruction": "Continue this text in a coherent manner:",
                    "input": text[:len(text)//2],
                    "output": text[len(text)//2:],
                    "domain": domain
                })
        
        return training_data
    
    def process_directory(self, input_dir: str, domain: str, output_dir: str):
        """Process all files in a directory"""
        input_path = Path(input_dir)
        all_texts = []
        
        # Process different file types
        for file_path in input_path.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.txt':
                        text = self.process_text_file(str(file_path))
                        all_texts.append(text)
                    elif file_path.suffix.lower() == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        texts = self.process_json_data(data)
                        all_texts.extend(texts)
                    elif file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(file_path)
                        # Assume text is in a column called 'text' or 'content'
                        text_col = 'text' if 'text' in df.columns else 'content'
                        if text_col in df.columns:
                            texts = df[text_col].dropna().astype(str).tolist()
                            all_texts.extend(texts)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
        
        logger.info(f"Collected {len(all_texts)} text samples")
        
        # Create training data
        training_data = self.create_training_pairs(all_texts, domain)
        
        # Tokenize
        tokenized_data = []
        for item in tqdm(training_data, desc="Tokenizing"):
            # Create full prompt
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            
            # Tokenize
            tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            tokenized_data.append({
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
                "domain": item["domain"]
            })
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataset
        dataset = Dataset.from_list(tokenized_data)
        dataset.save_to_disk(os.path.join(output_dir, "processed"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        # Save metadata
        metadata = {
            "domain": domain,
            "total_samples": len(tokenized_data),
            "max_length": self.max_length,
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "source_directory": str(input_dir)
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Custom dataset saved to {output_dir}")
        logger.info(f"Processed {len(tokenized_data)} training samples")

def main():
    parser = argparse.ArgumentParser(description="Process custom datasets for domain-specific fine-tuning")
    parser.add_argument("--input_dir", required=True, help="Input directory containing data files")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., medical, research, legal)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer to use")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create processor
    processor = CustomDatasetProcessor(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length
    )
    
    # Process dataset
    processor.process_directory(
        input_dir=args.input_dir,
        domain=args.domain,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
