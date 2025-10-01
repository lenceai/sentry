#!/usr/bin/env python3
"""
FinePDFs Download and Processing Script
Version: v1.0.0

Default dataset for base training. Streams from HuggingFace to avoid large downloads.
Dataset: HuggingFaceFW/finepdfs
Docs: https://huggingface.co/datasets/HuggingFaceFW/finepdfs

Usage:
    python scripts/datasets/download_finepdfs.py --output_dir ./data/finepdfs --max_samples 100000
    python scripts/datasets/download_finepdfs.py --languages eng_Latn deu_Latn --max_samples 50000
"""

import argparse
import os
import json
from typing import Dict, Optional, Iterable, List
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinePDFsProcessor:
    """Process FinePDFs with streaming, basic cleaning, and tokenization."""

    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def clean_text(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        text = " ".join(text.split())
        if len(text) < 50:
            return None
        if len(text) > 100_000:
            text = text[:100_000]
        return text.strip()

    def tokenize_text(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "text_length": len(text),
        }

    def iter_examples(self, languages: Optional[List[str]], max_samples: int) -> Iterable[Dict]:
        # FinePDFs subsets are language codes like 'eng_Latn'. If no languages specified, default to eng_Latn.
        target_langs = languages or ["eng_Latn"]
        processed = 0

        for lang in target_langs:
            logger.info(f"Loading FinePDFs subset: {lang} (streaming)")
            ds = load_dataset("HuggingFaceFW/finepdfs", lang, split="train", streaming=True)
            for ex in ds:
                if processed >= max_samples:
                    return
                text = ex.get("text") if isinstance(ex, dict) else None
                cleaned = self.clean_text(text)
                if not cleaned:
                    continue
                tokenized = self.tokenize_text(cleaned)
                if not tokenized:
                    continue
                yield {
                    "id": processed,
                    "language": lang,
                    "text": cleaned,
                    **tokenized,
                }
                processed += 1

    def process(self, output_dir: str, languages: Optional[List[str]], max_samples: int) -> None:
        logger.info("Starting FinePDFs processing (streaming mode)...")
        os.makedirs(output_dir, exist_ok=True)

        processed = []
        progress = tqdm(total=max_samples, desc="Processing FinePDFs")
        for item in self.iter_examples(languages, max_samples):
            processed.append(item)
            progress.update(1)
        progress.close()

        from datasets import Dataset
        dataset = Dataset.from_list(processed)
        dataset.save_to_disk(os.path.join(output_dir, "processed"))

        self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        metadata = {
            "dataset_name": "HuggingFaceFW/finepdfs",
            "languages": languages or ["eng_Latn"],
            "total_samples": len(processed),
            "max_length": self.max_length,
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "source": "HuggingFaceFW/finepdfs",
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"FinePDFs processing complete. Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and process FinePDFs with streaming")
    parser.add_argument("--output_dir", required=True, help="Output directory (e.g., ./data/finepdfs)")
    parser.add_argument("--max_samples", type=int, default=100000, help="Maximum samples to process")
    parser.add_argument("--languages", nargs="*", help="Language subsets (e.g., eng_Latn deu_Latn)")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer to use")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")

    args = parser.parse_args()

    processor = FinePDFsProcessor(tokenizer_name=args.tokenizer, max_length=args.max_length)
    processor.process(output_dir=args.output_dir, languages=args.languages, max_samples=args.max_samples)


if __name__ == "__main__":
    main()


