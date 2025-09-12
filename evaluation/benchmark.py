#!/usr/bin/env python3
"""
LLM Evaluation and Benchmarking Script
Version: v1.0.0

This script evaluates trained models on various benchmarks including:
- Language modeling perplexity
- Common sense reasoning
- Domain-specific tasks
- Generation quality metrics

Usage:
    python evaluation/benchmark.py --model_path ./models/checkpoints/transformer-base --architecture transformer
    python evaluation/benchmark.py --model_path ./models/checkpoints/mamba-base --architecture mamba
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMBenchmark:
    """Comprehensive evaluation suite for language models"""
    
    def __init__(self, model_path: str, architecture: str = "transformer"):
        self.model_path = model_path
        self.architecture = architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if architecture == "transformer":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Load Mamba model
            self.model = self._load_mamba_model()
        
        self.model.eval()
        
        # Initialize metrics
        self.metrics = {
            'perplexity': evaluate.load("perplexity"),
            'rouge': evaluate.load("rouge"),
            'bleu': evaluate.load("bleu"),
            'bertscore': evaluate.load("bertscore")
        }
    
    def _load_mamba_model(self):
        """Load Mamba model from checkpoint"""
        # This would need to be implemented based on your Mamba model structure
        # For now, we'll use a placeholder
        logger.warning("Mamba model loading not fully implemented")
        return None
    
    def evaluate_perplexity(self, dataset_name: str = "wikitext", split: str = "test") -> float:
        """Evaluate perplexity on a dataset"""
        logger.info(f"Evaluating perplexity on {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        # Prepare data
        texts = [item["text"] for item in dataset if item["text"].strip()]
        
        # Calculate perplexity
        results = self.metrics['perplexity'].compute(
            model=self.model,
            tokenizer=self.tokenizer,
            add_start_token=False,
            texts=texts
        )
        
        return results['mean_perplexity']
    
    def evaluate_common_sense(self) -> Dict[str, float]:
        """Evaluate on common sense reasoning tasks"""
        logger.info("Evaluating common sense reasoning")
        
        # Load HellaSwag dataset
        dataset = load_dataset("Rowan/hellaswag", split="test")
        
        correct = 0
        total = 0
        
        for item in tqdm(dataset, desc="HellaSwag evaluation"):
            # Format as multiple choice
            context = item["ctx"]
            choices = item["endings"]
            label = item["label"]
            
            # Score each choice
            scores = []
            for choice in choices:
                prompt = f"{context} {choice}"
                score = self._score_sequence(prompt)
                scores.append(score)
            
            # Select highest scoring choice
            predicted = np.argmax(scores)
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        return {"hellaswag_accuracy": accuracy}
    
    def evaluate_medical_knowledge(self) -> Dict[str, float]:
        """Evaluate on medical knowledge tasks"""
        logger.info("Evaluating medical knowledge")
        
        # Sample medical questions (in practice, you'd load a real medical dataset)
        medical_questions = [
            {
                "question": "What is the primary cause of Type 1 diabetes?",
                "answer": "Autoimmune destruction of pancreatic beta cells",
                "choices": [
                    "Autoimmune destruction of pancreatic beta cells",
                    "Insulin resistance",
                    "Obesity",
                    "Genetic mutation only"
                ]
            },
            {
                "question": "Which antibody is most commonly associated with rheumatoid arthritis?",
                "answer": "Rheumatoid factor (RF)",
                "choices": [
                    "Rheumatoid factor (RF)",
                    "Anti-CCP",
                    "ANA",
                    "Anti-dsDNA"
                ]
            }
        ]
        
        correct = 0
        total = len(medical_questions)
        
        for item in medical_questions:
            question = item["question"]
            choices = item["choices"]
            correct_answer = item["answer"]
            
            # Score each choice
            scores = []
            for choice in choices:
                prompt = f"Question: {question}\nAnswer: {choice}"
                score = self._score_sequence(prompt)
                scores.append(score)
            
            # Select highest scoring choice
            predicted_idx = np.argmax(scores)
            predicted_answer = choices[predicted_idx]
            
            if predicted_answer == correct_answer:
                correct += 1
        
        accuracy = correct / total
        return {"medical_accuracy": accuracy}
    
    def evaluate_generation_quality(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate text generation quality"""
        logger.info("Evaluating generation quality")
        
        prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The key to understanding complex systems is",
            "When faced with difficult decisions,",
            "The most important aspect of learning is"
        ]
        
        generated_texts = []
        reference_texts = []
        
        for prompt in prompts:
            # Generate text
            generated = self._generate_text(prompt, max_length=100)
            generated_texts.append(generated)
            
            # For reference, we'll use the prompt itself (in practice, use real references)
            reference_texts.append(prompt)
        
        # Calculate metrics
        rouge_scores = self.metrics['rouge'].compute(
            predictions=generated_texts,
            references=reference_texts
        )
        
        bleu_scores = self.metrics['bleu'].compute(
            predictions=generated_texts,
            references=reference_texts
        )
        
        return {
            "rouge_1": rouge_scores["rouge1"],
            "rouge_2": rouge_scores["rouge2"],
            "rouge_l": rouge_scores["rougeL"],
            "bleu": bleu_scores["bleu"]
        }
    
    def _score_sequence(self, text: str) -> float:
        """Score a sequence using the model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Calculate log probability
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Sum log probabilities (excluding first token)
            score = log_probs[0, :-1].sum().item()
        
        return score
    
    def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()
    
    def run_full_evaluation(self) -> Dict[str, float]:
        """Run complete evaluation suite"""
        logger.info("Starting full evaluation")
        
        results = {}
        
        # Perplexity evaluation
        try:
            perplexity = self.evaluate_perplexity()
            results["perplexity"] = perplexity
        except Exception as e:
            logger.error(f"Perplexity evaluation failed: {e}")
            results["perplexity"] = None
        
        # Common sense reasoning
        try:
            common_sense = self.evaluate_common_sense()
            results.update(common_sense)
        except Exception as e:
            logger.error(f"Common sense evaluation failed: {e}")
        
        # Medical knowledge (if applicable)
        try:
            medical = self.evaluate_medical_knowledge()
            results.update(medical)
        except Exception as e:
            logger.error(f"Medical evaluation failed: {e}")
        
        # Generation quality
        try:
            generation = self.evaluate_generation_quality()
            results.update(generation)
        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}")
        
        # Log results
        logger.info("Evaluation Results:")
        for metric, value in results.items():
            if value is not None:
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save results
        results_path = os.path.join(self.model_path, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM model")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--architecture", choices=["transformer", "mamba"], default="transformer",
                       help="Model architecture")
    parser.add_argument("--benchmark", choices=["all", "perplexity", "common_sense", "medical", "generation"],
                       default="all", help="Specific benchmark to run")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = LLMBenchmark(args.model_path, args.architecture)
    
    # Run evaluation
    if args.benchmark == "all":
        results = benchmark.run_full_evaluation()
    else:
        # Run specific benchmark
        if args.benchmark == "perplexity":
            results = {"perplexity": benchmark.evaluate_perplexity()}
        elif args.benchmark == "common_sense":
            results = benchmark.evaluate_common_sense()
        elif args.benchmark == "medical":
            results = benchmark.evaluate_medical_knowledge()
        elif args.benchmark == "generation":
            results = benchmark.evaluate_generation_quality()
    
    print(f"Evaluation completed. Results: {results}")

if __name__ == "__main__":
    main()
