#!/usr/bin/env python3
"""
LLM Inference and Deployment Script
Version: v1.0.0

This script provides inference capabilities for trained models with support for:
- Single text generation
- Batch processing
- API server deployment
- Interactive chat interface

Usage:
    # Single inference
    python deployment/inference.py --model_path ./models/transformer-base --text "Hello, how are you?"
    
    # Start API server
    python deployment/inference.py --model_path ./models/transformer-base --mode api --port 8000
    
    # Interactive chat
    python deployment/inference.py --model_path ./models/transformer-base --mode chat
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import gradio as gr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    text: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

class InferenceResponse(BaseModel):
    generated_text: str
    input_text: str
    generation_time: float
    model_info: Dict

class LLMInference:
    """High-performance inference engine for trained LLMs"""
    
    def __init__(self, model_path: str, architecture: str = "transformer", device: str = "auto"):
        self.model_path = model_path
        self.architecture = architecture
        self.device = self._get_device(device)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if architecture == "transformer":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if device == "auto" else None
            )
            if device != "auto":
                self.model = self.model.to(self.device)
        else:
            # Load Mamba model
            self.model = self._load_mamba_model()
        
        self.model.eval()
        
        # Thread pool for async inference
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Model loaded on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_mamba_model(self):
        """Load Mamba model from checkpoint"""
        # This would need to be implemented based on your Mamba model structure
        logger.warning("Mamba model loading not fully implemented")
        return None
    
    def generate_text(
        self,
        text: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """Generate text from input prompt"""
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input text from generated text
        if generated_text.startswith(text):
            generated_text = generated_text[len(text):].strip()
        
        return generated_text
    
    async def generate_async(self, request: InferenceRequest) -> InferenceResponse:
        """Asynchronous text generation"""
        import time
        start_time = time.time()
        
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        generated_text = await loop.run_in_executor(
            self.executor,
            self.generate_text,
            request.text,
            request.max_length,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repetition_penalty
        )
        
        generation_time = time.time() - start_time
        
        return InferenceResponse(
            generated_text=generated_text,
            input_text=request.text,
            generation_time=generation_time,
            model_info={
                "architecture": self.architecture,
                "model_path": self.model_path,
                "device": str(self.device)
            }
        )
    
    def batch_generate(self, texts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple inputs"""
        results = []
        for text in texts:
            generated = self.generate_text(text, **kwargs)
            results.append(generated)
        return results
    
    def create_gradio_interface(self):
        """Create Gradio interface for interactive chat"""
        
        def chat_interface(message, history, max_length, temperature):
            # Format conversation history
            context = ""
            for human, assistant in history:
                context += f"Human: {human}\nAssistant: {assistant}\n"
            
            # Add current message
            full_prompt = f"{context}Human: {message}\nAssistant:"
            
            # Generate response
            response = self.generate_text(
                full_prompt,
                max_length=max_length,
                temperature=temperature
            )
            
            return response
        
        # Create Gradio interface
        with gr.Blocks(title="Amazing LLM Chat") as interface:
            gr.Markdown("# Amazing LLM Chat Interface")
            gr.Markdown(f"**Model:** {self.architecture} from {self.model_path}")
            
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox(label="Message", placeholder="Type your message here...")
                    with gr.Row():
                        send_btn = gr.Button("Send")
                        clear_btn = gr.Button("Clear")
                
                with gr.Column():
                    max_length = gr.Slider(1, 500, value=100, label="Max Length")
                    temperature = gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
            
            # Event handlers
            msg.submit(chat_interface, [msg, chatbot, max_length, temperature], [chatbot, msg])
            send_btn.click(chat_interface, [msg, chatbot, max_length, temperature], [chatbot, msg])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        return interface

# FastAPI app
app = FastAPI(title="Amazing LLM API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine = None

@app.on_event("startup")
async def startup_event():
    global inference_engine
    # Initialize inference engine
    # This would be set by the main function
    pass

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Generate text from input prompt"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        response = await inference_engine.generate_async(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": inference_engine is not None}

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "architecture": inference_engine.architecture,
        "model_path": inference_engine.model_path,
        "device": str(inference_engine.device)
    }

def main():
    parser = argparse.ArgumentParser(description="LLM Inference and Deployment")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--architecture", choices=["transformer", "mamba"], default="transformer",
                       help="Model architecture")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--mode", choices=["single", "api", "chat"], default="single",
                       help="Inference mode")
    parser.add_argument("--text", help="Text to generate from (for single mode)")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for API server")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    global inference_engine
    inference_engine = LLMInference(
        model_path=args.model_path,
        architecture=args.architecture,
        device=args.device
    )
    
    if args.mode == "single":
        # Single text generation
        if not args.text:
            print("Please provide --text for single mode")
            return
        
        generated = inference_engine.generate_text(args.text)
        print(f"Input: {args.text}")
        print(f"Generated: {generated}")
    
    elif args.mode == "api":
        # Start API server
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.mode == "chat":
        # Start Gradio chat interface
        interface = inference_engine.create_gradio_interface()
        interface.launch(server_name=args.host, server_port=args.port)

if __name__ == "__main__":
    main()
