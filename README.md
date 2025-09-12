# Amazing LLM from Scratch

A comprehensive project for building state-of-the-art Large Language Models from scratch using FineWeb data, with support for both Transformer and Mamba architectures.

## Project Overview

This project implements a multi-stage training pipeline:
1. **Base Training**: FineWeb dataset for general language understanding
2. **Educational Training**: FineWeb-Edu for enhanced reasoning capabilities  
3. **Domain Fine-tuning**: Custom datasets for specific use cases (e.g., autoimmune disease research)

## Hardware Configuration

- **System 1**: RTX 3080 TI (12GB VRAM)
- **System 2-4**: RTX 3090 (24GB VRAM each)
- **Total VRAM**: 84GB across 4 systems
- **Distributed Training**: Multi-node, multi-GPU setup

## Supported Architectures

- **Transformers**: Standard attention-based models
- **Mamba**: State-space models for efficient long-sequence processing

## Project Structure

```
├── data/                    # Data processing and storage
├── models/                  # Model architectures and checkpoints
├── training/               # Training scripts and configurations
├── evaluation/             # Evaluation and benchmarking
├── deployment/             # Inference and deployment scripts
├── configs/                # Configuration files
└── utils/                  # Utility functions and helpers
```

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Download and process FineWeb data: `python data/download_fineweb.py`
3. Start training: `python training/train_transformer.py --config configs/base_config.yaml`

## Version

v1.0.0 - Initial release with multi-GPU support and dual architecture training