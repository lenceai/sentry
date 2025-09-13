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

1. Setup environment: `python setup.py --setup-all`
2. Activate conda environment: `conda activate LLM`
3. Download and process data: `python train.py --pipeline data --download-fineweb`
4. Start training: `python train.py --pipeline complete --architecture transformer`

## Requirements

- **Python**: 3.11
- **PyTorch**: 2.7.1 with CUDA 12.6 support
- **TorchVision**: 0.22.1
- **CUDA**: 11.7+ (automatically detected, optimized for CUDA 12.6)
- **Hardware**: Multi-GPU setup (RTX 3080 TI + 3x RTX 3090)

## Version

v1.0.0 - Initial release with multi-GPU support and dual architecture training