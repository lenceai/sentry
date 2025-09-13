# Amazing LLM Usage Guide

This guide will help you get started with training your amazing LLM from scratch using your multi-GPU setup.

## Prerequisites

- **Python**: 3.11 (automatically installed via conda)
- **PyTorch**: 2.7.1 with CUDA 12.6 support (automatically installed)
- **TorchVision**: 0.22.1 (automatically installed)
- **CUDA**: 11.7+ (automatically detected, optimized for CUDA 12.6)
- **Hardware**: Multi-GPU setup with NVIDIA GPUs

## Quick Start

### 1. Initial Setup

```bash
# Run the complete setup
python setup.py --setup-all

# Activate conda environment
conda activate LLM
```

### 2. Download Training Data

```bash
# Download FineWeb dataset (base training)
python train.py --pipeline data --download-fineweb

# Download FineWeb-Edu dataset (educational training)
python train.py --pipeline data --download-fineweb-edu

# Process custom medical dataset
python train.py --pipeline data --custom-data ./research_papers --domain medical
```

### 3. Start Training

```bash
# Complete training pipeline (recommended)
python train.py --pipeline complete --architecture transformer
# Or use the convenience script: ./train.sh --pipeline complete --architecture transformer

# Or train Mamba architecture
python train.py --pipeline complete --architecture mamba
# Or: ./train.sh --pipeline complete --architecture mamba

# With custom medical data
python train.py --pipeline complete --architecture transformer --custom-data ./research_papers --domain medical
# Or: ./train.sh --pipeline complete --architecture transformer --custom-data ./research_papers --domain medical
```

## Detailed Usage

### Data Preparation

#### FineWeb Dataset
```bash
python data/download_fineweb.py \
    --dataset fineweb \
    --output_dir ./data/fineweb \
    --max_samples 1000000
```

#### FineWeb-Edu Dataset
```bash
python data/download_fineweb.py \
    --dataset fineweb-edu \
    --output_dir ./data/fineweb-edu \
    --max_samples 500000
```

#### Custom Domain Dataset
```bash
python data/custom_dataset.py \
    --input_dir ./research_papers \
    --domain medical \
    --output_dir ./data/medical
```

### Training

#### Base Training (FineWeb)
```bash
# Transformer
python training/train_transformer.py \
    --config configs/base_config.yaml \
    --dataset_path ./data/fineweb \
    --stage base \
    --output_dir ./models/transformer-base

# Mamba
python training/train_mamba.py \
    --config configs/mamba_config.yaml \
    --dataset_path ./data/fineweb \
    --stage base \
    --output_dir ./models/mamba-base
```

#### Educational Training (FineWeb-Edu)
```bash
# Transformer
python training/train_transformer.py \
    --config configs/base_config.yaml \
    --dataset_path ./data/fineweb-edu \
    --stage base \
    --output_dir ./models/transformer-edu

# Mamba
python training/train_mamba.py \
    --config configs/mamba_config.yaml \
    --dataset_path ./data/fineweb-edu \
    --stage base \
    --output_dir ./models/mamba-edu
```

#### Fine-tuning (Custom Domain)
```bash
# Transformer
python training/train_transformer.py \
    --config configs/base_config.yaml \
    --dataset_path ./data/medical \
    --stage finetune \
    --output_dir ./models/transformer-medical \
    --resume_from ./models/transformer-edu

# Mamba
python training/train_mamba.py \
    --config configs/mamba_config.yaml \
    --dataset_path ./data/medical \
    --stage finetune \
    --output_dir ./models/mamba-medical \
    --resume_from ./models/mamba-edu
```

### Distributed Training

#### Using the Launcher
```bash
# Launch distributed training
python training/launch_distributed.py \
    --config configs/base_config.yaml \
    --architecture transformer \
    --stage base \
    --dataset_path ./data/fineweb \
    --output_dir ./models/transformer-base \
    --method manual
```

#### Using SLURM (if available)
```bash
# Submit SLURM job
sbatch slurm_transformer_base.sh
```

#### Using Docker Compose
```bash
# Start distributed training with Docker
docker-compose -f docker-compose-transformer-base.yml up
```

### Evaluation

```bash
# Evaluate model
python evaluation/benchmark.py \
    --model_path ./models/transformer-medical \
    --architecture transformer \
    --benchmark all

# Specific benchmarks
python evaluation/benchmark.py \
    --model_path ./models/transformer-medical \
    --architecture transformer \
    --benchmark medical
```

### Deployment

#### API Server
```bash
# Start API server
python deployment/inference.py \
    --model_path ./models/transformer-medical \
    --architecture transformer \
    --mode api \
    --port 8000
```

#### Interactive Chat
```bash
# Start chat interface
python deployment/inference.py \
    --model_path ./models/transformer-medical \
    --architecture transformer \
    --mode chat \
    --port 7860
```

#### Single Inference
```bash
# Generate text
python deployment/inference.py \
    --model_path ./models/transformer-medical \
    --architecture transformer \
    --mode single \
    --text "What are the symptoms of rheumatoid arthritis?"
```

## GPU Monitoring

```bash
# Monitor GPU usage
python utils/gpu_monitor.py --monitor

# Check memory usage
python utils/gpu_monitor.py --check-memory

# Calculate optimal batch size
python utils/gpu_monitor.py --optimal-batch 1.0

# Clean up GPU memory
python utils/gpu_monitor.py --cleanup
```

## Configuration

### Model Configuration
Edit `configs/base_config.yaml` or `configs/mamba_config.yaml` to adjust:
- Model size (layers, hidden dimensions)
- Training parameters (learning rate, batch size)
- Sequence length
- Optimization settings

### GPU Configuration
Edit `configs/gpu_config.yaml` to configure:
- System specifications
- Memory allocation
- Network settings
- Load balancing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Distributed Training Issues**
   - Check network connectivity between systems
   - Verify NCCL installation
   - Check firewall settings

3. **Data Loading Issues**
   - Ensure sufficient disk space
   - Check data format compatibility
   - Verify tokenizer configuration

### Performance Optimization

1. **Memory Optimization**
   - Use gradient checkpointing
   - Enable mixed precision training
   - Optimize batch size per GPU

2. **Training Speed**
   - Use compiled models (PyTorch 2.0)
   - Optimize data loading
   - Use efficient attention mechanisms

3. **Model Quality**
   - Adjust learning rate schedule
   - Use appropriate warmup steps
   - Regular evaluation and checkpointing

## Hardware Specifications

Your setup includes:
- **System 1**: RTX 3080 TI (12GB VRAM)
- **System 2-4**: RTX 3090 (24GB VRAM each)
- **Total VRAM**: 84GB across 4 systems

Recommended configurations:
- **RTX 3080 TI**: Batch size 2, max sequence length 1024
- **RTX 3090**: Batch size 4, max sequence length 2048

## Next Steps

1. **Start with base training** on FineWeb data
2. **Evaluate model performance** on standard benchmarks
3. **Fine-tune on your domain** (e.g., medical research)
4. **Deploy and test** your model
5. **Iterate and improve** based on results

## Support

For issues and questions:
1. Check the logs in `./logs/` directory
2. Review GPU monitoring output
3. Check model evaluation results
4. Verify data processing pipeline

Happy training! 🚀
