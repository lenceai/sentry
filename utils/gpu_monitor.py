#!/usr/bin/env python3
"""
GPU Monitoring and Resource Management
Version: v1.0.0

This script monitors GPU usage across all systems and provides
resource management for distributed training.

Usage:
    python utils/gpu_monitor.py --monitor
    python utils/gpu_monitor.py --check-memory
"""

import argparse
import subprocess
import json
import logging
from typing import Dict, List
import psutil
import GPUtil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitor GPU resources across distributed systems"""
    
    def __init__(self):
        self.systems = [
            {"name": "system_1", "gpu": "RTX_3080_TI", "vram_gb": 12},
            {"name": "system_2", "gpu": "RTX_3090", "vram_gb": 24},
            {"name": "system_3", "gpu": "RTX_3090", "vram_gb": 24},
            {"name": "system_4", "gpu": "RTX_3090", "vram_gb": 24}
        ]
    
    def get_gpu_info(self) -> Dict:
        """Get current GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = {}
            
            for i, gpu in enumerate(gpus):
                gpu_info[f"gpu_{i}"] = {
                    "name": gpu.name,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "memory_free": gpu.memoryFree,
                    "utilization": gpu.load * 100,
                    "temperature": gpu.temperature
                }
            
            return gpu_info
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {}
    
    def check_memory_usage(self) -> Dict:
        """Check memory usage and provide recommendations"""
        gpu_info = self.get_gpu_info()
        recommendations = []
        
        for gpu_id, info in gpu_info.items():
            memory_usage_pct = (info["memory_used"] / info["memory_total"]) * 100
            
            if memory_usage_pct > 90:
                recommendations.append(f"WARNING: {gpu_id} memory usage is {memory_usage_pct:.1f}%")
            elif memory_usage_pct > 80:
                recommendations.append(f"CAUTION: {gpu_id} memory usage is {memory_usage_pct:.1f}%")
            
            if info["temperature"] > 80:
                recommendations.append(f"WARNING: {gpu_id} temperature is {info['temperature']}°C")
        
        return {
            "gpu_info": gpu_info,
            "recommendations": recommendations
        }
    
    def get_optimal_batch_size(self, model_size_gb: float) -> Dict[str, int]:
        """Calculate optimal batch sizes for each GPU"""
        gpu_info = self.get_gpu_info()
        batch_sizes = {}
        
        for gpu_id, info in gpu_info.items():
            available_memory = info["memory_free"] / 1024  # Convert to GB
            # Reserve 2GB for system overhead
            usable_memory = available_memory - 2
            
            # Estimate batch size based on model size
            # Rough estimate: 1GB per batch item for 1B parameter model
            estimated_batch_size = int(usable_memory / model_size_gb)
            
            # Apply safety factor
            safe_batch_size = max(1, estimated_batch_size // 2)
            
            batch_sizes[gpu_id] = safe_batch_size
        
        return batch_sizes
    
    def monitor_training(self, log_file: str = "gpu_monitor.log"):
        """Continuously monitor GPU usage during training"""
        logger.info("Starting GPU monitoring...")
        
        with open(log_file, "w") as f:
            f.write("timestamp,gpu_id,memory_used,memory_total,utilization,temperature\n")
            
            while True:
                try:
                    gpu_info = self.get_gpu_info()
                    timestamp = psutil.time.time()
                    
                    for gpu_id, info in gpu_info.items():
                        f.write(f"{timestamp},{gpu_id},{info['memory_used']},{info['memory_total']},"
                               f"{info['utilization']},{info['temperature']}\n")
                        f.flush()
                    
                    # Log to console every 30 seconds
                    if int(timestamp) % 30 == 0:
                        logger.info(f"GPU Status: {gpu_info}")
                    
                except KeyboardInterrupt:
                    logger.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        try:
            # Clear CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear GPU memory: {e}")

def main():
    parser = argparse.ArgumentParser(description="GPU Monitoring and Management")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--check-memory", action="store_true", help="Check memory usage")
    parser.add_argument("--optimal-batch", type=float, help="Calculate optimal batch size for model size (GB)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up GPU memory")
    parser.add_argument("--log-file", default="gpu_monitor.log", help="Log file for monitoring")
    
    args = parser.parse_args()
    
    monitor = GPUMonitor()
    
    if args.monitor:
        monitor.monitor_training(args.log_file)
    elif args.check_memory:
        result = monitor.check_memory_usage()
        print(json.dumps(result, indent=2))
    elif args.optimal_batch:
        batch_sizes = monitor.get_optimal_batch_size(args.optimal_batch)
        print(json.dumps(batch_sizes, indent=2))
    elif args.cleanup:
        monitor.cleanup_gpu_memory()
    else:
        # Default: show current GPU status
        gpu_info = monitor.get_gpu_info()
        print(json.dumps(gpu_info, indent=2))

if __name__ == "__main__":
    main()
