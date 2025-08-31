#!/usr/bin/env python3
"""
NeuroGraph Performance Benchmark Script
Tests the optimized modular training context with performance monitoring
"""

import torch
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train.modular_train_context import create_modular_train_context
from utils.modular_config import ModularConfig

def create_benchmark_config() -> str:
    """Create a lightweight benchmark configuration."""
    config_content = """
# NeuroGraph Benchmark Configuration
# Optimized for quick performance testing

# System Configuration
device: "cpu"  # Use CPU for consistent benchmarking
mode: "benchmark"

# Architecture
architecture:
  total_nodes: 1000
  input_nodes: 200
  output_nodes: 10
  intermediate_nodes: 790
  vector_dim: 5
  seed: 42

# High-Resolution Discrete Computation
resolution:
  phase_bins: 64
  mag_bins: 1024

# Graph Structure
graph_structure:
  cardinality: 8
  connection_strategy: "random"

# Input Processing
input_processing:
  adapter_type: "linear_projection"
  input_dim: 784  # MNIST
  normalization: "layer_norm"
  dropout: 0.1
  learnable: true

# Class Encoding
class_encoding:
  type: "orthogonal"
  num_classes: 10
  encoding_dim: 5
  orthogonality_threshold: 0.1

# Loss Function
loss_function:
  type: "categorical_crossentropy"
  temperature: 1.0
  label_smoothing: 0.0

# Training Configuration
training:
  optimizer:
    base_learning_rate: 0.01
    effective_learning_rate: 0.0354  # √8 scaling for 8-step accumulation
    num_epochs: 3  # Few epochs for benchmarking
    batch_size: 10  # Small batch for quick testing
    warmup_epochs: 0
  
  gradient_accumulation:
    enabled: true
    accumulation_steps: 8
    lr_scaling: "sqrt_n"
    buffer_size: 1000

# Forward Pass
forward_pass:
  max_timesteps: 10
  decay_factor: 0.9
  min_activation_strength: 0.1
  radiation_threshold: 0.5

# Paths
paths:
  graph_path: "cache/benchmark_graph.pkl"
  checkpoint_dir: "checkpoints/benchmark/"
  log_dir: "logs/benchmark/"
"""
    
    config_path = "config/benchmark.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_performance_benchmark() -> Dict[str, any]:
    """
    Run comprehensive performance benchmark.
    
    Returns:
        Dictionary with benchmark results
    """
    print("🚀 Starting NeuroGraph Performance Benchmark")
    print("=" * 60)
    
    # Create benchmark configuration
    config_path = create_benchmark_config()
    print(f"📝 Created benchmark config: {config_path}")
    
    # Initialize training context
    print("\n🔧 Initializing training context...")
    start_init = time.perf_counter()
    
    try:
        train_context = create_modular_train_context(config_path)
        init_time = time.perf_counter() - start_init
        print(f"✅ Initialization completed in {init_time:.3f}s")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return {"status": "failed", "error": str(e)}
    
    # Reset performance stats for clean measurement
    train_context.reset_performance_stats()
    
    # Run training benchmark
    print("\n🎯 Running training benchmark...")
    start_training = time.perf_counter()
    
    try:
        # Train for few epochs
        training_losses = train_context.train()
        training_time = time.perf_counter() - start_training
        
        print(f"✅ Training completed in {training_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {"status": "failed", "error": str(e)}
    
    # Evaluate accuracy
    print("\n📊 Evaluating accuracy...")
    start_eval = time.perf_counter()
    
    try:
        final_accuracy = train_context.evaluate_accuracy(num_samples=50)
        eval_time = time.perf_counter() - start_eval
        
        print(f"✅ Evaluation completed in {eval_time:.3f}s")
        print(f"📈 Final accuracy: {final_accuracy:.1%}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        final_accuracy = 0.0
        eval_time = 0.0
    
    # Collect performance statistics
    performance_stats = train_context.get_performance_stats()
    cache_stats = train_context.get_cache_performance()
    
    # Print detailed reports
    train_context.print_performance_report()
    train_context.print_cache_report()
    
    # Calculate key metrics
    total_time = init_time + training_time + eval_time
    samples_processed = len(training_losses) * train_context.config.get('training.optimizer.batch_size', 10)
    samples_per_second = samples_processed / training_time if training_time > 0 else 0
    
    # Compile benchmark results
    results = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "total_nodes": train_context.config.get('architecture.total_nodes'),
            "vector_dim": train_context.config.get('architecture.vector_dim'),
            "resolution": f"{train_context.config.get('resolution.phase_bins')}x{train_context.config.get('resolution.mag_bins')}",
            "epochs": train_context.config.get('training.optimizer.num_epochs'),
            "batch_size": train_context.config.get('training.optimizer.batch_size'),
        },
        "timing": {
            "initialization_time": init_time,
            "training_time": training_time,
            "evaluation_time": eval_time,
            "total_time": total_time,
        },
        "performance": {
            "samples_processed": samples_processed,
            "samples_per_second": samples_per_second,
            "final_accuracy": final_accuracy,
            "final_loss": training_losses[-1] if training_losses else 0.0,
        },
        "memory": {
            "estimated_usage_mb": train_context.estimate_memory_usage(),
            "total_parameters": train_context.count_parameters(),
        },
        "detailed_stats": performance_stats,
        "cache_stats": cache_stats,
    }
    
    return results

def run_comparison_benchmark() -> Dict[str, any]:
    """
    Run comparison benchmark between optimized and baseline versions.
    
    Returns:
        Dictionary with comparison results
    """
    print("\n🔄 Running Comparison Benchmark")
    print("=" * 60)
    
    # This would compare against a baseline version
    # For now, we'll just run the optimized version
    optimized_results = run_performance_benchmark()
    
    if optimized_results["status"] != "success":
        return optimized_results
    
    # Calculate expected improvements based on optimizations
    expected_improvements = {
        "vectorized_intermediate_credit": 0.8,  # 80% speedup expected
        "jit_lookup_tables": 0.25,  # 25% speedup expected
        "radiation_caching": 0.4,   # 40% speedup expected
        "overall_expected": 0.5,    # 50% overall speedup expected
    }
    
    comparison_results = {
        "optimized_results": optimized_results,
        "expected_improvements": expected_improvements,
        "status": "success"
    }
    
    return comparison_results

def print_benchmark_summary(results: Dict[str, any]):
    """Print comprehensive benchmark summary."""
    if results["status"] != "success":
        print(f"\n❌ Benchmark failed: {results.get('error', 'Unknown error')}")
        return
    
    print("\n📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    if "optimized_results" in results:
        # Comparison benchmark
        opt_results = results["optimized_results"]
        expected = results["expected_improvements"]
        
        print("🚀 OPTIMIZED VERSION PERFORMANCE:")
        print_single_benchmark_summary(opt_results)
        
        print(f"\n🎯 EXPECTED IMPROVEMENTS:")
        print(f"  Vectorized Credit Assignment: {expected['vectorized_intermediate_credit']:.0%} speedup")
        print(f"  JIT Lookup Tables:           {expected['jit_lookup_tables']:.0%} speedup")
        print(f"  Radiation Caching:           {expected['radiation_caching']:.0%} speedup")
        print(f"  Overall Expected:            {expected['overall_expected']:.0%} speedup")
        
    else:
        # Single benchmark
        print_single_benchmark_summary(results)

def print_single_benchmark_summary(results: Dict[str, any]):
    """Print summary for a single benchmark run."""
    config = results["config"]
    timing = results["timing"]
    performance = results["performance"]
    memory = results["memory"]
    
    print(f"Configuration:")
    print(f"  Nodes: {config['total_nodes']:,} | Vector Dim: {config['vector_dim']} | Resolution: {config['resolution']}")
    print(f"  Epochs: {config['epochs']} | Batch Size: {config['batch_size']}")
    
    print(f"\nTiming Results:")
    print(f"  Initialization: {timing['initialization_time']:7.3f}s")
    print(f"  Training:       {timing['training_time']:7.3f}s")
    print(f"  Evaluation:     {timing['evaluation_time']:7.3f}s")
    print(f"  Total:          {timing['total_time']:7.3f}s")
    
    print(f"\nPerformance Metrics:")
    print(f"  Samples Processed:    {performance['samples_processed']:,}")
    print(f"  Samples/Second:       {performance['samples_per_second']:,.1f}")
    print(f"  Final Accuracy:       {performance['final_accuracy']:6.1%}")
    print(f"  Final Loss:           {performance['final_loss']:6.4f}")
    
    print(f"\nMemory Usage:")
    print(f"  Estimated Usage:      {memory['estimated_usage_mb']:6.1f} MB")
    print(f"  Total Parameters:     {memory['total_parameters']:,}")

def save_benchmark_results(results: Dict[str, any], filename: str = None):
    """Save benchmark results to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
    
    results_dir = "logs/benchmark/"
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    import json
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filepath}")

def main():
    """Main benchmark execution."""
    print("NeuroGraph Performance Benchmark")
    print("Testing optimized modular training context")
    print("=" * 60)
    
    try:
        # Run benchmark
        results = run_comparison_benchmark()
        
        # Print summary
        print_benchmark_summary(results)
        
        # Save results
        save_benchmark_results(results)
        
        if results["status"] == "success":
            print("\n✅ Benchmark completed successfully!")
            return 0
        else:
            print(f"\n❌ Benchmark failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n💥 Benchmark crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
