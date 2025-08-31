"""
NeuroGraph Production Training Script
Optimized for 5-10x evaluation speedup with batch processing
"""

import torch
import argparse
import os
from datetime import datetime
from typing import Optional

from train.modular_train_context import create_modular_train_context
from core.batch_evaluation_engine import create_batch_evaluation_engine
from utils.gpu_profiler import create_cuda_profiler, create_performance_monitor


def main(
    config_path: str = "config/production.yaml",
    mode: str = "train",
    num_epochs: Optional[int] = None,
    evaluation_samples: int = 1000,
    quick_mode: bool = False,
    benchmark: bool = False
):
    """
    Main production training and evaluation script.
    
    Args:
        config_path: Path to production configuration
        mode: Operation mode ('train', 'evaluate', 'benchmark')
        num_epochs: Override number of training epochs
        evaluation_samples: Number of samples for evaluation
        quick_mode: Use quick training mode for testing
        benchmark: Run performance benchmarks
    """
    print("🚀 NeuroGraph Production System")
    print("=" * 50)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        print(f"✅ GPU: {gpu_name}")
        print(f"   💾 Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("⚠️  Running on CPU (GPU recommended for optimal performance)")
    
    # Initialize training context
    print(f"\n🔧 Initializing training context...")
    training_context = create_modular_train_context(config_path)
    
    # Initialize profiling if benchmarking
    profiler = None
    performance_monitor = None
    if benchmark:
        profiler = create_cuda_profiler(device=device)
        performance_monitor = create_performance_monitor(profiler)
        performance_monitor.start_monitoring()
        print("🔍 Performance monitoring enabled")
    
    try:
        if mode == "train":
            # Training mode
            print(f"\n🎯 Starting Training")
            print("-" * 30)
            
            # Override epochs if specified
            if num_epochs is not None:
                training_context.num_epochs = num_epochs
                print(f"   📊 Epochs: {num_epochs} (overridden)")
            
            # Use quick mode if requested
            if quick_mode:
                training_context.num_epochs = training_context.config.get('training.quick_mode.epochs', 3)
                print(f"   ⚡ Quick mode: {training_context.num_epochs} epochs")
            
            # Train the model
            start_time = datetime.now()
            training_losses = training_context.train()
            training_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n✅ Training completed in {training_time:.1f}s")
            print(f"   📉 Final loss: {training_losses[-1]:.4f}")
            
            # Final evaluation with batch optimization
            print(f"\n📊 Final Evaluation ({evaluation_samples} samples)")
            eval_start = datetime.now()
            final_accuracy = training_context.evaluate_accuracy(
                num_samples=evaluation_samples, use_batch_evaluation=True
            )
            eval_time = (datetime.now() - eval_start).total_seconds()
            
            print(f"   🎯 Final Accuracy: {final_accuracy:.1%}")
            print(f"   ⚡ Evaluation Speed: {evaluation_samples/eval_time:.1f} samples/s")
            
            # Save results
            results = {
                'training_time': training_time,
                'final_accuracy': final_accuracy,
                'evaluation_time': eval_time,
                'evaluation_speed': evaluation_samples / eval_time,
                'training_losses': training_losses
            }
            
            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"checkpoints/production_model_{timestamp}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            training_context.save_checkpoint(checkpoint_path)
            
        elif mode == "evaluate":
            # Evaluation only mode
            print(f"\n📊 Evaluation Mode ({evaluation_samples} samples)")
            print("-" * 30)
            
            # Create batch evaluator
            batch_evaluator = create_batch_evaluation_engine(
                training_context, batch_size=16, device=device, verbose=True
            )
            
            # Run evaluation
            eval_results = batch_evaluator.evaluate_accuracy_batched(
                num_samples=evaluation_samples, streaming=True
            )
            
            print(f"\n✅ Evaluation Results:")
            print(f"   🎯 Accuracy: {eval_results['accuracy']:.1%}")
            print(f"   ⚡ Speed: {eval_results['samples_per_second']:.1f} samples/s")
            print(f"   🗄️  Cache Hit Rate: {eval_results['cache_hit_rate']:.1%}")
            
        elif mode == "benchmark":
            # Benchmark mode
            print(f"\n⚡ Benchmark Mode")
            print("-" * 30)
            
            # Import and run comprehensive benchmark
            from test_batch_evaluation_performance import BatchEvaluationPerformanceTest
            
            benchmark_test = BatchEvaluationPerformanceTest(config_path)
            benchmark_results = benchmark_test.run_comprehensive_test()
            
            # Save benchmark results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"logs/production_benchmark_{timestamp}.json"
            os.makedirs("logs", exist_ok=True)
            
            import json
            with open(results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            
            print(f"\n💾 Benchmark results saved to: {results_file}")
            
        else:
            print(f"❌ Unknown mode: {mode}")
            return
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop monitoring if enabled
        if performance_monitor:
            performance_monitor.stop_monitoring()
            performance_monitor.print_performance_report()
        
        # Print performance statistics
        if hasattr(training_context, 'print_performance_report'):
            training_context.print_performance_report()
    
    print(f"\n🎯 NeuroGraph production run completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroGraph Production Training")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/production.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "evaluate", "benchmark"],
        default="train",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--eval-samples", 
        type=int, 
        default=1000,
        help="Number of samples for evaluation"
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Use quick training mode for testing"
    )
    
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Enable performance benchmarking"
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        mode=args.mode,
        num_epochs=args.epochs,
        evaluation_samples=args.eval_samples,
        quick_mode=args.quick,
        benchmark=args.benchmark
    )
