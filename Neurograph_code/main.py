"""
NeuroGraph Unified Entry Point
Flexible, config-agnostic training system with optional production features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import argparse
import json
from typing import Optional

from train.modular_train_context import create_modular_train_context


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def determine_config_file(args):
    """Smart config file selection with flexible defaults."""
    if args.config:
        if os.path.exists(args.config):
            return args.config
        else:
            raise FileNotFoundError(f"Specified config file not found: {args.config}")
    
    # Smart defaults based on available files and mode
    candidates = []
    
    if args.production:
        candidates.append('config/production.yaml')
    
    candidates.extend([
        'config/neurograph.yaml',
        'config/production.yaml',
        'config/default.yaml'
    ])
    
    for config_path in candidates:
        if os.path.exists(config_path):
            print(f"Auto-selected config: {config_path}")
            return config_path
    
    raise FileNotFoundError("No configuration file found. Please specify --config or ensure a default config exists.")


def plot_training_curves(losses, accuracies=None, save_path=None, config=None):
    """Plot and save training curves."""
    if save_path is None and config is not None:
        save_path = config.get('paths.training_curves_path', 'logs/training_curves.png')
    elif save_path is None:
        save_path = "logs/training_curves.png"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get visualization settings from config
    viz_config = config.get('visualization', {}) if config else {}
    figure_width = viz_config.get('figure_width', 10)
    figure_height = viz_config.get('figure_height', 6)
    line_width = viz_config.get('line_width', 2)
    grid_alpha = viz_config.get('grid_alpha', 0.3)
    font_size_title = viz_config.get('font_size_title', 14)
    font_size_label = viz_config.get('font_size_label', 12)
    dpi = viz_config.get('dpi', 300)
    bbox_inches = viz_config.get('bbox_inches', 'tight')
    
    if accuracies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figure_width + 5, figure_height))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figure_width, figure_height))
    
    # Loss curve
    ax1.plot(losses, 'b-', linewidth=line_width, label='Training Loss')
    ax1.set_title('NeuroGraph Training Loss', fontsize=font_size_title, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=font_size_label)
    ax1.set_ylabel('Loss', fontsize=font_size_label)
    ax1.grid(True, alpha=grid_alpha)
    ax1.legend()
    
    # Accuracy curve (if available)
    if accuracies:
        ax2.plot(accuracies, 'g-', linewidth=line_width, label='Validation Accuracy')
        ax2.set_title('NeuroGraph Validation Accuracy', fontsize=font_size_title, fontweight='bold')
        ax2.set_xlabel('Validation Check', fontsize=font_size_label)
        ax2.set_ylabel('Accuracy', fontsize=font_size_label)
        ax2.grid(True, alpha=grid_alpha)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def evaluate_model(trainer, num_samples=None, use_batch_evaluation=False):
    """Evaluate the trained model with optional batch optimization."""
    if num_samples is None:
        num_samples = trainer.config.get('debugging.evaluation_samples', 300)
    
    print(f"\nEvaluating model ({num_samples} samples)...")
    
    accuracy = trainer.evaluate_accuracy(
        num_samples=num_samples, 
        use_batch_evaluation=use_batch_evaluation
    )
    
    print(f"Accuracy: {accuracy:.1%} (evaluated on {num_samples} samples)")
    
    # Print diagnostic summary if available
    if hasattr(trainer, 'get_diagnostic_summary'):
        diagnostic_summary = trainer.get_diagnostic_summary()
        if diagnostic_summary:
            print(f"\n📊 Training Diagnostics Summary:")
            
            # Show gradient effectiveness
            discrete_update_analysis = diagnostic_summary.get('discrete_update_analysis', {})
            if discrete_update_analysis:
                effectiveness = discrete_update_analysis.get('mean_effectiveness', {})
                if effectiveness:
                    print(f"   🎯 Gradient Effectiveness: {effectiveness.get('mean', 0.0):.1%} ± {effectiveness.get('std', 0.0):.1%}")
            
            # Show parameter update stats
            backward_diagnostics = diagnostic_summary.get('backward_pass_diagnostics', {})
            if backward_diagnostics:
                param_stats = backward_diagnostics.get('parameter_updates', {})
                if param_stats:
                    phase_changes = param_stats.get('avg_phase_change', {})
                    mag_changes = param_stats.get('avg_mag_change', {})
                    if phase_changes and mag_changes:
                        print(f"   📈 Parameter Changes: Phase={phase_changes.get('mean', 0.0):.3f}, Mag={mag_changes.get('mean', 0.0):.3f}")
    
    return accuracy


def setup_production_monitoring(device='cuda'):
    """Setup production monitoring if available."""
    profiler = None
    performance_monitor = None
    
    try:
        from utils.gpu_profiler import create_cuda_profiler, create_performance_monitor
        profiler = create_cuda_profiler(device=device)
        performance_monitor = create_performance_monitor(profiler)
        performance_monitor.start_monitoring()
        print("🔍 Production monitoring enabled")
        return profiler, performance_monitor
    except ImportError:
        print("⚠️  Production monitoring not available (missing gpu_profiler)")
        return None, None


def run_benchmark_mode(config_path, evaluation_samples):
    """Run comprehensive benchmark mode."""
    print(f"\n⚡ Benchmark Mode")
    print("-" * 30)
    
    try:
        from test_batch_evaluation_performance import BatchEvaluationPerformanceTest
        
        benchmark_test = BatchEvaluationPerformanceTest(config_path)
        benchmark_results = benchmark_test.run_comprehensive_test()
        
        # Save benchmark results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"logs/benchmark_{timestamp}.json"
        os.makedirs("logs", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print(f"\n💾 Benchmark results saved to: {results_file}")
        return benchmark_results
        
    except ImportError:
        print("❌ Benchmark mode not available (missing test_batch_evaluation_performance)")
        return None


def run_evaluate_mode(trainer, evaluation_samples, production_mode):
    """Run evaluation-only mode."""
    print(f"\n📊 Evaluation Mode ({evaluation_samples} samples)")
    print("-" * 30)
    
    if production_mode:
        try:
            from core.batch_evaluation_engine import create_batch_evaluation_engine
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Create batch evaluator with config-based batch size
            batch_size = trainer.config.get('batch_evaluation.batch_size', 16)
            batch_evaluator = create_batch_evaluation_engine(
                trainer, batch_size=batch_size, device=device, verbose=True
            )
            
            # Run evaluation
            eval_results = batch_evaluator.evaluate_accuracy_batched(
                num_samples=evaluation_samples, streaming=True
            )
            
            print(f"\n✅ Production Evaluation Results:")
            print(f"   🎯 Accuracy: {eval_results['accuracy']:.1%}")
            print(f"   ⚡ Speed: {eval_results['samples_per_second']:.1f} samples/s")
            print(f"   🗄️  Cache Hit Rate: {eval_results['cache_hit_rate']:.1%}")
            
            return eval_results['accuracy']
            
        except ImportError:
            print("⚠️  Batch evaluation not available, using standard evaluation")
    
    # Standard evaluation
    return evaluate_model(trainer, num_samples=evaluation_samples)


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='NeuroGraph Unified Training System')
    
    # Configuration
    parser.add_argument('--config', type=str, 
                       help='Configuration file path (auto-detected if not specified)')
    
    # Modes
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'evaluate', 'benchmark'], 
                       default='train', help='Operation mode')
    parser.add_argument('--production', action='store_true',
                       help='Enable production features (GPU profiling, batch optimization)')
    
    # Training options
    parser.add_argument('--epochs', type=int, 
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test mode (reduced epochs)')
    # Evaluation options
    parser.add_argument('--eval-samples', type=int, 
                       help='Number of samples for evaluation (uses config default if not specified)')
    
    # Other options
    parser.add_argument('--seed', type=int, help='Random seed (uses config default if not specified)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Disable training curve plotting')
    
    args = parser.parse_args()
    
    # Display header
    if args.production:
        print("🚀 NeuroGraph Production System")
        print("=" * 50)
    else:
        print("NeuroGraph Training System")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU information
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available() and args.production:
        gpu_name = torch.cuda.get_device_name()
        print(f"✅ GPU: {gpu_name}")
        print(f"   💾 Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    elif torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  Running on CPU (GPU recommended for optimal performance)")
    
    # Determine configuration file first to get config-based defaults
    try:
        config_path = determine_config_file(args)
        print(f"Using configuration: {config_path}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    
    # Initialize training context to get config
    print("🔧 Initializing training context..." if args.production else "Initializing training context...")
    trainer = create_modular_train_context(config_path)
    
    # Set config-based defaults for arguments
    if args.seed is None:
        args.seed = trainer.config.get('architecture.seed', 42)
    if args.eval_samples is None:
        args.eval_samples = trainer.config.get('evaluation.default_samples', 100)
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Setup production monitoring if enabled
    profiler = None
    performance_monitor = None
    if args.production:
        profiler, performance_monitor = setup_production_monitoring(device)
    
    try:
        
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            print(f"Loaded checkpoint: {args.checkpoint}")
        
        # Handle different modes
        if args.mode == 'benchmark':
            return run_benchmark_mode(config_path, args.eval_samples)
        
        elif args.mode == 'evaluate':
            return run_evaluate_mode(trainer, args.eval_samples, args.production)
        
        else:  # train mode
            # Override epochs if specified
            if args.epochs is not None:
                trainer.num_epochs = args.epochs
                print(f"📊 Epochs: {args.epochs} (overridden)" if args.production else f"Epochs overridden: {args.epochs}")
            
            # Quick test mode
            if args.quick:
                quick_epochs = trainer.config.get('training.quick_mode.epochs', 5)
                print(f"⚡ Quick mode: {quick_epochs} epochs" if args.production else f"Quick test mode: {quick_epochs} epochs")
                trainer.num_epochs = quick_epochs
            
            # Training
            print(f"\n🎯 Starting Training ({trainer.num_epochs} epochs)" if args.production else f"Starting training ({trainer.num_epochs} epochs)...")
            if args.production:
                print("-" * 30)
            
            start_time = datetime.now()
            losses = trainer.train()
            training_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n✅ Training completed in {training_time:.1f}s" if args.production else f"Training completed in {training_time:.1f} seconds")
            if args.production:
                print(f"   📉 Final loss: {losses[-1]:.4f}")
            
            # Plot training curves (unless disabled)
            if not args.no_plot:
                try:
                    plot_training_curves(losses, getattr(trainer, 'validation_accuracies', None), config=trainer.config)
                except Exception as e:
                    print(f"Warning: Could not save training curves: {e}")
            
            # Final evaluation
            print(f"\n📊 Final Evaluation ({args.eval_samples} samples)" if args.production else "")
            eval_start = datetime.now()
            final_accuracy = evaluate_model(trainer, num_samples=args.eval_samples, use_batch_evaluation=args.production)
            eval_time = (datetime.now() - eval_start).total_seconds()
            
            if args.production:
                print(f"   🎯 Final Accuracy: {final_accuracy:.1%}")
                print(f"   ⚡ Evaluation Speed: {args.eval_samples/eval_time:.1f} samples/s")
            
            # Results summary (detailed for non-production mode)
            if not args.production:
                print("\nFinal Results:")
                config = trainer.config
                print(f"Architecture: {config.get('architecture.total_nodes')} nodes, "
                      f"{config.get('resolution.phase_bins')}x{config.get('resolution.mag_bins')} resolution")
                print(f"Parameters: {trainer.count_parameters():,}")
                print(f"Final accuracy: {final_accuracy:.1%}")
                print(f"Training epochs: {len(losses)}")
                print(f"Final loss: {losses[-1]:.4f}")
                
                if len(losses) > 1:
                    improvement = losses[0] - losses[-1]
                    print(f"Loss improvement: {improvement:+.4f}")
            
            # Save checkpoint
            checkpoint_dir = trainer.config.get('paths.checkpoint_path', 'checkpoints/')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"production_model_{timestamp}.pt" if args.production else f"model_{timestamp}.pt"
            checkpoint_path = f"{checkpoint_dir}{checkpoint_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainer.save_checkpoint(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}" if args.production else f"Checkpoint saved: {checkpoint_path}")
            
            print(f"\n🎯 NeuroGraph run completed!" if args.production else f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return final_accuracy
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user" if args.production else "\nTraining interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}" if args.production else f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Stop monitoring if enabled
        if performance_monitor:
            performance_monitor.stop_monitoring()
            performance_monitor.print_performance_report()
        
        # Print performance statistics
        if hasattr(trainer, 'print_performance_report'):
            trainer.print_performance_report()


if __name__ == "__main__":
    # Run main training
    result = main()
    
    # Exit with appropriate code
    if result is not None:
        print(f"Final result: {result:.1%}" if isinstance(result, float) else f"Completed successfully")
        exit(0)
    else:
        print("Execution failed")
        exit(1)
