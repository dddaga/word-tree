"""
NeuroGraph Main Entry Point
Production-ready modular system with gradient accumulation and high-resolution computation
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

from train.modular_train_context import ModularTrainContext, create_modular_train_context
from utils.modular_config import ModularConfig

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def plot_training_curves(losses, accuracies=None, save_path=None, config=None):
    """Plot and save training curves."""
    if save_path is None and config is not None:
        save_path = config.get('paths.training_curves_path', 'logs/modular/training_curves.png')
    elif save_path is None:
        save_path = "logs/modular/training_curves.png"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if accuracies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Loss curve
    ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('NeuroGraph Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy curve (if available)
    if accuracies:
        ax2.plot(accuracies, 'g-', linewidth=2, label='Validation Accuracy')
        ax2.set_title('NeuroGraph Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Validation Check', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")

def evaluate_model(trainer, num_samples=None):
    """Evaluate the trained model."""
    if num_samples is None:
        num_samples = trainer.config.get('debugging.evaluation_samples', 300)
    
    print("\nEvaluating model...")
    accuracy = trainer.evaluate_accuracy(num_samples=num_samples)
    print(f"Accuracy: {accuracy:.1%} (evaluated on {num_samples} samples)")
    return accuracy

def main():
    """Main execution function."""
    print("NeuroGraph Training System")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='NeuroGraph Training')
    parser.add_argument('--config', type=str, default='config/neurograph.yaml',
                       help='Configuration file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (5 epochs)')
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only mode')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    print(f"Random seed: {args.seed}")
    
    try:
        # Initialize training context
        print("Initializing training context...")
        trainer = create_modular_train_context(args.config)
        
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            print(f"Loaded checkpoint: {args.checkpoint}")
        
        # Quick test mode
        if args.quick:
            quick_epochs = trainer.config.get('training.quick_mode.epochs', 5)
            quick_warmup = trainer.config.get('training.quick_mode.warmup_epochs', 2)
            print(f"Quick test mode: {quick_epochs} epochs")
            trainer.num_epochs = quick_epochs
            trainer.warmup_epochs = quick_warmup
        
        # Evaluation only mode
        if args.eval_only:
            print("Evaluation only mode")
            final_eval_samples = trainer.config.get('debugging.final_evaluation_samples', 500)
            final_accuracy = evaluate_model(trainer, num_samples=final_eval_samples)
            print(f"Final accuracy: {final_accuracy:.1%}")
            return final_accuracy
        
        # Training
        print(f"Starting training ({trainer.num_epochs} epochs)...")
        
        start_time = datetime.now()
        losses = trainer.train()
        end_time = datetime.now()
        
        training_duration = (end_time - start_time).total_seconds()
        print(f"Training completed in {training_duration:.1f} seconds")
        
        # Plot training curves
        plot_training_curves(losses, trainer.validation_accuracies, config=trainer.config)
        
        # Final evaluation
        final_eval_samples = trainer.config.get('debugging.final_evaluation_samples', 500)
        final_accuracy = evaluate_model(trainer, num_samples=final_eval_samples)
        
        # Results summary
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
        checkpoint_dir = trainer.config.get('paths.checkpoint_path', 'checkpoints/modular/')
        checkpoint_path = f"{checkpoint_dir}final_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return final_accuracy
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run main training
    accuracy = main()
    
    # Exit with appropriate code
    if accuracy is not None:
        print(f"Final accuracy: {accuracy:.1%}")
        exit(0)
    else:
        print("Training failed")
        exit(1)
