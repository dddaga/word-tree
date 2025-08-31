#!/usr/bin/env python3
# main_1000.py
# Main execution file for 1000-node NeuroGraph network

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

from train.train_context_1000 import TrainContext1000 as TrainContext

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def plot_training_curve(losses, save_path="logs/large_1000/training_curve.png"):
    """Plot and save training loss curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.title('1000-Node NeuroGraph Training Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations
    if len(losses) > 1:
        improvement = losses[0] - losses[-1]
        plt.annotate(f'Improvement: {improvement:+.4f}', 
                    xy=(len(losses)-1, losses[-1]), 
                    xytext=(len(losses)*0.7, max(losses)*0.8),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training curve saved to: {save_path}")

def comprehensive_evaluation(trainer, num_samples=200):
    """Perform comprehensive evaluation of the trained model."""
    print("\n🔍 Comprehensive Model Evaluation")
    print("=" * 50)
    
    # Overall accuracy
    accuracy = trainer.evaluate_accuracy(num_samples=num_samples)
    
    # Per-class analysis
    print(f"\n📊 Per-class analysis (sample size: {min(num_samples//10, 20)} per class):")
    class_correct = [0] * 10
    class_total = [0] * 10
    
    dataset_size = len(trainer.input_adapter.mnist)
    samples_per_class = min(num_samples // 10, 20)
    
    for digit in range(10):
        # Find samples of this digit
        digit_indices = []
        for i in range(dataset_size):
            if len(digit_indices) >= samples_per_class:
                break
            _, label = trainer.input_adapter.get_input_context(i, trainer.input_nodes)
            if label == digit:
                digit_indices.append(i)
        
        # Evaluate samples of this digit
        for idx in digit_indices:
            try:
                pred_label, true_label = trainer.evaluate_sample(idx)
                if pred_label == true_label:
                    class_correct[digit] += 1
                class_total[digit] += 1
            except:
                continue
    
    # Print per-class results
    for digit in range(10):
        if class_total[digit] > 0:
            class_acc = class_correct[digit] / class_total[digit]
            print(f"   Digit {digit}: {class_correct[digit]}/{class_total[digit]} = {class_acc:.1%}")
        else:
            print(f"   Digit {digit}: No samples evaluated")
    
    return accuracy

def compare_with_baseline():
    """Compare results with previous baseline."""
    print("\n📈 Performance Comparison")
    print("=" * 50)
    print("Previous Results:")
    print("   🔸 Original (batch training): 10% accuracy")
    print("   🔸 Single-sample (50 nodes): 18% accuracy")
    print("   🔸 Specialized (50 nodes): 18% accuracy")
    print("\nCurrent Network:")
    print("   🔸 1000 nodes (200 input): Testing...")

def main():
    """Main execution function."""
    print("🚀 NeuroGraph 1000-Node Network Training")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set seeds for reproducibility
    set_seeds(42)
    print("🎲 Random seeds set for reproducibility")
    
    try:
        # Initialize training context
        print("\n🔧 Initializing 1000-Node Training Context...")
        trainer = TrainContext()
        
        # Display network information
        info = trainer.input_adapter.get_dataset_info()
        print(f"\n📋 Network Information:")
        print(f"   🏗️  Total nodes: {trainer.config['total_nodes']}")
        print(f"   📥 Input nodes: {info['input_nodes']}")
        print(f"   📤 Output nodes: {len(trainer.output_nodes)}")
        print(f"   🎯 PCA dimensions: {info['pca_dims']}")
        print(f"   📈 Capacity increase: {info['capacity_increase']:.1f}x vs baseline")
        
        # Show baseline comparison
        compare_with_baseline()
        
        # Training
        print(f"\n🎯 Starting Full Training ({trainer.config['num_epochs']} epochs)")
        print("=" * 60)
        
        start_time = datetime.now()
        losses = trainer.train()
        end_time = datetime.now()
        
        training_duration = (end_time - start_time).total_seconds()
        print(f"\n⏱️  Training completed in {training_duration:.1f} seconds")
        
        # Plot training curve
        plot_training_curve(losses)
        
        # Comprehensive evaluation
        final_accuracy = comprehensive_evaluation(trainer, num_samples=300)
        
        # Results summary
        print("\n🎉 FINAL RESULTS")
        print("=" * 60)
        print(f"📊 Network Architecture:")
        print(f"   • Total nodes: {trainer.config['total_nodes']}")
        print(f"   • Input nodes: {trainer.config['num_input_nodes']} (vs 5 baseline)")
        print(f"   • PCA dimensions: {info['pca_dims']} (vs 50 baseline)")
        print(f"   • Capacity increase: {info['capacity_increase']:.1f}x")
        print()
        print(f"🎯 Performance:")
        print(f"   • Final accuracy: {final_accuracy:.1%}")
        print(f"   • Training epochs: {len(losses)}")
        print(f"   • Final loss: {losses[-1]:.4f}")
        print(f"   • Loss improvement: {losses[0] - losses[-1]:+.4f}")
        print()
        print(f"📈 Comparison with baselines:")
        baseline_10 = 0.10
        baseline_18 = 0.18
        improvement_vs_10 = (final_accuracy - baseline_10) / baseline_10 * 100
        improvement_vs_18 = (final_accuracy - baseline_18) / baseline_18 * 100
        print(f"   • vs 10% baseline: {improvement_vs_10:+.1f}% relative improvement")
        print(f"   • vs 18% baseline: {improvement_vs_18:+.1f}% relative improvement")
        
        # Success criteria
        print(f"\n✅ Success Criteria:")
        success_threshold = 0.25  # 25% accuracy target
        if final_accuracy >= success_threshold:
            print(f"   🎉 SUCCESS: Achieved {final_accuracy:.1%} ≥ {success_threshold:.1%} target!")
            print(f"   🚀 Ready for production migration!")
        else:
            print(f"   📊 Progress: {final_accuracy:.1%} (target: {success_threshold:.1%})")
            print(f"   🔧 Consider further optimization or architecture changes")
        
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return final_accuracy, losses
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def quick_test():
    """Quick test with reduced parameters for development."""
    print("🧪 Quick Test Mode (5 epochs)")
    print("=" * 40)
    
    try:
        trainer = TrainContext()
        
        # Override config for quick test
        trainer.config['num_epochs'] = 5
        trainer.config['warmup_epochs'] = 2
        
        losses = trainer.train()
        accuracy = trainer.evaluate_accuracy(num_samples=50)
        
        print(f"\n📋 Quick Test Results:")
        print(f"   ✅ Training: {len(losses)} epochs completed")
        print(f"   📊 Final loss: {losses[-1]:.4f}")
        print(f"   🎯 Accuracy: {accuracy:.1%}")
        print(f"   🚀 System ready for full training!")
        
        return accuracy, losses
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuroGraph 1000-Node Network')
    parser.add_argument('--quick', action='store_true', help='Run quick test (5 epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set custom seed if provided
    if args.seed != 42:
        set_seeds(args.seed)
        print(f"🎲 Using custom seed: {args.seed}")
    
    # Run appropriate mode
    if args.quick:
        accuracy, losses = quick_test()
    else:
        accuracy, losses = main()
    
    # Exit with appropriate code
    if accuracy is not None:
        print(f"\n🎯 Final accuracy: {accuracy:.1%}")
        exit(0)
    else:
        print(f"\n❌ Training failed")
        exit(1)
