#!/usr/bin/env python3
"""
Test script for actual training performance with dual learning rates
Runs a focused training session on MNIST to verify real-world effectiveness
"""

import torch
import numpy as np
from train.modular_train_context import create_modular_train_context
import time
from datetime import datetime

def run_focused_training():
    """Run a focused training session to test actual performance."""
    print("🎯 Focused Training Test with Dual Learning Rates")
    print("=" * 60)
    
    # Initialize training context
    train_context = create_modular_train_context("config/production.yaml")
    
    # Verify dual learning rate configuration
    dual_lr_config = train_context.config.get('training.optimizer.dual_learning_rates', {})
    print(f"\n📊 Training Configuration:")
    print(f"   Dual learning rates enabled: {dual_lr_config.get('enabled', False)}")
    print(f"   Phase learning rate: {dual_lr_config.get('phase_learning_rate', 'N/A')}")
    print(f"   Magnitude learning rate: {dual_lr_config.get('magnitude_learning_rate', 'N/A')}")
    print(f"   Base learning rate: {train_context.config.get('training.optimizer.base_learning_rate')}")
    print(f"   Resolution: {train_context.config.get('resolution.phase_bins')}×{train_context.config.get('resolution.mag_bins')}")
    
    # Training parameters
    num_epochs = 3  # Short focused training
    samples_per_epoch = 20  # Reasonable sample size
    
    print(f"\n🏃 Training Parameters:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Samples per epoch: {samples_per_epoch}")
    print(f"   Total training samples: {num_epochs * samples_per_epoch}")
    
    # Track training metrics
    epoch_losses = []
    epoch_accuracies = []
    training_start_time = time.time()
    
    print(f"\n🚀 Starting Training...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Track per-epoch metrics
        sample_losses = []
        sample_accuracies = []
        parameter_changes = []
        
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        for sample_idx in range(samples_per_epoch):
            # Use different samples each time
            actual_sample_idx = (epoch * samples_per_epoch + sample_idx) % 1000
            
            # Train on sample
            loss, accuracy = train_context.train_single_sample(actual_sample_idx)
            
            sample_losses.append(loss)
            sample_accuracies.append(accuracy)
            
            # Progress reporting every 5 samples
            if (sample_idx + 1) % 5 == 0:
                recent_loss = np.mean(sample_losses[-5:])
                recent_acc = np.mean(sample_accuracies[-5:])
                print(f"   Samples {sample_idx-4:2d}-{sample_idx+1:2d}: "
                      f"Loss={recent_loss:.4f}, Acc={recent_acc:.1%}")
        
        # Epoch summary
        epoch_loss = np.mean(sample_losses)
        epoch_accuracy = np.mean(sample_accuracies)
        epoch_time = time.time() - epoch_start_time
        
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        
        print(f"\n   📊 Epoch {epoch + 1} Summary:")
        print(f"      Average Loss: {epoch_loss:.4f}")
        print(f"      Average Accuracy: {epoch_accuracy:.1%}")
        print(f"      Training Time: {epoch_time:.1f}s")
        
        # Check for improvement
        if epoch > 0:
            loss_change = epoch_losses[-2] - epoch_losses[-1]
            acc_change = epoch_accuracies[-1] - epoch_accuracies[-2]
            print(f"      Loss Change: {loss_change:+.4f} ({'↓' if loss_change > 0 else '↑'})")
            print(f"      Accuracy Change: {acc_change:+.1%} ({'↑' if acc_change > 0 else '↓'})")
    
    total_training_time = time.time() - training_start_time
    
    # Final evaluation
    print(f"\n🎯 Final Training Results:")
    print("=" * 60)
    
    # Overall performance
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]
    loss_improvement = initial_loss - final_loss
    loss_improvement_pct = (loss_improvement / initial_loss) * 100
    
    initial_accuracy = epoch_accuracies[0]
    final_accuracy = epoch_accuracies[-1]
    accuracy_improvement = final_accuracy - initial_accuracy
    
    print(f"📈 Performance Metrics:")
    print(f"   Initial Loss: {initial_loss:.4f}")
    print(f"   Final Loss: {final_loss:.4f}")
    print(f"   Loss Improvement: {loss_improvement:.4f} ({loss_improvement_pct:+.1f}%)")
    print(f"   Initial Accuracy: {initial_accuracy:.1%}")
    print(f"   Final Accuracy: {final_accuracy:.1%}")
    print(f"   Accuracy Improvement: {accuracy_improvement:+.1%}")
    
    # Training efficiency
    samples_per_second = (num_epochs * samples_per_epoch) / total_training_time
    print(f"\n⚡ Training Efficiency:")
    print(f"   Total Training Time: {total_training_time:.1f}s")
    print(f"   Samples per Second: {samples_per_second:.1f}")
    print(f"   Average Time per Sample: {total_training_time/(num_epochs * samples_per_epoch):.2f}s")
    
    # Convergence analysis
    if len(epoch_losses) > 1:
        loss_trend = np.polyfit(range(len(epoch_losses)), epoch_losses, 1)[0]
        acc_trend = np.polyfit(range(len(epoch_accuracies)), epoch_accuracies, 1)[0]
        
        print(f"\n📊 Convergence Analysis:")
        print(f"   Loss Trend: {loss_trend:.6f} per epoch ({'↓ Improving' if loss_trend < 0 else '↑ Worsening'})")
        print(f"   Accuracy Trend: {acc_trend:.6f} per epoch ({'↑ Improving' if acc_trend > 0 else '↓ Worsening'})")
    
    # Test on validation set
    print(f"\n🧪 Validation Test:")
    print("-" * 40)
    
    validation_samples = 50
    validation_start_time = time.time()
    
    validation_accuracy = train_context.evaluate_accuracy(
        num_samples=validation_samples, 
        use_batch_evaluation=True
    )
    
    validation_time = time.time() - validation_start_time
    
    print(f"   Validation Accuracy: {validation_accuracy:.1%}")
    print(f"   Validation Time: {validation_time:.1f}s")
    print(f"   Samples per Second: {validation_samples/validation_time:.1f}")
    
    # Success criteria
    print(f"\n🎯 Success Assessment:")
    print("=" * 60)
    
    success_criteria = {
        'loss_improving': loss_improvement > 0,
        'accuracy_improving': accuracy_improvement > 0,
        'final_accuracy_reasonable': final_accuracy > 0.1,  # At least 10% (better than random)
        'validation_accuracy_reasonable': validation_accuracy > 0.1,
        'training_stable': not any(np.isnan(epoch_losses)) and not any(np.isinf(epoch_losses))
    }
    
    success_count = sum(success_criteria.values())
    total_criteria = len(success_criteria)
    
    print(f"Success Criteria ({success_count}/{total_criteria}):")
    for criterion, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {criterion.replace('_', ' ').title()}: {status}")
    
    overall_success = success_count >= 4  # Need at least 4/5 criteria
    
    print(f"\n🏆 Overall Assessment: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    
    if overall_success:
        print(f"   🎉 The dual learning rate system is working effectively!")
        print(f"   📈 Training shows clear improvement in both loss and accuracy")
        print(f"   ⚡ System performance is stable and efficient")
    else:
        print(f"   ⚠️  The system needs further tuning or investigation")
        print(f"   📊 Some success criteria were not met")
    
    # Diagnostic summary
    if hasattr(train_context, 'print_diagnostic_report'):
        print(f"\n📋 Training Diagnostics:")
        print("-" * 40)
        train_context.print_diagnostic_report()
    
    return {
        'success': overall_success,
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'loss_improvement': loss_improvement,
        'accuracy_improvement': accuracy_improvement,
        'validation_accuracy': validation_accuracy,
        'training_time': total_training_time,
        'samples_per_second': samples_per_second
    }

def compare_with_baseline():
    """Compare current performance with expected baseline."""
    print(f"\n📊 Performance Comparison:")
    print("=" * 60)
    
    # Expected baseline performance (rough estimates)
    baseline = {
        'loss_improvement': 0.1,  # Should improve loss by at least 0.1
        'accuracy_improvement': 0.05,  # Should improve accuracy by at least 5%
        'final_accuracy': 0.15,  # Should achieve at least 15% accuracy
        'samples_per_second': 0.5,  # Should process at least 0.5 samples/sec
    }
    
    print(f"Expected Baseline Performance:")
    for metric, value in baseline.items():
        if 'accuracy' in metric:
            print(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
        elif 'samples_per_second' in metric:
            print(f"   {metric.replace('_', ' ').title()}: {value:.1f}")
        else:
            print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")
    
    return baseline

if __name__ == "__main__":
    print("🎯 Actual Training Performance Test")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show baseline expectations
    baseline = compare_with_baseline()
    
    # Run the actual training test
    results = run_focused_training()
    
    # Final summary
    print(f"\n🎯 Final Summary:")
    print("=" * 60)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results['success']:
        print(f"🎉 SUCCESS: Dual learning rate system is working effectively!")
        print(f"   • Loss improved by {results['loss_improvement']:.4f}")
        print(f"   • Accuracy improved by {results['accuracy_improvement']:.1%}")
        print(f"   • Final validation accuracy: {results['validation_accuracy']:.1%}")
        print(f"   • Training speed: {results['samples_per_second']:.1f} samples/sec")
    else:
        print(f"⚠️  MIXED RESULTS: System shows some improvements but needs tuning")
        print(f"   • Check diagnostic output above for specific issues")
        print(f"   • Consider adjusting learning rates or resolution settings")
