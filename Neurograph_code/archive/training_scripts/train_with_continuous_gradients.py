"""
Full training script using continuous gradient approximation.
This should achieve significantly better accuracy than the previous 10-18%.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from train.modular_train_context import create_modular_train_context

def train_neurograph_with_continuous_gradients():
    """Train NeuroGraph using the new continuous gradient approximation."""
    print("🚀 NeuroGraph Training with Continuous Gradient Approximation")
    print("=" * 70)
    
    # Initialize training context
    print("Initializing training context...")
    train_context = create_modular_train_context()
    
    print(f"\n📊 Training Configuration:")
    print(f"   🎯 Resolution: {train_context.config.get('resolution.phase_bins')}×{train_context.config.get('resolution.mag_bins')}")
    print(f"   📈 Learning rate: {train_context.effective_lr:.4f}")
    print(f"   🔄 Gradient accumulation: {train_context.config.get('training.gradient_accumulation.enabled')}")
    print(f"   📊 Epochs: {train_context.num_epochs}")
    print(f"   💾 Device: {train_context.device}")
    
    # Enable gradient debugging for first few samples
    train_context.debug_gradients = False  # Set to True for detailed gradient info
    
    # Pre-training evaluation
    print(f"\n📊 Pre-training evaluation...")
    initial_accuracy = train_context.evaluate_accuracy(num_samples=100)
    print(f"   Initial accuracy: {initial_accuracy:.1%}")
    
    # Training loop
    print(f"\n🎯 Starting Training")
    print("=" * 50)
    
    start_time = datetime.now()
    training_losses = []
    training_accuracies = []
    validation_accuracies = []
    
    best_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(train_context.num_epochs):
        epoch_start = datetime.now()
        
        # Training phase
        epoch_loss, epoch_accuracy = train_context.train_epoch()
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)
        
        # Validation phase (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            val_accuracy = train_context.evaluate_accuracy(num_samples=200)
            validation_accuracies.append(val_accuracy)
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch + 1
                
                # Save best model
                checkpoint_path = f"checkpoints/best_continuous_gradients_epoch_{epoch+1}.pt"
                train_context.save_checkpoint(checkpoint_path)
        else:
            validation_accuracies.append(validation_accuracies[-1] if validation_accuracies else initial_accuracy)
        
        # Progress reporting
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        elapsed = (datetime.now() - start_time).total_seconds()
        eta = elapsed * (train_context.num_epochs - epoch - 1) / (epoch + 1)
        
        print(f"Epoch {epoch+1:3d}/{train_context.num_epochs}: "
              f"Loss={epoch_loss:.4f}, "
              f"Train_Acc={epoch_accuracy:.1%}, "
              f"Val_Acc={validation_accuracies[-1]:.1%}, "
              f"Time={epoch_time:.1f}s, "
              f"ETA={eta/60:.1f}min")
        
        # Early stopping if accuracy is very good
        if validation_accuracies[-1] > 0.90:
            print(f"🎉 Excellent accuracy achieved! Stopping early.")
            break
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Final evaluation
    print(f"\n📊 Final Evaluation")
    print("=" * 30)
    
    final_accuracy = train_context.evaluate_accuracy(num_samples=500)
    print(f"Final accuracy (500 samples): {final_accuracy:.1%}")
    print(f"Best accuracy: {best_accuracy:.1%} (epoch {best_epoch})")
    print(f"Training time: {total_time:.1f} seconds")
    
    # Compare with previous results
    print(f"\n📈 Performance Comparison")
    print("=" * 30)
    print(f"Previous system accuracy: 10-18%")
    print(f"New system accuracy: {final_accuracy:.1%}")
    
    if final_accuracy > 0.18:
        improvement = (final_accuracy - 0.18) / 0.18 * 100
        print(f"🎉 Improvement: +{improvement:.1f}% relative to previous best!")
    else:
        print(f"⚠️  Accuracy still below previous best of 18%")
    
    # Plot training curves
    plot_training_curves(training_losses, training_accuracies, validation_accuracies)
    
    # Detailed analysis
    analyze_training_results(train_context, final_accuracy)
    
    return {
        'final_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'training_losses': training_losses,
        'training_accuracies': training_accuracies,
        'validation_accuracies': validation_accuracies,
        'training_time': total_time
    }

def plot_training_curves(losses, train_accs, val_accs):
    """Plot training curves."""
    print(f"\n📊 Plotting training curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy curves
    epochs = range(1, len(train_accs) + 1)
    ax2.plot(epochs, [acc * 100 for acc in train_accs], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc * 100 for acc in val_accs], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Progress')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('continuous_gradients_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Training curves saved to 'continuous_gradients_training_curves.png'")

def analyze_training_results(train_context, final_accuracy):
    """Analyze training results and gradient behavior."""
    print(f"\n🔍 Training Analysis")
    print("=" * 30)
    
    # Test gradient computation on a few samples
    print("Analyzing gradient behavior...")
    
    gradient_norms = []
    for i in range(5):
        # Get sample
        input_context, target_label = train_context.input_adapter.get_input_context(i, train_context.input_nodes)
        output_signals = train_context.forward_pass(input_context)
        
        if output_signals:
            loss, logits = train_context.compute_loss(output_signals, target_label)
            node_gradients = train_context.backward_pass(loss, output_signals)
            
            # Compute total gradient norm
            total_norm = 0.0
            for node_id, (phase_grad, mag_grad) in node_gradients.items():
                total_norm += torch.norm(phase_grad).item() + torch.norm(mag_grad).item()
            
            gradient_norms.append(total_norm)
    
    avg_grad_norm = np.mean(gradient_norms)
    print(f"   Average gradient norm: {avg_grad_norm:.4f}")
    
    # Check parameter diversity
    print("Analyzing parameter diversity...")
    phase_diversity = []
    mag_diversity = []
    
    for node_id in train_context.output_nodes:
        string_node_id = f"n{node_id}"
        if string_node_id in train_context.node_store.phase_table:
            phase_idx = train_context.node_store.phase_table[string_node_id]
            mag_idx = train_context.node_store.mag_table[string_node_id]
            
            phase_diversity.extend(phase_idx.cpu().numpy().flatten())
            mag_diversity.extend(mag_idx.cpu().numpy().flatten())
    
    phase_unique = len(np.unique(phase_diversity))
    mag_unique = len(np.unique(mag_diversity))
    
    print(f"   Phase parameter diversity: {phase_unique}/{train_context.config.get('resolution.phase_bins')} bins used")
    print(f"   Magnitude parameter diversity: {mag_unique}/{train_context.config.get('resolution.mag_bins')} bins used")
    
    # Success criteria
    print(f"\n✅ Success Criteria:")
    print(f"   🎯 Target accuracy: >40% (vs 10-18% previous)")
    print(f"   📊 Achieved accuracy: {final_accuracy:.1%}")
    
    if final_accuracy > 0.40:
        print(f"   🎉 SUCCESS: Target exceeded!")
    elif final_accuracy > 0.25:
        print(f"   ✅ GOOD: Significant improvement over previous system")
    elif final_accuracy > 0.18:
        print(f"   📈 PROGRESS: Better than previous best")
    else:
        print(f"   ⚠️  NEEDS WORK: Still below previous performance")

def test_specific_samples(train_context, num_samples=10):
    """Test the model on specific samples to understand behavior."""
    print(f"\n🧪 Testing on {num_samples} specific samples...")
    
    correct_predictions = 0
    
    for i in range(num_samples):
        input_context, target_label = train_context.input_adapter.get_input_context(i, train_context.input_nodes)
        output_signals = train_context.forward_pass(input_context)
        
        if output_signals:
            class_encodings = train_context.class_encoder.get_all_encodings()
            logits = train_context.loss_function.compute_logits_from_signals(
                output_signals, class_encodings, train_context.lookup_tables
            )
            
            predicted_class = torch.argmax(logits).item()
            confidence = torch.softmax(logits, dim=0)[predicted_class].item()
            
            is_correct = predicted_class == target_label
            if is_correct:
                correct_predictions += 1
            
            print(f"   Sample {i}: Target={target_label}, Predicted={predicted_class}, "
                  f"Confidence={confidence:.1%}, {'✅' if is_correct else '❌'}")
    
    accuracy = correct_predictions / num_samples
    print(f"   Sample accuracy: {accuracy:.1%}")

if __name__ == "__main__":
    print("🚀 NeuroGraph Continuous Gradient Training")
    print("=" * 60)
    
    # Create checkpoints directory
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        # Run training
        results = train_neurograph_with_continuous_gradients()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   🎯 Best accuracy: {results['best_accuracy']:.1%}")
        print(f"   📈 Final accuracy: {results['final_accuracy']:.1%}")
        print(f"   ⏱️  Training time: {results['training_time']:.1f} seconds")
        
        # Save results
        results_file = f"continuous_gradients_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write("NeuroGraph Continuous Gradient Training Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Final accuracy: {results['final_accuracy']:.1%}\n")
            f.write(f"Best accuracy: {results['best_accuracy']:.1%} (epoch {results['best_epoch']})\n")
            f.write(f"Training time: {results['training_time']:.1f} seconds\n")
            f.write(f"Training losses: {results['training_losses']}\n")
            f.write(f"Validation accuracies: {results['validation_accuracies']}\n")
        
        print(f"📄 Results saved to {results_file}")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
