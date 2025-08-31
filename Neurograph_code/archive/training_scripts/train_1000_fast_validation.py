"""
1000-Node NeuroGraph Fast Validation Training
Quick validation run with updated radiation.py and reduced epochs (10 instead of 60)
"""

import torch
import numpy as np
import time
import os
import psutil
from datetime import datetime
from train.modular_train_context import create_modular_train_context

def log_system_info():
    """Log system information for performance monitoring."""
    print(f"🖥️  System Information:")
    print(f"   🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   🎮 GPU: {torch.cuda.get_device_name()}")
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"   🎮 GPU memory: {gpu_props.total_memory / (1024**3):.1f} GB")
    
    print(f"   🖥️  CPU cores: {psutil.cpu_count()}")
    print(f"   💾 System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

def log_memory_usage(device='cuda'):
    """Log current memory usage."""
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"   💾 GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

def train_1000_node_fast_validation():
    """Fast validation training for 1000-node NeuroGraph with updated radiation.py."""
    print("🚀 1000-Node NeuroGraph Ultra-Fast Validation Training")
    print("🎯 Testing updated radiation.py with 5 epochs for quick architecture validation")
    print("=" * 80)
    
    # Log system information
    log_system_info()
    
    # Initialize training context with fast test config
    print(f"\n📊 Initializing 1000-node training context with fast config...")
    start_init = time.time()
    
    train_context = create_modular_train_context("config/neurograph.yaml")
    
    init_time = time.time() - start_init
    print(f"⏱️  Initialization time: {init_time:.2f} seconds")
    
    # Log memory usage after initialization
    log_memory_usage(train_context.device)
    
    # Pre-training evaluation
    print(f"\n📊 Pre-training evaluation...")
    eval_start = time.time()
    
    initial_accuracy = train_context.evaluate_accuracy(num_samples=100)
    
    eval_time = time.time() - eval_start
    print(f"   Initial accuracy: {initial_accuracy:.1%}")
    print(f"   ⏱️  Evaluation time: {eval_time:.2f} seconds")
    
    # Training configuration summary
    print(f"\n📊 Fast Validation Configuration:")
    print(f"   🎯 Total nodes: {train_context.config.get('architecture.total_nodes')}")
    print(f"   📥 Input nodes: {train_context.config.get('architecture.input_nodes')}")
    print(f"   📤 Output nodes: {train_context.config.get('architecture.output_nodes')}")
    print(f"   🔄 Epochs: {train_context.num_epochs} (reduced from 60)")
    print(f"   🔥 Warmup epochs: {train_context.config.get('training.optimizer.warmup_epochs')} (reduced from 25)")
    print(f"   📚 Samples per epoch: {train_context.config.get('training.optimizer.batch_size')}")
    print(f"   💾 Device: {train_context.device}")
    print(f"   🌐 Radiation neighbors: {train_context.config.get('graph_structure.top_k_neighbors')}")
    
    # Validation focus
    print(f"\n🎯 Validation Focus:")
    print(f"   ✅ Updated radiation.py integration")
    print(f"   ✅ 1000-node system stability")
    print(f"   ✅ Vectorized radiation performance")
    print(f"   ✅ Training pipeline functionality")
    print(f"   ✅ Memory efficiency")
    
    # Training loop with validation focus
    print(f"\n🎯 Starting Fast Validation Training")
    print("=" * 60)
    
    training_start = time.time()
    training_losses = []
    training_accuracies = []
    epoch_times = []
    
    best_accuracy = initial_accuracy
    best_epoch = 0
    
    # Performance tracking
    total_samples_processed = 0
    total_forward_time = 0
    total_backward_time = 0
    
    for epoch in range(train_context.num_epochs):
        epoch_start = time.time()
        
        # Enhanced progress display
        progress_percent = ((epoch) / train_context.num_epochs) * 100
        elapsed_time = time.time() - training_start
        
        print(f"\n🚀 1000-Node Validation Progress")
        print("=" * 50)
        print(f"📊 Overall: Epoch {epoch+1}/{train_context.num_epochs} ({progress_percent:.1f}% complete)")
        print(f"⏱️  Elapsed: {elapsed_time/60:.1f} minutes")
        if epoch > 0:
            avg_epoch_time = elapsed_time / epoch
            eta_minutes = (avg_epoch_time * (train_context.num_epochs - epoch)) / 60
            print(f"📈 ETA: ~{eta_minutes:.1f} minutes remaining")
            print(f"⚡ Avg epoch time: {avg_epoch_time:.1f} seconds")
        if best_accuracy > initial_accuracy:
            print(f"🌟 Best accuracy so far: {best_accuracy:.1%} (Epoch {best_epoch})")
        print("-" * 50)
        
        # Training phase
        epoch_losses = []
        epoch_accuracies = []
        
        # Get dataset size and sample indices
        dataset_size = train_context.input_adapter.get_dataset_info()['dataset_size']
        samples_per_epoch = train_context.config.get('training.optimizer.batch_size', 5)
        sample_indices = np.random.choice(dataset_size, samples_per_epoch, replace=False)
        
        for i, sample_idx in enumerate(sample_indices):
            sample_start = time.time()
            
            # Real-time sample progress
            sample_progress = ((i) / samples_per_epoch) * 100
            print(f"🎯 Processing Sample {i+1}/{samples_per_epoch} in Epoch {epoch+1}/{train_context.num_epochs} ({sample_progress:.0f}% of epoch)")
            
            # Get input context
            input_context, target_label = train_context.input_adapter.get_input_context(
                sample_idx, train_context.input_nodes
            )
            
            # Forward pass with timing
            forward_start = time.time()
            output_signals = train_context.forward_pass(input_context)
            forward_time = time.time() - forward_start
            total_forward_time += forward_time
            
            if output_signals:
                # Compute loss
                loss, logits = train_context.compute_loss(output_signals, target_label)
                accuracy = train_context.loss_function.compute_accuracy(
                    logits, torch.tensor(target_label, device=train_context.device)
                )
                
                # Backward pass with timing
                backward_start = time.time()
                node_gradients = train_context.backward_pass(loss, output_signals)
                backward_time = time.time() - backward_start
                total_backward_time += backward_time
                
                # Apply updates
                train_context.apply_direct_updates(node_gradients)
                
                # Track metrics
                sample_time = time.time() - sample_start
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy)
                total_samples_processed += 1
                
                # Enhanced sample logging with real-time metrics
                print(f"   ✅ Sample {i+1} Complete: Loss={loss.item():.4f}, Acc={accuracy:.1%}, Time={sample_time:.2f}s")
                print(f"      ⏱️  Timing: Forward={forward_time:.3f}s, Backward={backward_time:.3f}s")
                print(f"      📊 Outputs: {len(output_signals)} signals, Target: digit {target_label}")
                
                # Show running averages for this epoch
                if len(epoch_losses) > 1:
                    running_loss = np.mean(epoch_losses)
                    running_acc = np.mean(epoch_accuracies)
                    print(f"      📈 Running Avg: Loss={running_loss:.4f}, Acc={running_acc:.1%}")
                
            else:
                print(f"   ❌ Sample {i+1}/{samples_per_epoch}: No output signals generated")
            
            print()  # Add spacing between samples
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            training_losses.append(avg_loss)
            training_accuracies.append(avg_accuracy)
            
            # Performance metrics
            samples_per_second = samples_per_epoch / epoch_time
            
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   📈 Average Loss: {avg_loss:.4f}")
            print(f"   🎯 Average Accuracy: {avg_accuracy:.1%}")
            print(f"   ⏱️  Epoch Time: {epoch_time:.2f} seconds")
            print(f"   🚀 Performance: {samples_per_second:.2f} samples/sec")
            
            # Memory usage
            log_memory_usage(train_context.device)
            
            # Check for improvement
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_epoch = epoch + 1
                print(f"   🌟 New best accuracy: {best_accuracy:.1%}")
        
        # Validation every 3 epochs for fast test
        if (epoch + 1) % 3 == 0:
            val_start = time.time()
            val_accuracy = train_context.evaluate_accuracy(num_samples=100)
            val_time = time.time() - val_start
            
            print(f"   📊 Validation (100 samples): {val_accuracy:.1%} (time: {val_time:.2f}s)")
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch + 1
                print(f"   🌟 New validation best: {best_accuracy:.1%}")
    
    total_training_time = time.time() - training_start
    
    # Final evaluation
    print(f"\n📊 Final Validation Results")
    print("=" * 50)
    
    final_eval_start = time.time()
    final_accuracy = train_context.evaluate_accuracy(num_samples=200)
    final_eval_time = time.time() - final_eval_start
    
    print(f"Final accuracy (200 samples): {final_accuracy:.1%}")
    print(f"Best training accuracy: {best_accuracy:.1%} (epoch {best_epoch})")
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    print(f"Final evaluation time: {final_eval_time:.2f} seconds")
    
    # Validation success analysis
    print(f"\n✅ Validation Success Analysis")
    print("=" * 50)
    
    if epoch_times:
        avg_epoch_time = np.mean(epoch_times)
        overall_samples_per_sec = total_samples_processed / total_training_time
        
        print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        print(f"Overall samples per second: {overall_samples_per_sec:.2f}")
        print(f"Total samples processed: {total_samples_processed}")
        
        # Performance validation
        if avg_epoch_time < 120:  # Less than 2 minutes per epoch
            print(f"🚀 PERFORMANCE: ✅ Excellent - {avg_epoch_time:.1f}s per epoch")
        elif avg_epoch_time < 300:  # Less than 5 minutes per epoch
            print(f"⚡ PERFORMANCE: ✅ Good - {avg_epoch_time:.1f}s per epoch")
        else:
            print(f"⏱️  PERFORMANCE: ⚠️  Slow - {avg_epoch_time:.1f}s per epoch")
    
    # Integration validation
    print(f"\n🔧 Integration Validation:")
    integration_success = True
    
    if total_samples_processed > 0:
        print(f"✅ TRAINING PIPELINE: Working - {total_samples_processed} samples processed")
    else:
        print(f"❌ TRAINING PIPELINE: Failed - No samples processed")
        integration_success = False
    
    if training_losses and len(training_losses) > 0:
        print(f"✅ LOSS COMPUTATION: Working - Loss values generated")
    else:
        print(f"❌ LOSS COMPUTATION: Failed - No loss values")
        integration_success = False
    
    if final_accuracy > initial_accuracy:
        improvement = (final_accuracy - initial_accuracy) / initial_accuracy * 100
        print(f"✅ LEARNING: Working - {improvement:.1f}% improvement")
    elif final_accuracy >= initial_accuracy * 0.9:  # Within 10% of initial
        print(f"✅ LEARNING: Stable - Accuracy maintained")
    else:
        print(f"⚠️  LEARNING: Degraded - Accuracy decreased")
    
    if total_forward_time > 0 and total_backward_time > 0:
        print(f"✅ RADIATION INTEGRATION: Working - Forward/backward passes completed")
    else:
        print(f"❌ RADIATION INTEGRATION: Failed - Missing forward/backward timing")
        integration_success = False
    
    # Overall validation result
    print(f"\n🎯 Overall Validation Result:")
    if integration_success and final_accuracy >= initial_accuracy * 0.8:
        print(f"🎉 SUCCESS: Updated radiation.py integrates correctly!")
        print(f"✅ 1000-node system is stable and functional")
        print(f"✅ Ready for full training (60 epochs)")
        validation_status = "SUCCESS"
    elif integration_success:
        print(f"✅ PARTIAL SUCCESS: System works but accuracy needs attention")
        print(f"⚠️  Consider hyperparameter tuning for full training")
        validation_status = "PARTIAL"
    else:
        print(f"❌ VALIDATION FAILED: Integration issues detected")
        print(f"🔧 Requires debugging before full training")
        validation_status = "FAILED"
    
    return {
        'validation_status': validation_status,
        'final_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
        'initial_accuracy': initial_accuracy,
        'training_time_minutes': total_training_time / 60,
        'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
        'samples_processed': total_samples_processed,
        'integration_success': integration_success,
        'training_losses': training_losses,
        'training_accuracies': training_accuracies
    }

if __name__ == "__main__":
    print("🚀 1000-Node NeuroGraph Fast Validation with Updated Radiation")
    print("=" * 80)
    
    # Create necessary directories
    os.makedirs('logs/modular', exist_ok=True)
    os.makedirs('checkpoints/modular', exist_ok=True)
    os.makedirs('cache/encodings', exist_ok=True)
    
    try:
        # Run fast validation
        results = train_1000_node_fast_validation()
        
        print(f"\n🎉 Fast validation completed!")
        print(f"📊 Validation Results:")
        print(f"   🎯 Status: {results['validation_status']}")
        print(f"   📈 Final accuracy: {results['final_accuracy']:.1%}")
        print(f"   🌟 Best accuracy: {results['best_accuracy']:.1%}")
        print(f"   📊 Initial accuracy: {results['initial_accuracy']:.1%}")
        print(f"   ⏱️  Training time: {results['training_time_minutes']:.1f} minutes")
        print(f"   📚 Samples processed: {results['samples_processed']}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"fast_validation_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write("1000-Node NeuroGraph Fast Validation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Validation Status: {results['validation_status']}\n")
            f.write(f"Final accuracy: {results['final_accuracy']:.1%}\n")
            f.write(f"Best accuracy: {results['best_accuracy']:.1%}\n")
            f.write(f"Initial accuracy: {results['initial_accuracy']:.1%}\n")
            f.write(f"Training time: {results['training_time_minutes']:.1f} minutes\n")
            f.write(f"Average epoch time: {results['avg_epoch_time']:.2f} seconds\n")
            f.write(f"Samples processed: {results['samples_processed']}\n")
            f.write(f"Integration success: {results['integration_success']}\n")
            f.write(f"Training losses: {results['training_losses']}\n")
            f.write(f"Training accuracies: {results['training_accuracies']}\n")
        
        print(f"📄 Results saved to {results_file}")
        
        # Recommendation
        if results['validation_status'] == 'SUCCESS':
            print(f"\n🎯 RECOMMENDATION: Proceed with full 60-epoch training!")
            print(f"✅ Updated radiation.py is working correctly")
        elif results['validation_status'] == 'PARTIAL':
            print(f"\n⚠️  RECOMMENDATION: System works but consider tuning before full training")
        else:
            print(f"\n🔧 RECOMMENDATION: Debug integration issues before full training")
        
    except Exception as e:
        print(f"❌ Fast validation failed with error: {e}")
        import traceback
        traceback.print_exc()
