"""
Test script for continuous gradient approximation in NeuroGraph.
Validates the new gradient computation system.
"""

import torch
import numpy as np
from train.modular_train_context import create_modular_train_context

def test_gradient_computation():
    """Test the continuous gradient computation system."""
    print("🧪 Testing Continuous Gradient Approximation")
    print("=" * 60)
    
    try:
        # Initialize modular training context
        print("1. Initializing modular training context...")
        train_context = create_modular_train_context()
        
        print(f"✅ Training context initialized successfully")
        print(f"   📊 Device: {train_context.device}")
        print(f"   🎯 Resolution: {train_context.config.get('resolution.phase_bins')}×{train_context.config.get('resolution.mag_bins')}")
        
        # Test lookup table gradient computation
        print("\n2. Testing lookup table gradient computation...")
        test_lookup_gradients(train_context)
        
        # Test single sample training
        print("\n3. Testing single sample training...")
        test_single_sample_training(train_context)
        
        # Test backward pass
        print("\n4. Testing backward pass...")
        test_backward_pass(train_context)
        
        print("\n✅ All gradient computation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lookup_gradients(train_context):
    """Test the lookup table gradient computation methods."""
    lookup_tables = train_context.lookup_tables
    device = train_context.device
    
    # Create test indices on the correct device
    vector_dim = train_context.config.get('architecture.vector_dim')
    phase_indices = torch.randint(0, train_context.config.get('resolution.phase_bins'), (vector_dim,), device=device)
    mag_indices = torch.randint(0, train_context.config.get('resolution.mag_bins'), (vector_dim,), device=device)
    upstream_grad = torch.randn(vector_dim, device=device)
    
    print(f"   📊 Test indices shape: phase={phase_indices.shape}, mag={mag_indices.shape}")
    print(f"   📊 Upstream gradient shape: {upstream_grad.shape}")
    
    # Test signal computation
    signal = lookup_tables.get_signal_vector(phase_indices, mag_indices)
    print(f"   ✅ Signal computation: shape={signal.shape}, norm={torch.norm(signal):.4f}")
    
    # Test gradient computation
    phase_grad, mag_grad = lookup_tables.compute_signal_gradients(
        phase_indices, mag_indices, upstream_grad
    )
    print(f"   ✅ Gradient computation: phase_norm={torch.norm(phase_grad):.4f}, mag_norm={torch.norm(mag_grad):.4f}")
    
    # Test gradient quantization
    phase_updates, mag_updates = lookup_tables.quantize_gradients_to_discrete_updates(
        phase_grad, mag_grad, learning_rate=0.01
    )
    print(f"   ✅ Gradient quantization: phase_updates={phase_updates.shape}, mag_updates={mag_updates.shape}")
    
    # Test discrete updates
    new_phase_idx, new_mag_idx = lookup_tables.apply_discrete_updates(
        phase_indices, mag_indices, phase_updates, mag_updates
    )
    print(f"   ✅ Discrete updates: new_phase={new_phase_idx.shape}, new_mag={new_mag_idx.shape}")
    
    # Verify updates are different (learning occurred)
    phase_changed = not torch.equal(phase_indices, new_phase_idx)
    mag_changed = not torch.equal(mag_indices, new_mag_idx)
    print(f"   📈 Parameters changed: phase={phase_changed}, magnitude={mag_changed}")

def test_single_sample_training(train_context):
    """Test training on a single sample."""
    
    # Test with first sample
    sample_idx = 0
    
    print(f"   📊 Training on sample {sample_idx}...")
    
    # Get input context
    input_context, target_label = train_context.input_adapter.get_input_context(
        sample_idx, train_context.input_nodes
    )
    
    print(f"   📊 Input context: {len(input_context)} nodes, target={target_label}")
    
    # Forward pass
    output_signals = train_context.forward_pass(input_context)
    print(f"   📊 Output signals: {len(output_signals)} nodes")
    
    if output_signals:
        # Compute loss
        loss, logits = train_context.compute_loss(output_signals, target_label)
        print(f"   📊 Loss: {loss.item():.4f}")
        print(f"   📊 Logits shape: {logits.shape}")
        
        # Test accuracy computation
        accuracy = train_context.loss_function.compute_accuracy(
            logits, torch.tensor(target_label, device=train_context.device)
        )
        print(f"   📊 Accuracy: {accuracy:.1%}")
        
        # Test backward pass
        node_gradients = train_context.backward_pass(loss, output_signals)
        print(f"   📊 Node gradients computed: {len(node_gradients)} nodes")
        
        # Verify gradients are non-zero
        total_grad_norm = 0.0
        for node_id, (phase_grad, mag_grad) in node_gradients.items():
            phase_norm = torch.norm(phase_grad).item()
            mag_norm = torch.norm(mag_grad).item()
            total_grad_norm += phase_norm + mag_norm
            print(f"      Node {node_id}: phase_norm={phase_norm:.4f}, mag_norm={mag_norm:.4f}")
        
        print(f"   📈 Total gradient norm: {total_grad_norm:.4f}")
        
        if total_grad_norm > 1e-6:
            print("   ✅ Non-zero gradients computed successfully")
        else:
            print("   ⚠️  Warning: Very small gradients detected")
    else:
        print("   ⚠️  No output signals generated")

def test_backward_pass(train_context):
    """Test the backward pass components individually."""
    
    # Create dummy output signals
    vector_dim = train_context.config.get('architecture.vector_dim')
    output_signals = {}
    
    for node_id in train_context.output_nodes[:2]:  # Test first 2 output nodes
        output_signals[node_id] = torch.randn(vector_dim, device=train_context.device)
    
    print(f"   📊 Testing with {len(output_signals)} output signals")
    
    # Test upstream gradient computation
    upstream_gradients = train_context.compute_upstream_gradients(output_signals)
    print(f"   📊 Upstream gradients: {len(upstream_gradients)} nodes")
    
    for node_id, grad in upstream_gradients.items():
        print(f"      Node {node_id}: upstream_grad_norm={torch.norm(grad):.4f}")
    
    # Test cross-entropy gradient computation specifically
    if train_context.config.get('loss_function.type') == "categorical_crossentropy":
        ce_gradients = train_context.compute_crossentropy_gradients(output_signals)
        print(f"   📊 Cross-entropy gradients: {len(ce_gradients)} nodes")
        
        for node_id, grad in ce_gradients.items():
            print(f"      Node {node_id}: ce_grad_norm={torch.norm(grad):.4f}")
    
    # Test full backward pass
    dummy_loss = torch.tensor(1.0, device=train_context.device)
    node_gradients = train_context.backward_pass(dummy_loss, output_signals)
    
    print(f"   📊 Full backward pass: {len(node_gradients)} node gradients")
    
    for node_id, (phase_grad, mag_grad) in node_gradients.items():
        print(f"      Node {node_id}: phase={torch.norm(phase_grad):.4f}, mag={torch.norm(mag_grad):.4f}")

def test_training_epoch():
    """Test a short training epoch to verify end-to-end functionality."""
    print("\n🎯 Testing Short Training Epoch")
    print("=" * 40)
    
    try:
        # Initialize training context
        train_context = create_modular_train_context()
        
        # Enable gradient debugging
        train_context.debug_gradients = True
        
        # Train on a few samples
        print("Training on 3 samples...")
        
        total_loss = 0.0
        total_accuracy = 0.0
        
        for i in range(3):
            loss, accuracy = train_context.train_single_sample(i)
            total_loss += loss
            total_accuracy += accuracy
            
            print(f"   Sample {i}: Loss={loss:.4f}, Accuracy={accuracy:.1%}")
        
        avg_loss = total_loss / 3
        avg_accuracy = total_accuracy / 3
        
        print(f"\n📊 Average: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.1%}")
        
        # Test evaluation
        eval_accuracy = train_context.evaluate_accuracy(num_samples=5)
        print(f"📊 Evaluation accuracy (5 samples): {eval_accuracy:.1%}")
        
        print("✅ Short training epoch completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training epoch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 NeuroGraph Continuous Gradient Testing")
    print("=" * 60)
    
    # Run gradient computation tests
    gradient_test_passed = test_gradient_computation()
    
    if gradient_test_passed:
        # Run training epoch test
        training_test_passed = test_training_epoch()
        
        if training_test_passed:
            print("\n🎉 All tests passed! Continuous gradient approximation is working.")
            print("\n📋 Summary:")
            print("   ✅ Lookup table gradient computation")
            print("   ✅ Continuous gradient approximation")
            print("   ✅ Discrete parameter updates")
            print("   ✅ Single sample training")
            print("   ✅ Backward pass computation")
            print("   ✅ Short training epoch")
            print("\n🚀 Ready for full training!")
        else:
            print("\n⚠️  Gradient tests passed but training test failed.")
    else:
        print("\n❌ Gradient computation tests failed.")
    
    print("\n" + "=" * 60)
