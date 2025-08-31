#!/usr/bin/env python3
"""
Test script to verify actual learning effectiveness
Tracks parameter changes over multiple training steps to see if nodes are actually learning
"""

import torch
import numpy as np
from train.modular_train_context import create_modular_train_context
import copy

def test_actual_learning():
    """Test if nodes are actually learning by tracking parameter changes over time."""
    print("🧠 Testing Actual Learning Effectiveness")
    print("=" * 60)
    
    # Initialize training context
    train_context = create_modular_train_context("config/production.yaml")
    
    # Get initial parameters for a subset of nodes
    test_nodes = ['n200', 'n201', 'n202', 'n203', 'n204']  # Output nodes
    
    print(f"\n📊 Tracking parameters for nodes: {test_nodes}")
    
    # Store initial parameters
    initial_params = {}
    for node_id in test_nodes:
        if node_id in train_context.node_store.phase_table:
            initial_params[node_id] = {
                'phase': train_context.node_store.phase_table[node_id].clone(),
                'magnitude': train_context.node_store.mag_table[node_id].clone()
            }
    
    print(f"\n🎯 Initial Parameters:")
    for node_id, params in initial_params.items():
        phase_vals = params['phase'][:3]  # Show first 3 dimensions
        mag_vals = params['magnitude'][:3]
        print(f"   {node_id}: Phase={phase_vals.tolist()}, Mag={mag_vals.tolist()}")
    
    # Train for multiple steps and track changes
    num_training_steps = 10
    parameter_history = {node_id: {'phase_changes': [], 'mag_changes': []} for node_id in test_nodes}
    loss_history = []
    accuracy_history = []
    
    print(f"\n🏃 Training for {num_training_steps} steps...")
    
    for step in range(num_training_steps):
        # Train on a sample
        sample_idx = step % 5  # Cycle through first 5 samples
        loss, accuracy = train_context.train_single_sample(sample_idx)
        
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        
        # Track parameter changes
        total_phase_change = 0
        total_mag_change = 0
        
        for node_id in test_nodes:
            if node_id in train_context.node_store.phase_table:
                current_phase = train_context.node_store.phase_table[node_id]
                current_mag = train_context.node_store.mag_table[node_id]
                
                # Calculate changes from initial
                phase_change = torch.sum(torch.abs(current_phase - initial_params[node_id]['phase'])).item()
                mag_change = torch.sum(torch.abs(current_mag - initial_params[node_id]['magnitude'])).item()
                
                parameter_history[node_id]['phase_changes'].append(phase_change)
                parameter_history[node_id]['mag_changes'].append(mag_change)
                
                total_phase_change += phase_change
                total_mag_change += mag_change
        
        print(f"   Step {step+1:2d}: Loss={loss:.4f}, Acc={accuracy:.1%}, "
              f"Total Δ: Phase={total_phase_change:.1f}, Mag={total_mag_change:.1f}")
    
    # Analyze learning effectiveness
    print(f"\n📈 Learning Analysis:")
    
    # Check if parameters are changing
    nodes_learning = 0
    for node_id in test_nodes:
        final_phase_change = parameter_history[node_id]['phase_changes'][-1]
        final_mag_change = parameter_history[node_id]['mag_changes'][-1]
        
        is_learning = final_phase_change > 0 or final_mag_change > 0
        if is_learning:
            nodes_learning += 1
        
        print(f"   {node_id}: Phase Δ={final_phase_change:.1f}, Mag Δ={final_mag_change:.1f} "
              f"{'✓ Learning' if is_learning else '✗ Stagnant'}")
    
    learning_percentage = (nodes_learning / len(test_nodes)) * 100
    print(f"\n📊 Learning Summary:")
    print(f"   Nodes learning: {nodes_learning}/{len(test_nodes)} ({learning_percentage:.1f}%)")
    
    # Check loss trend
    if len(loss_history) > 1:
        loss_trend = np.polyfit(range(len(loss_history)), loss_history, 1)[0]
        loss_improvement = loss_history[0] - loss_history[-1]
        print(f"   Loss trend: {loss_trend:.6f} per step")
        print(f"   Loss improvement: {loss_improvement:.4f} ({loss_improvement/loss_history[0]*100:.1f}%)")
    
    # Check accuracy trend
    if len(accuracy_history) > 1:
        accuracy_trend = np.polyfit(range(len(accuracy_history)), accuracy_history, 1)[0]
        accuracy_improvement = accuracy_history[-1] - accuracy_history[0]
        print(f"   Accuracy trend: {accuracy_trend:.6f} per step")
        print(f"   Accuracy improvement: {accuracy_improvement:.4f} ({accuracy_improvement*100:.1f}%)")
    
    # Test gradient effectiveness calculation
    print(f"\n🔍 Testing Gradient Effectiveness Calculation:")
    
    # Get a sample and compute gradients
    input_context, target_label = train_context.input_adapter.get_input_context(0, train_context.input_nodes)
    output_signals = train_context.forward_pass(input_context)
    
    if output_signals:
        loss, logits = train_context.compute_loss(output_signals, target_label)
        node_gradients = train_context.backward_pass(loss, output_signals)
        
        print(f"   Nodes with gradients: {len(node_gradients)}")
        
        # Calculate our own effectiveness metric
        total_discrete_changes = 0
        total_gradient_magnitude = 0
        
        for node_id, (phase_grad, mag_grad) in node_gradients.items():
            # Get dual learning rates
            dual_lr_config = train_context.config.get('training.optimizer.dual_learning_rates', {})
            phase_lr = dual_lr_config.get('phase_learning_rate', 0.01)
            magnitude_lr = dual_lr_config.get('magnitude_learning_rate', 0.01)
            
            # Calculate expected discrete changes
            phase_step_size = 2 * np.pi / train_context.config.get('resolution.phase_bins')
            mag_step_size = 6.0 / train_context.config.get('resolution.mag_bins')
            
            phase_continuous_update = phase_lr * torch.norm(phase_grad).item()
            mag_continuous_update = magnitude_lr * torch.norm(mag_grad).item()
            
            expected_phase_changes = phase_continuous_update / phase_step_size
            expected_mag_changes = mag_continuous_update / mag_step_size
            
            # Test actual quantization
            phase_updates, mag_updates = train_context.lookup_tables.quantize_gradients_to_discrete_updates(
                phase_grad, mag_grad, phase_lr, magnitude_lr, node_id=f"n{node_id}"
            )
            
            actual_phase_changes = torch.sum(torch.abs(phase_updates)).item()
            actual_mag_changes = torch.sum(torch.abs(mag_updates)).item()
            
            total_discrete_changes += actual_phase_changes + actual_mag_changes
            total_gradient_magnitude += expected_phase_changes + expected_mag_changes
            
            print(f"   Node {node_id}:")
            print(f"      Expected: Phase={expected_phase_changes:.2f}, Mag={expected_mag_changes:.2f}")
            print(f"      Actual:   Phase={actual_phase_changes:.0f}, Mag={actual_mag_changes:.0f}")
            
            # Calculate effectiveness for this node
            if expected_phase_changes + expected_mag_changes > 0:
                node_effectiveness = (actual_phase_changes + actual_mag_changes) / (expected_phase_changes + expected_mag_changes)
                print(f"      Effectiveness: {node_effectiveness:.1%}")
        
        # Overall effectiveness
        if total_gradient_magnitude > 0:
            overall_effectiveness = total_discrete_changes / total_gradient_magnitude
            print(f"\n   📊 Overall Gradient Effectiveness: {overall_effectiveness:.1%}")
        else:
            print(f"\n   📊 Overall Gradient Effectiveness: 0.0% (no gradients)")
    
    # Final assessment
    print(f"\n🎯 Final Assessment:")
    
    if nodes_learning > 0:
        print(f"   ✅ SUCCESS: {nodes_learning} nodes are learning!")
        print(f"   📈 Parameters are changing over training steps")
        
        if loss_improvement > 0:
            print(f"   📉 Loss is improving: {loss_improvement:.4f}")
        
        if accuracy_improvement > 0:
            print(f"   🎯 Accuracy is improving: {accuracy_improvement*100:.1f}%")
        
        return True
    else:
        print(f"   ❌ FAILURE: No nodes are learning")
        print(f"   📊 Parameters are not changing")
        return False

def test_parameter_update_mechanics():
    """Test the parameter update mechanics in detail."""
    print(f"\n🔧 Testing Parameter Update Mechanics:")
    print("=" * 60)
    
    # Initialize training context
    train_context = create_modular_train_context("config/production.yaml")
    
    # Get a test node
    test_node = 'n200'
    
    if test_node not in train_context.node_store.phase_table:
        print(f"❌ Test node {test_node} not found")
        return False
    
    # Get initial parameters
    initial_phase = train_context.node_store.phase_table[test_node].clone()
    initial_mag = train_context.node_store.mag_table[test_node].clone()
    
    print(f"Initial parameters for {test_node}:")
    print(f"   Phase: {initial_phase[:5].tolist()}")
    print(f"   Magnitude: {initial_mag[:5].tolist()}")
    
    # Create synthetic gradients
    vector_dim = train_context.config.get('architecture.vector_dim')
    synthetic_phase_grad = torch.randn(vector_dim, device=train_context.device) * 0.5
    synthetic_mag_grad = torch.randn(vector_dim, device=train_context.device) * 0.5
    
    print(f"\nSynthetic gradients:")
    print(f"   Phase grad norm: {torch.norm(synthetic_phase_grad).item():.4f}")
    print(f"   Mag grad norm: {torch.norm(synthetic_mag_grad).item():.4f}")
    
    # Test quantization with dual learning rates
    dual_lr_config = train_context.config.get('training.optimizer.dual_learning_rates', {})
    phase_lr = dual_lr_config.get('phase_learning_rate', 0.01)
    magnitude_lr = dual_lr_config.get('magnitude_learning_rate', 0.01)
    
    print(f"\nDual learning rates:")
    print(f"   Phase LR: {phase_lr}")
    print(f"   Magnitude LR: {magnitude_lr}")
    
    # Apply quantization
    phase_updates, mag_updates = train_context.lookup_tables.quantize_gradients_to_discrete_updates(
        synthetic_phase_grad, synthetic_mag_grad, phase_lr, magnitude_lr, node_id=test_node
    )
    
    print(f"\nQuantized updates:")
    print(f"   Phase updates: {torch.sum(torch.abs(phase_updates)).item():.0f} changes")
    print(f"   Mag updates: {torch.sum(torch.abs(mag_updates)).item():.0f} changes")
    print(f"   Phase update values: {phase_updates[:5].tolist()}")
    print(f"   Mag update values: {mag_updates[:5].tolist()}")
    
    # Apply updates manually
    new_phase_idx, new_mag_idx = train_context.lookup_tables.apply_discrete_updates(
        initial_phase, initial_mag, phase_updates, mag_updates
    )
    
    # Update node store
    train_context.node_store.phase_table[test_node].data = new_phase_idx.detach()
    train_context.node_store.mag_table[test_node].data = new_mag_idx.detach()
    
    # Check changes
    final_phase = train_context.node_store.phase_table[test_node]
    final_mag = train_context.node_store.mag_table[test_node]
    
    phase_change = torch.sum(torch.abs(final_phase - initial_phase)).item()
    mag_change = torch.sum(torch.abs(final_mag - initial_mag)).item()
    
    print(f"\nActual parameter changes:")
    print(f"   Phase change: {phase_change:.1f}")
    print(f"   Magnitude change: {mag_change:.1f}")
    print(f"   Total change: {phase_change + mag_change:.1f}")
    
    print(f"\nFinal parameters:")
    print(f"   Phase: {final_phase[:5].tolist()}")
    print(f"   Magnitude: {final_mag[:5].tolist()}")
    
    # Success if any parameters changed
    success = phase_change > 0 or mag_change > 0
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILURE'}: Parameters {'did' if success else 'did not'} change")
    
    return success

if __name__ == "__main__":
    print("🧠 Learning Effectiveness Test Suite")
    print("=" * 60)
    
    # Test parameter update mechanics first
    mechanics_success = test_parameter_update_mechanics()
    
    # Test actual learning over time
    learning_success = test_actual_learning()
    
    print(f"\n🎯 Final Results:")
    print(f"   Parameter update mechanics: {'✅ WORKING' if mechanics_success else '❌ BROKEN'}")
    print(f"   Actual learning over time: {'✅ WORKING' if learning_success else '❌ BROKEN'}")
    
    if mechanics_success and learning_success:
        print(f"\n🎉 SUCCESS: The dual learning rate system is working and nodes are learning!")
    else:
        print(f"\n❌ ISSUES DETECTED: The system needs further investigation.")
