#!/usr/bin/env python3
"""
Test script to compare batch training vs single-sample training
and validate the fix for the 10% accuracy issue.
"""

import torch
import random
import os
import matplotlib.pyplot as plt
from train.train_context import enhanced_train_context
from train.single_sample_train_context import single_sample_train_context
from utils.config import load_config

from modules.input_adapters import MNISTPCAAdapter
from modules.class_encoding import generate_digit_class_encodings
from modules.output_adapters import predict_label_from_output
from core.tables import ExtendedLookupTableModule
from core.graph import load_or_build_graph
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.forward_engine import run_enhanced_forward


def evaluate_model(config_path="config/default.yaml", num_samples=100):
    """
    Evaluate the model using the same logic as main.py
    """
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reload graph and components
    graph_keys = [
        "total_nodes", "num_input_nodes", "num_output_nodes",
        "vector_dim", "phase_bins", "mag_bins", "cardinality", "seed"
    ]
    graph_config = {k: cfg[k] for k in graph_keys}
    graph_df = load_or_build_graph(**graph_config, overwrite=False, 
                                   save_path=cfg.get("graph_path", "config/static_graph.pkl"))
    node_store = NodeStore(graph_df, cfg["vector_dim"], cfg["phase_bins"], cfg["mag_bins"])
    lookup = ExtendedLookupTableModule(cfg["phase_bins"], cfg["mag_bins"], device=device)
    phase_cell = PhaseCell(cfg["vector_dim"], lookup)

    input_nodes = list(node_store.input_nodes)
    output_nodes = node_store.output_nodes

    adapter = MNISTPCAAdapter(
        vector_dim=cfg["vector_dim"],
        num_input_nodes=len(input_nodes),
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        device=device
    )

    class_phase_encodings, class_mag_encodings = generate_digit_class_encodings(
        num_classes=10,
        vector_dim=cfg["vector_dim"],
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        seed=cfg.get("seed", 42)
    )
    
    # Convert to the format expected by predict_label_from_output
    class_encodings = {}
    for digit in range(10):
        class_encodings[digit] = (class_phase_encodings[digit], class_mag_encodings[digit])

    correct = 0
    total = num_samples
    predictions = []
    true_labels = []
    
    for i in range(total):
        idx = random.randint(0, len(adapter.mnist) - 1)
        input_context, label = adapter.get_input_context(idx, input_nodes)

        activation = run_enhanced_forward(
            graph_df=graph_df,
            node_store=node_store,
            phase_cell=phase_cell,
            lookup_table=lookup,
            input_context=input_context,
            vector_dim=cfg["vector_dim"],
            phase_bins=cfg["phase_bins"],
            mag_bins=cfg["mag_bins"],
            decay_factor=cfg["decay_factor"],
            min_strength=cfg["min_activation_strength"],
            max_timesteps=cfg["max_timesteps"],
            use_radiation=cfg["use_radiation"],
            top_k_neighbors=cfg["top_k_neighbors"],
            min_output_activation_timesteps=cfg.get("min_output_activation_timesteps", 3),
            device=device,
            verbose=False
        )

        pred = predict_label_from_output(activation, output_nodes, class_encodings, lookup)
        predictions.append(pred)
        true_labels.append(label)
        
        if pred == label:
            correct += 1
            
        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i + 1}/{total} samples...")

    accuracy = correct / total
    return accuracy, predictions, true_labels


def run_comparison_test():
    """
    Run comparison between batch training and single-sample training
    """
    print("="*80)
    print("🧪 TRAINING-EVALUATION MISMATCH FIX VALIDATION")
    print("="*80)
    
    # Create a smaller config for faster testing
    test_config_path = "config/test_fix.yaml"
    
    # Create test configuration
    test_config = """# Test configuration for training fix validation
total_nodes: 50
num_input_nodes: 5
num_output_nodes: 10
vector_dim: 5
phase_bins: 8
mag_bins: 256
cardinality: 4
seed: 42

# Forward pass behavior
decay_factor: 0.925
min_activation_strength: 0.01
max_timesteps: 15
min_output_activation_timesteps: 3
top_k_neighbors: 4
use_radiation: true

# Learning - Faster for testing
learning_rate: 0.01  # Increased learning rate
warmup_epochs: 10
num_epochs: 30  # Fewer epochs for testing
batch_size: 3

# Activation balancing - ENABLED
enable_activation_balancing: true
balancing_strategy: "round_robin"
max_activations_per_epoch: 10
min_activations_per_epoch: 2
force_activation_probability: 0.4

# Multi-output loss - ENABLED
enable_multi_output_loss: true
continue_timesteps_after_first: 2
max_outputs_to_train: 3

# Paths
graph_path: config/test_static_graph.pkl
log_path: logs/test_fix/
"""
    
    with open(test_config_path, 'w') as f:
        f.write(test_config)
    
    print(f"📝 Created test configuration: {test_config_path}")
    
    # Test 1: Original batch training
    print("\n" + "="*60)
    print("🔄 TEST 1: ORIGINAL BATCH TRAINING")
    print("="*60)
    
    try:
        print("Training with original batch method...")
        loss_log_batch, _, _ = enhanced_train_context(test_config_path)
        
        print("Evaluating batch-trained model...")
        accuracy_batch, _, _ = evaluate_model(test_config_path, num_samples=50)
        
        print(f"✅ Batch Training Results:")
        print(f"   Final Loss: {loss_log_batch[-1]:.4f}")
        print(f"   Accuracy: {accuracy_batch:.2%}")
        
    except Exception as e:
        print(f"❌ Batch training failed: {e}")
        accuracy_batch = 0.0
        loss_log_batch = []
    
    # Test 2: Fixed single-sample training
    print("\n" + "="*60)
    print("🔧 TEST 2: FIXED SINGLE-SAMPLE TRAINING")
    print("="*60)
    
    try:
        print("Training with FIXED single-sample method...")
        loss_log_single, _, _ = single_sample_train_context(test_config_path)
        
        print("Evaluating single-sample trained model...")
        accuracy_single, _, _ = evaluate_model(test_config_path, num_samples=50)
        
        print(f"✅ Single-Sample Training Results:")
        print(f"   Final Loss: {loss_log_single[-1]:.4f}")
        print(f"   Accuracy: {accuracy_single:.2%}")
        
    except Exception as e:
        print(f"❌ Single-sample training failed: {e}")
        accuracy_single = 0.0
        loss_log_single = []
    
    # Comparison and analysis
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    
    print(f"Batch Training Accuracy:        {accuracy_batch:.2%}")
    print(f"Single-Sample Training Accuracy: {accuracy_single:.2%}")
    
    if accuracy_single > accuracy_batch:
        improvement = accuracy_single - accuracy_batch
        print(f"🎉 IMPROVEMENT: +{improvement:.2%} ({improvement*100:.1f} percentage points)")
        
        if accuracy_single > 0.2:  # Better than 20% (double random chance)
            print("✅ SUCCESS: Single-sample training shows significant improvement!")
        else:
            print("⚠️  PARTIAL: Improvement detected but still low accuracy")
    else:
        print("❌ NO IMPROVEMENT: Single-sample training did not help")
    
    # Plot comparison if we have data
    if loss_log_batch and loss_log_single:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_log_batch, label='Batch Training', color='red', alpha=0.7)
        plt.plot(loss_log_single, label='Single-Sample Training', color='blue', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        accuracies = [accuracy_batch, accuracy_single]
        methods = ['Batch\nTraining', 'Single-Sample\nTraining']
        colors = ['red', 'blue']
        bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('Final Accuracy Comparison')
        plt.ylim(0, max(0.5, max(accuracies) * 1.1))
        
        # Add accuracy labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs("logs/test_fix", exist_ok=True)
        plt.savefig("logs/test_fix/training_comparison.png", dpi=150, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved to: logs/test_fix/training_comparison.png")
        plt.show()
    
    # Cleanup
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    
    return accuracy_batch, accuracy_single


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    batch_acc, single_acc = run_comparison_test()
    
    print("\n" + "="*80)
    print("🏁 FINAL VERDICT")
    print("="*80)
    
    if single_acc > batch_acc + 0.1:  # At least 10% improvement
        print("🎉 MAJOR SUCCESS: Training fix provides significant improvement!")
        print("   Recommendation: Use single-sample training going forward.")
    elif single_acc > batch_acc:
        print("✅ SUCCESS: Training fix shows improvement.")
        print("   Recommendation: Use single-sample training and investigate other issues.")
    else:
        print("❌ INCONCLUSIVE: Training fix did not provide clear improvement.")
        print("   Recommendation: Investigate other potential causes of low accuracy.")
    
    print(f"\nNext steps based on results:")
    if single_acc > 0.3:
        print("- Model is learning! Focus on hyperparameter optimization.")
        print("- Try increasing learning rate further (0.05, 0.1)")
        print("- Extend warmup period and total epochs")
    else:
        print("- Model still struggling. Check:")
        print("  * Class encoding separability")
        print("  * Output node activation patterns")
        print("  * Gradient computation effectiveness")
