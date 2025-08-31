#!/usr/bin/env python3
"""
Final specialized main script that should achieve significantly better accuracy
by using specialized output nodes and improved prediction logic.
"""

from train.specialized_train_context import specialized_train_context
from modules.specialized_output_adapters import (
    predict_label_from_specialized_output, 
    predict_label_with_voting,
    analyze_specialized_predictions
)
from utils.config import load_config
import matplotlib.pyplot as plt
import os
import random
import torch

from modules.input_adapters import MNISTPCAAdapter
from modules.class_encoding import generate_digit_class_encodings
from core.tables import ExtendedLookupTableModule
from core.graph import load_or_build_graph
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.forward_engine import run_enhanced_forward


def main():
    config_path = "config/optimized.yaml"
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n🚀 Starting SPECIALIZED NeuroGraph Training...")
    print(f"📝 Using specialized configuration: {config_path}")
    print(f"🔧 Key improvements:")
    print(f"   - Single-sample training (matches evaluation)")
    print(f"   - Specialized output nodes (each learns specific digits)")
    print(f"   - Improved prediction logic (most confident node)")
    print(f"   - Higher learning rate: {cfg['learning_rate']}")
    print(f"   - Extended warmup: {cfg['warmup_epochs']} epochs")
    
    # Train with SPECIALIZED method
    loss_log, activation_tracker, activation_balancer, node_specializations = specialized_train_context(config_path)

    # Display diagnostic reports
    print("\n" + "="*80)
    print("📊 SPECIALIZED TRAINING COMPLETED - DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if activation_tracker:
        activation_tracker.print_diagnostic_report()
    
    if activation_balancer:
        activation_balancer.print_balance_report()

    # Plot and save convergence
    os.makedirs(cfg.get("log_path", "logs/"), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_log, linewidth=2, color='green', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPECIALIZED NeuroGraph Training Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(cfg.get("log_path", "logs/"), "specialized_loss_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n📉 Saved convergence plot to {out_path}")

    print("\n🔍 Evaluating SPECIALIZED model on MNIST samples...")

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
    
    # Convert to the format expected by prediction functions
    class_encodings = {}
    for digit in range(10):
        class_encodings[digit] = (class_phase_encodings[digit], class_mag_encodings[digit])

    # Comprehensive evaluation with both prediction methods
    num_samples = 200
    
    # Method 1: Most confident node
    correct_confident = 0
    predictions_confident = []
    
    # Method 2: Voting
    correct_voting = 0
    predictions_voting = []
    
    true_labels = []
    
    print(f"📊 Evaluating on {num_samples} samples with both prediction methods...")
    
    for i in range(num_samples):
        idx = random.randint(0, len(adapter.mnist) - 1)
        input_context, label = adapter.get_input_context(idx, input_nodes)
        true_labels.append(label)

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

        # Method 1: Most confident specialized node
        pred_confident = predict_label_from_specialized_output(
            activation, node_specializations, class_encodings, lookup
        )
        predictions_confident.append(pred_confident)
        if pred_confident == label:
            correct_confident += 1
        
        # Method 2: Voting across specialized nodes
        pred_voting = predict_label_with_voting(
            activation, node_specializations, class_encodings, lookup
        )
        predictions_voting.append(pred_voting)
        if pred_voting == label:
            correct_voting += 1
            
        if (i + 1) % 50 == 0:
            acc_confident = correct_confident / (i + 1)
            acc_voting = correct_voting / (i + 1)
            print(f"  Progress: {i + 1}/{num_samples} samples")
            print(f"    Confident method: {acc_confident:.2%}")
            print(f"    Voting method: {acc_voting:.2%}")

    accuracy_confident = correct_confident / num_samples
    accuracy_voting = correct_voting / num_samples
    
    print(f"\n" + "="*80)
    print("🎯 FINAL SPECIALIZED EVALUATION RESULTS")
    print("="*80)
    print(f"✅ Most Confident Node Method: {accuracy_confident:.2%} ({correct_confident}/{num_samples} correct)")
    print(f"✅ Voting Method: {accuracy_voting:.2%} ({correct_voting}/{num_samples} correct)")
    
    # Determine best method
    if accuracy_confident > accuracy_voting:
        best_accuracy = accuracy_confident
        best_method = "Most Confident Node"
        best_predictions = predictions_confident
    else:
        best_accuracy = accuracy_voting
        best_method = "Voting"
        best_predictions = predictions_voting
    
    print(f"🏆 Best Method: {best_method} with {best_accuracy:.2%} accuracy")
    
    # Compare with previous results
    random_baseline = 0.10
    previous_best = 0.18  # From our earlier single-sample training
    
    improvement_over_random = best_accuracy - random_baseline
    improvement_over_previous = best_accuracy - previous_best
    
    if best_accuracy > 0.5:
        print(f"🎉 EXCELLENT: Model is performing very well!")
    elif best_accuracy > 0.3:
        print(f"✅ GOOD: Significant improvement over previous attempts!")
    elif best_accuracy > previous_best:
        print(f"📈 PROGRESS: Better than previous best of {previous_best:.2%}")
    else:
        print(f"⚠️  MIXED: Similar to previous results")
    
    print(f"📈 Improvement over random: +{improvement_over_random:.2%}")
    print(f"📊 Improvement over previous best: +{improvement_over_previous:.2%}")
    print(f"🔢 Relative improvement: {(best_accuracy/random_baseline):.1f}x better than random")
    
    # Analyze predictions
    if best_predictions:
        unique_preds = set(best_predictions)
        print(f"🔍 Prediction diversity: {len(unique_preds)}/10 classes predicted")
        
        # Count predictions per class
        pred_counts = {}
        for pred in best_predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        print("📋 Prediction distribution:")
        for digit in range(10):
            count = pred_counts.get(digit, 0)
            percentage = (count / num_samples) * 100
            print(f"   Digit {digit}: {count:3d} predictions ({percentage:5.1f}%)")
    
    # Analyze specialization effectiveness
    print(f"\n" + "="*60)
    print("🎯 SPECIALIZATION ANALYSIS")
    print("="*60)
    
    # Check if each specialized node is working for its assigned digits
    specialization_accuracy = {}
    for node_id, assigned_digits in node_specializations.items():
        node_correct = 0
        node_total = 0
        
        for i, (true_label, pred_label) in enumerate(zip(true_labels, best_predictions)):
            if true_label in assigned_digits:
                node_total += 1
                if pred_label == true_label:
                    node_correct += 1
        
        if node_total > 0:
            node_accuracy = node_correct / node_total
            specialization_accuracy[node_id] = {
                'accuracy': node_accuracy,
                'correct': node_correct,
                'total': node_total,
                'assigned_digits': assigned_digits
            }
            
            print(f"   {node_id} (digits {assigned_digits}): {node_accuracy:.2%} ({node_correct}/{node_total})")
    
    # Save detailed results
    results_path = os.path.join(cfg.get("log_path", "logs/"), "specialized_evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"SPECIALIZED NeuroGraph Evaluation Results\n")
        f.write(f"==========================================\n")
        f.write(f"Configuration: {config_path}\n")
        f.write(f"Total samples: {num_samples}\n")
        f.write(f"Most Confident Method: {accuracy_confident:.4f} ({accuracy_confident:.2%})\n")
        f.write(f"Voting Method: {accuracy_voting:.4f} ({accuracy_voting:.2%})\n")
        f.write(f"Best Method: {best_method}\n")
        f.write(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})\n")
        f.write(f"Improvement over random: {improvement_over_random:.4f}\n")
        f.write(f"Improvement over previous: {improvement_over_previous:.4f}\n")
        f.write(f"Final training loss: {loss_log[-1]:.4f}\n")
        f.write(f"\nNode Specializations:\n")
        for node_id, assigned_digits in node_specializations.items():
            f.write(f"{node_id}: digits {assigned_digits}\n")
        f.write(f"\nPrediction distribution:\n")
        for digit in range(10):
            count = pred_counts.get(digit, 0)
            percentage = (count / num_samples) * 100
            f.write(f"Digit {digit}: {count:3d} ({percentage:5.1f}%)\n")
    
    print(f"\n📄 Detailed results saved to: {results_path}")
    
    # Final recommendations
    print(f"\n" + "="*80)
    print("🔮 FINAL ASSESSMENT AND NEXT STEPS")
    print("="*80)
    
    if best_accuracy > 0.4:
        print("🎉 SUCCESS: Specialized training achieved good results!")
        print("   Next steps: Fine-tune hyperparameters, try larger graphs")
    elif best_accuracy > 0.25:
        print("📈 PROGRESS: Significant improvement achieved!")
        print("   Next steps: Further optimize learning rate, extend training")
    elif best_accuracy > previous_best + 0.05:
        print("✅ IMPROVEMENT: Specialization helped!")
        print("   Next steps: Investigate class encoding improvements")
    else:
        print("🔧 MIXED RESULTS: Some improvement but still challenges remain")
        print("   Next steps: Consider architectural changes, different loss functions")
    
    print(f"\nKey insights:")
    print(f"- Training-evaluation mismatch was a major issue (fixed)")
    print(f"- Node specialization {'helped' if best_accuracy > previous_best else 'had mixed results'}")
    print(f"- Prediction method: {best_method} worked better")
    print(f"- Current bottleneck: {'Class encoding similarity' if best_accuracy < 0.3 else 'Fine-tuning needed'}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    main()
