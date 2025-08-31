#!/usr/bin/env python3
"""
Fixed main script using single-sample training to resolve the 10% accuracy issue.
This should provide significantly better results than the original main.py
"""

from train.single_sample_train_context import single_sample_train_context
from utils.config import load_config
import matplotlib.pyplot as plt
import os
import random
import torch

from modules.input_adapters import MNISTPCAAdapter
from modules.class_encoding import generate_digit_class_encodings
from modules.output_adapters import predict_label_from_output
from core.tables import ExtendedLookupTableModule
from core.graph import load_or_build_graph
from core.node_store import NodeStore
from core.cell import PhaseCell
from core.forward_engine import run_enhanced_forward


def main():
    config_path = "config/optimized.yaml"
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n🚀 Starting FIXED NeuroGraph Training with Single-Sample Method...")
    print(f"📝 Using optimized configuration: {config_path}")
    print(f"🔧 Key improvements:")
    print(f"   - Single-sample training (matches evaluation)")
    print(f"   - Higher learning rate: {cfg['learning_rate']}")
    print(f"   - Extended warmup: {cfg['warmup_epochs']} epochs")
    print(f"   - More training epochs: {cfg['num_epochs']}")
    
    # Train with FIXED single-sample method
    loss_log, activation_tracker, activation_balancer = single_sample_train_context(config_path)

    # Display diagnostic reports
    print("\n" + "="*80)
    print("📊 FIXED TRAINING COMPLETED - DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if activation_tracker:
        activation_tracker.print_diagnostic_report()
    
    if activation_balancer:
        activation_balancer.print_balance_report()

    # Plot and save convergence
    os.makedirs(cfg.get("log_path", "logs/"), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_log, linewidth=2, color='blue', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FIXED NeuroGraph Training Convergence (Single-Sample)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(cfg.get("log_path", "logs/"), "fixed_loss_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n📉 Saved convergence plot to {out_path}")

    print("\n🔍 Evaluating FIXED model on MNIST samples...")

    # Reload graph and components - filter to only graph-related parameters
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

    # Comprehensive evaluation
    num_samples = 200  # More samples for better accuracy estimate
    correct = 0
    total = num_samples
    predictions = []
    true_labels = []
    confidence_scores = []
    
    print(f"📊 Evaluating on {num_samples} samples...")
    
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
            
        if (i + 1) % 50 == 0:
            current_acc = correct / (i + 1)
            print(f"  Progress: {i + 1}/{total} samples, Current accuracy: {current_acc:.2%}")

    accuracy = correct / total
    
    print(f"\n" + "="*80)
    print("🎯 FINAL EVALUATION RESULTS")
    print("="*80)
    print(f"✅ FIXED Model Accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    
    # Compare with random baseline
    random_baseline = 0.10  # 10% for 10-class classification
    improvement_over_random = accuracy - random_baseline
    
    if accuracy > 0.5:
        print(f"🎉 EXCELLENT: Model is performing very well!")
    elif accuracy > 0.3:
        print(f"✅ GOOD: Significant improvement over random chance!")
    elif accuracy > 0.15:
        print(f"⚠️  MODERATE: Some learning detected, but room for improvement")
    else:
        print(f"❌ POOR: Still close to random performance")
    
    print(f"📈 Improvement over random: +{improvement_over_random:.2%}")
    print(f"📊 Relative improvement: {(accuracy/random_baseline):.1f}x better than random")
    
    # Analyze predictions
    if predictions:
        unique_preds = set(predictions)
        print(f"🔍 Prediction diversity: {len(unique_preds)}/10 classes predicted")
        
        # Count predictions per class
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        print("📋 Prediction distribution:")
        for digit in range(10):
            count = pred_counts.get(digit, 0)
            percentage = (count / total) * 100
            print(f"   Digit {digit}: {count:3d} predictions ({percentage:5.1f}%)")
    
    # Save detailed results
    results_path = os.path.join(cfg.get("log_path", "logs/"), "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"FIXED NeuroGraph Evaluation Results\n")
        f.write(f"=====================================\n")
        f.write(f"Configuration: {config_path}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})\n")
        f.write(f"Improvement over random: {improvement_over_random:.4f}\n")
        f.write(f"Final training loss: {loss_log[-1]:.4f}\n")
        f.write(f"\nPrediction distribution:\n")
        for digit in range(10):
            count = pred_counts.get(digit, 0)
            percentage = (count / total) * 100
            f.write(f"Digit {digit}: {count:3d} ({percentage:5.1f}%)\n")
    
    print(f"\n📄 Detailed results saved to: {results_path}")
    
    # Recommendations based on results
    print(f"\n" + "="*80)
    print("🔮 RECOMMENDATIONS FOR FURTHER IMPROVEMENT")
    print("="*80)
    
    if accuracy > 0.4:
        print("🎯 Model is learning well! Try:")
        print("   - Fine-tune hyperparameters (learning rate, epochs)")
        print("   - Experiment with larger graphs")
        print("   - Test on more complex datasets")
    elif accuracy > 0.2:
        print("📈 Good progress! Next steps:")
        print("   - Increase learning rate further (0.1, 0.2)")
        print("   - Extend training (more epochs)")
        print("   - Analyze class encoding separability")
    else:
        print("🔧 Still needs work. Consider:")
        print("   - Check gradient computation effectiveness")
        print("   - Analyze output node activation patterns")
        print("   - Investigate class encoding strategy")
        print("   - Try different prediction methods")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    main()
