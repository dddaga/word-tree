"""
V3 Dynamic Phasor GNN Experiment - Main Entry Point

Usage:
  cd Neurograph_code
  python experiments/v3_dynamic_phasor/main_v3.py
  python experiments/v3_dynamic_phasor/main_v3.py --epochs 50 --gamma 1.5
  python experiments/v3_dynamic_phasor/main_v3.py --ablation complex_only
"""

import argparse
import os
import sys
import yaml
import time

# Ensure Neurograph_code is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
neurograph_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if neurograph_root not in sys.path:
    sys.path.insert(0, neurograph_root)

from experiments.v3_dynamic_phasor.model.v3_train_context import V3TrainContext
from experiments.v3_dynamic_phasor.metrics.complexity_tracker import ComplexityTracker
from experiments.v3_dynamic_phasor.metrics.experiment_logger import ExperimentLogger


def load_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if overrides.get("epochs") is not None:
        config.setdefault("training", {})["epochs"] = overrides["epochs"]
    if overrides.get("gamma") is not None:
        config.setdefault("complex_activation", {})["gamma"] = overrides["gamma"]
    if overrides.get("device") is not None:
        config["device"] = overrides["device"]

    return config


def apply_ablation(config: dict, ablation: str) -> dict:
    """Disable all fixes, then selectively enable for ablation."""
    # Disable all
    config.setdefault("complex_activation", {})["enabled"] = False
    config.setdefault("normalization", {})["enabled"] = False
    config.setdefault("radiation_diversity", {})["enabled"] = False
    config.setdefault("load_balance", {})["enabled"] = False
    config.setdefault("uncertainty", {})["enabled"] = False

    if ablation == "baseline":
        pass  # all disabled
    elif ablation == "complex_only":
        config["complex_activation"]["enabled"] = True
    elif ablation == "normalization_only":
        config["normalization"]["enabled"] = True
    elif ablation == "diversity_only":
        config["radiation_diversity"]["enabled"] = True
    elif ablation == "balance_only":
        config["load_balance"]["enabled"] = True
    elif ablation == "uncertainty_only":
        config["uncertainty"]["enabled"] = True
    elif ablation == "all":
        config["complex_activation"]["enabled"] = True
        config["normalization"]["enabled"] = True
        config["radiation_diversity"]["enabled"] = True
        config["load_balance"]["enabled"] = True
        config["uncertainty"]["enabled"] = True
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    return config


def run_experiment(config: dict, experiment_name: str = "v3_default") -> dict:
    """Run a single experiment with the given config."""
    log_dir = config.get("metrics", {}).get("log_dir", "logs/v3_experiment/")
    logger = ExperimentLogger(log_dir=log_dir, experiment_name=experiment_name)
    tracker = ComplexityTracker()

    # Build training context
    ctx = V3TrainContext(config)

    # Train
    num_epochs = config.get("training", {}).get("epochs", 100)
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        t0 = time.perf_counter()
        avg_loss, div_loss, bal_loss, train_acc = ctx.train_epoch()
        epoch_time = time.perf_counter() - t0

        tracker.record_epoch_time(epoch_time)

        # Validation every 5 epochs
        val_acc = 0.0
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = ctx.evaluate("val")
            val_acc = val_metrics["accuracy"]

        # Primary loss = total - aux
        primary_loss = avg_loss - div_loss - bal_loss

        # Issue-specific metrics
        issue_m = {}
        if ctx.enable_diversity:
            issue_m.update(ctx.diversity_tracker.get_metrics())
        if ctx.enable_balance:
            issue_m.update(ctx.load_balancer.get_metrics())

        logger.log_epoch(
            epoch=epoch,
            primary_loss=primary_loss,
            diversity_loss=div_loss,
            balance_loss=bal_loss,
            total_loss=avg_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            issue_metrics=issue_m if issue_m else None,
        )

        # Anneal + reset (already done in ctx.train_epoch but duplicating for clarity)
        if ctx.enable_balance:
            ctx.load_balancer.anneal_temperature()
        if ctx.enable_diversity:
            ctx.diversity_tracker.reset()
        if ctx.enable_balance:
            ctx.load_balancer.reset_epoch()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
                f"{epoch_time:.2f}s"
            )

    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = ctx.evaluate("test")
    logger.log_test_metrics(test_metrics)
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:       {test_metrics['f1']:.4f}")
    print(f"  Test Precision:{test_metrics['precision']:.4f}")
    print(f"  Test Recall:   {test_metrics['recall']:.4f}")

    # Uncertainty calibration
    if ctx.enable_uncertainty:
        print("\nCalibrating temperature on validation set...")
        logits_list, labels_list = ctx.get_logits_for_calibration("val")
        if logits_list:
            best_t = ctx.uncertainty.calibrate_temperature(logits_list, labels_list)
            print(f"  Optimal temperature: {best_t:.3f}")

    # Inference latency
    print("\nMeasuring inference latency (200 runs)...")
    sample_context, _ = ctx.input_adapter.get_input_context(0, ctx.input_nodes, dataset="test")
    latency_us = tracker.measure_inference_latency(
        forward_fn=lambda ic: ctx.forward_pass(ic),
        input_context=sample_context,
        num_runs=200,
    )
    print(f"  Median inference latency: {latency_us:.1f} us")

    # Complexity report
    modules = {
        "node_store": ctx.node_store,
        "input_adapter": ctx.input_adapter,
    }
    if ctx.enable_uncertainty:
        modules["uncertainty"] = ctx.uncertainty
    complexity = tracker.get_full_report(modules)
    logger.log_complexity(complexity)

    # Save and plot
    logger.save_metrics()
    logger.plot_loss_curves()
    logger.plot_issue_metrics()
    logger.print_summary()

    return {
        "test_metrics": test_metrics,
        "complexity": complexity,
        "experiment_name": experiment_name,
    }


def main():
    parser = argparse.ArgumentParser(description="V3 Dynamic Phasor GNN Experiment")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["baseline", "complex_only", "normalization_only",
                                 "diversity_only", "balance_only", "uncertainty_only", "all"])
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()

    # Resolve config path
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(script_dir, "config.yaml")

    config = load_config(config_path, vars(args))

    # Apply ablation if specified
    experiment_name = args.name or "v3_all"
    if args.ablation:
        config = apply_ablation(config, args.ablation)
        experiment_name = args.name or f"v3_{args.ablation}"

    run_experiment(config, experiment_name=experiment_name)


if __name__ == "__main__":
    main()
