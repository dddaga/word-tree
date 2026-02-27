"""
Ablation Study Runner
Runs experiments with different combinations of fixes enabled/disabled.

Usage:
  cd Neurograph_code
  python experiments/v3_dynamic_phasor/run_ablation.py
  python experiments/v3_dynamic_phasor/run_ablation.py --epochs 30
"""

import argparse
import os
import sys
import yaml
import json
import time

# Ensure Neurograph_code is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
neurograph_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if neurograph_root not in sys.path:
    sys.path.insert(0, neurograph_root)

from experiments.v3_dynamic_phasor.main_v3 import load_config, apply_ablation, run_experiment


ABLATION_CONFIGS = [
    ("baseline",           "Baseline (no fixes)"),
    ("complex_only",       "Issue 1: Complex activation only"),
    ("normalization_only", "Issue 4: Normalization only"),
    ("diversity_only",     "Issue 5: Radiation diversity only"),
    ("balance_only",       "Issue 7: Load balancing only"),
    ("uncertainty_only",   "Issue 9: Uncertainty only"),
    ("all",                "All fixes enabled"),
]


def run_ablation_study(base_config_path: str, epochs: int = 30, device: str = None):
    """Run all ablation experiments and produce comparison table."""
    results = {}

    print("=" * 70)
    print("  V3 DYNAMIC PHASOR - ABLATION STUDY")
    print("=" * 70)

    for ablation_name, description in ABLATION_CONFIGS:
        print(f"\n{'='*70}")
        print(f"  Running: {description}")
        print(f"{'='*70}")

        overrides = {"epochs": epochs}
        if device:
            overrides["device"] = device
        config = load_config(base_config_path, overrides)
        config = apply_ablation(config, ablation_name)

        experiment_name = f"ablation_{ablation_name}"
        try:
            result = run_experiment(config, experiment_name=experiment_name)
            results[ablation_name] = {
                "description": description,
                "test_metrics": result["test_metrics"],
                "complexity": result["complexity"],
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[ablation_name] = {
                "description": description,
                "error": str(e),
            }

    # Print comparison table
    print("\n" + "=" * 90)
    print("  ABLATION COMPARISON TABLE")
    print("=" * 90)
    header = f"{'Ablation':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Params':>12}"
    print(header)
    print("-" * 90)

    for ablation_name, _ in ABLATION_CONFIGS:
        r = results.get(ablation_name, {})
        if "error" in r:
            print(f"{ablation_name:<25} {'ERROR':>10}")
            continue
        tm = r.get("test_metrics", {})
        sc = r.get("complexity", {}).get("space_complexity", {})
        print(
            f"{ablation_name:<25} "
            f"{tm.get('accuracy', 0):>10.4f} "
            f"{tm.get('f1', 0):>10.4f} "
            f"{tm.get('precision', 0):>10.4f} "
            f"{tm.get('recall', 0):>10.4f} "
            f"{sc.get('total_parameters', 0):>12,}"
        )

    print("=" * 90)

    # Save comparison JSON
    log_dir = "logs/v3_experiment/"
    os.makedirs(log_dir, exist_ok=True)
    comparison_path = os.path.join(log_dir, "ablation_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nComparison saved to {comparison_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="V3 Ablation Study")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or os.path.join(script_dir, "config.yaml")
    run_ablation_study(config_path, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
