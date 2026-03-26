"""CLI for running GA hyperparameter search by experiment name.

Usage:
    python scripts/run_ga_search.py --experiment stageA
    python scripts/run_ga_search.py --experiment exp1 --output results/exp1_ga.json
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import torch

from src.training.ga_search import GASearch, SEARCH_SPACE_AB, SEARCH_SPACE_C


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GA hyperparameter search")
    parser.add_argument(
        "--experiment",
        choices=["stageA", "exp1", "exp2"],
        required=True,
        help="Experiment name (stageA, exp1, exp2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/{experiment}_ga_results.json)",
    )
    parser.add_argument(
        "--store", default="data/store.h5",
        help="Path to HDF5 tensor store",
    )
    args = parser.parse_args()

    # Load data from HDF5
    with h5py.File(args.store, "r") as f:
        train_features = torch.tensor(f["train/features"][:])
        train_soft_labels = torch.tensor(f["train/soft_labels"][:])
        train_labels = torch.tensor(f["train/labels"][:])
        val_features = torch.tensor(f["val/features"][:])
        val_soft_labels = torch.tensor(f["val/soft_labels"][:])
        val_labels = torch.tensor(f["val/labels"][:])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    search_space = SEARCH_SPACE_C if args.experiment == "exp2" else SEARCH_SPACE_AB

    ga = GASearch(
        search_space=search_space,
        experiment_name=args.experiment,
        train_features=train_features,
        train_soft_labels=train_soft_labels,
        train_labels=train_labels,
        val_features=val_features,
        val_soft_labels=val_soft_labels,
        val_labels=val_labels,
        n_in=train_features.shape[1],
        device=device,
    )
    result = ga.run()

    output_path = args.output or f"results/{args.experiment}_ga_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Best score: {result['best_score']:.4f}")
    print(f"Best config: {result['best_config']}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
