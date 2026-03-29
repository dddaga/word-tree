"""Stage A training: GA hyperparameter search then full training.

Stage A = static binary C, no dynamic routing (use_proximity=False).
Establishes the baseline that Stages B and C must beat.

Output:
  results/stageA_ga_results.json  -- best hyperparams from GA search
  results/stageA_full.json        -- full training metrics (top1, mAP, per-class)
  checkpoints/stageA_best.pt      -- best model checkpoint
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch

from src.sgnnet.model_wave import SGNNET_Wave
from src.training.ga_search import GASearch, SEARCH_SPACE_AB
from src.training.trainer import Trainer
from src.utils.metrics import compute_all_metrics

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

CLASS_NAMES = [
    "tench", "english_springer", "cassette_player", "chain_saw", "church",
    "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute",
]

VGG16_FC_PARAMS = 123642856


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_data() -> tuple:
    """Load train/val features, soft labels, and labels from HDF5 store."""
    with h5py.File("data/store.h5", "r") as f:
        train_features = torch.tensor(f["train/features"][:])
        train_soft_labels = torch.tensor(f["train/soft_labels"][:])
        train_labels = torch.tensor(f["train/labels"][:])
        val_features = torch.tensor(f["val/features"][:])
        val_soft_labels = torch.tensor(f["val/soft_labels"][:])
        val_labels = torch.tensor(f["val/labels"][:])
    return (
        train_features, train_soft_labels, train_labels,
        val_features, val_soft_labels, val_labels,
    )


# -------------------------------------------------------------------
# Phase 1: GA search (15% data, 15 epochs per eval)
# -------------------------------------------------------------------

def run_ga_search(
    train_features, train_soft_labels, train_labels,
    val_features, val_soft_labels, val_labels,
    device: str,
) -> dict:
    """Run GA hyperparameter search for Stage A."""
    ga = GASearch(
        search_space=SEARCH_SPACE_AB,
        experiment_name="stageA",
        train_features=train_features,
        train_soft_labels=train_soft_labels,
        train_labels=train_labels,
        val_features=val_features,
        val_soft_labels=val_soft_labels,
        val_labels=val_labels,
        population=20, generations=10, top_k=5,
        partial_fraction=0.15, epochs_per_eval=15,
        n_in=25088, device=device,
    )
    ga_result = ga.run()

    os.makedirs("results", exist_ok=True)
    with open("results/stageA_ga_results.json", "w") as f:
        json.dump(ga_result, f, indent=2)
    print(f"GA best config: {ga_result['best_config']}")
    return ga_result


# -------------------------------------------------------------------
# Phase 2: Full training with best config (50 epochs)
# -------------------------------------------------------------------

def run_full_training(best: dict, data: tuple, device: str) -> tuple:
    """Train SGNNET_Wave Stage A with best GA config."""
    train_features, train_soft_labels, train_labels = data[:3]
    val_features, val_soft_labels, val_labels = data[3:]

    model = SGNNET_Wave(
        N_hidden=best["N_hidden"], N_out=10, D=4, N_in=25088,
        sparsity=0.90, K=best["K"], box_size=1.0,
        use_proximity=False,  # Stage A: no dynamic routing
    )

    train_ds = torch.utils.data.TensorDataset(
        train_features, train_soft_labels, train_labels,
    )
    val_ds = torch.utils.data.TensorDataset(
        val_features, val_soft_labels, val_labels,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=best["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=best["batch_size"], num_workers=0,
    )

    trainer = Trainer(
        model, train_loader, val_loader,
        lr_wpos=best["lr_Wpos"],
        lambda_safety=best["lambda_safety"],
        device=device,
    )
    history = trainer.train(n_epochs=50)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/stageA_best.pt")
    return model, trainer, history


# -------------------------------------------------------------------
# Phase 3: Final evaluation and result assembly
# -------------------------------------------------------------------

def evaluate_and_save(model, trainer, history, best: dict) -> dict:
    """Evaluate model, compute all metrics, save results JSON."""
    eval_result = trainer.evaluate()
    scores_np = eval_result["scores"].numpy()
    labels_np = eval_result["labels"].numpy()

    metrics = compute_all_metrics(scores_np, labels_np, CLASS_NAMES)

    # Parameter accounting (TRAIN-04)
    total_params = sum(p.numel() for p in model.parameters())

    # Sparsity check (TRAIN-05)
    c_input_sparsity = float((~model.C_input_mask).float().mean())
    c_hh_sparsity = float((~model.C_hh_mask).float().mean())
    c_ho_sparsity = float((~model.C_ho_mask).float().mean())

    result = {
        "experiment": "stageA_static_baseline",
        "hyperparams": best,
        "top1_accuracy": metrics["top1_accuracy"],
        "mAP": metrics["mAP"],
        "per_class": metrics["per_class"],  # TRAIN-06: full per_class dict
        "params": total_params,
        "percent_of_vgg16_fc": round(total_params / VGG16_FC_PARAMS * 100, 2),
        "sparsity": {
            "C_input": c_input_sparsity,
            "C_hh": c_hh_sparsity,
            "C_ho": c_ho_sparsity,
        },
        "training_history": [
            {"epoch": i, "train_loss": h["train_loss"]}
            for i, h in enumerate(history)
        ],
    }

    with open("results/stageA_full.json", "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Stage A: top1={metrics['top1_accuracy']:.4f}, "
        f"mAP={metrics['mAP']:.4f}, params={total_params}"
    )
    return result


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    data = load_data()
    print(f"Loaded data: train={data[0].shape[0]}, val={data[3].shape[0]}")

    ga_result = run_ga_search(*data, device=device)
    best = ga_result["best_config"]

    model, trainer, history = run_full_training(best, data, device)
    evaluate_and_save(model, trainer, history, best)


if __name__ == "__main__":
    main()
