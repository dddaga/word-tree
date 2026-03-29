"""Extended Exp 1 (Stage B): GA search over sparsity levels + longer training.

Extends the Phase 4 Exp 1 run by:
  - Adding sparsity (0.70 / 0.80 / 0.90 / 0.95) to the GA search space
  - Running full training for 150 epochs (vs 50 in the original)

Output:
  results/exp1_ext_ga_results.json   -- best config including sparsity
  results/exp1_ext_full.json         -- full metrics after 150-epoch training
  checkpoints/exp1_ext_best.pt       -- best model checkpoint
"""

from __future__ import annotations

import json
import math
import os
import random

import h5py
import torch
import torch.utils.data

from src.sgnnet.model_wave import SGNNET_Wave
from src.training.trainer import Trainer
from src.utils.metrics import compute_all_metrics

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

CLASS_NAMES = [
    "tench", "english_springer", "cassette_player", "chain_saw", "church",
    "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute",
]
VGG16_FC_PARAMS = 123_642_856
FULL_EPOCHS = 150

# Search space: adds sparsity to SEARCH_SPACE_AB
SEARCH_SPACE: dict[str, dict] = {
    "K":             {"type": "discrete",       "values": [1, 2, 3, 4]},
    "N_hidden":      {"type": "discrete",       "values": [64, 128, 256]},
    "lr_Wpos":       {"type": "continuous_log", "low": 1e-5, "high": 1e-2},
    "lambda_safety": {"type": "continuous",     "low": 0.0, "high": 1.0},
    "batch_size":    {"type": "discrete",       "values": [64, 128, 256]},
    "sparsity":      {"type": "discrete",       "values": [0.70, 0.80, 0.90, 0.95]},
}


# -------------------------------------------------------------------
# GA helpers
# -------------------------------------------------------------------

def _sample(spec: dict):
    if spec["type"] == "discrete":
        return random.choice(spec["values"])
    if spec["type"] == "continuous":
        return random.uniform(spec["low"], spec["high"])
    lo, hi = math.log10(spec["low"]), math.log10(spec["high"])
    return 10 ** random.uniform(lo, hi)


def _mutate(individual: dict) -> dict:
    result = {}
    for k, spec in SEARCH_SPACE.items():
        if random.random() < 0.3:
            if spec["type"] == "discrete":
                result[k] = random.choice(spec["values"])
            elif spec["type"] == "continuous":
                v = individual[k] + random.gauss(0, 0.1 * (spec["high"] - spec["low"]))
                result[k] = max(spec["low"], min(spec["high"], v))
            else:
                v = individual[k] * (10 ** random.gauss(0, 0.3))
                result[k] = max(spec["low"], min(spec["high"], v))
        else:
            result[k] = individual[k]
    return result


def _params_min(n_in: int) -> int:
    m = SGNNET_Wave(N_hidden=64, K=1, D=4, N_in=n_in)
    return sum(p.numel() for p in m.parameters())


def _evaluate(config: dict, data: tuple, device: str, params_min: int) -> float:
    """Train on 15% data for 15 epochs; return efficiency-ratio fitness."""
    train_f, train_sl, train_l, val_f, val_sl, val_l = data
    n = train_f.shape[0]
    idx = torch.randperm(n)[: max(1, int(n * 0.15))]
    partial_ds = torch.utils.data.TensorDataset(
        train_f[idx], train_sl[idx], train_l[idx]
    )
    val_ds = torch.utils.data.TensorDataset(val_f, val_sl, val_l)
    bs = config["batch_size"]
    train_loader = torch.utils.data.DataLoader(partial_ds, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs)

    model = SGNNET_Wave(
        N_hidden=config["N_hidden"], K=config["K"], D=4, N_in=train_f.shape[1],
        sparsity=config["sparsity"], use_proximity=True, use_wphase=False,
    )
    model_params = sum(p.numel() for p in model.parameters())

    trainer = Trainer(
        model, train_loader, val_loader,
        lr_wpos=config["lr_Wpos"], lambda_safety=config["lambda_safety"],
        device=device,
    )
    history = trainer.train(30)

    if any(h.get("nan_detected", False) for h in history):
        return -1e6
    losses = [h["train_loss"] for h in history if not math.isnan(h["train_loss"])]
    if not losses:
        return -1e6
    final_loss = sum(losses[-3:]) / len(losses[-3:])
    if math.isnan(final_loss) or math.isinf(final_loss):
        return -1e6

    return -final_loss * (params_min / model_params) ** 0.2


# -------------------------------------------------------------------
# GA search
# -------------------------------------------------------------------

def run_ga(data: tuple, device: str, population: int = 20, generations: int = 10) -> dict:
    n_in = data[0].shape[1]
    pm = _params_min(n_in)
    pop = [{k: _sample(s) for k, s in SEARCH_SPACE.items()} for _ in range(population)]
    best_config, best_score = {}, -float("inf")
    history = []

    for gen in range(generations):
        scored = [(cfg, _evaluate(cfg, data, device, pm)) for cfg in pop]
        gen_best_cfg, gen_best_score = max(scored, key=lambda x: x[1])
        if gen_best_score > best_score:
            best_score, best_config = gen_best_score, gen_best_cfg
        history.append({"generation": gen, "best_score": gen_best_score})
        print(f"Gen {gen}: best_score={gen_best_score:.4f}  sparsity={gen_best_cfg['sparsity']}  N_hidden={gen_best_cfg['N_hidden']}  (fitness = -train_loss * param_penalty)")

        top5 = [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:5]]
        pop = list(top5)
        while len(pop) < population:
            pop.append(_mutate(random.choice(top5)))

    return {"best_config": best_config, "best_score": best_score, "history": history}


# -------------------------------------------------------------------
# Full training (150 epochs)
# -------------------------------------------------------------------

def run_full_training(best: dict, data: tuple, device: str) -> tuple:
    train_f, train_sl, train_l, val_f, val_sl, val_l = data
    bs = best["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_f, train_sl, train_l),
        batch_size=bs, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_f, val_sl, val_l),
        batch_size=bs, num_workers=0,
    )
    model = SGNNET_Wave(
        N_hidden=best["N_hidden"], K=best["K"], D=4, N_in=train_f.shape[1],
        sparsity=best["sparsity"], use_proximity=True, use_wphase=False,
    )
    trainer = Trainer(
        model, train_loader, val_loader,
        lr_wpos=best["lr_Wpos"], lambda_safety=best["lambda_safety"],
        device=device,
    )
    history = trainer.train(FULL_EPOCHS)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/exp1_ext_best.pt")
    return model, trainer, history


# -------------------------------------------------------------------
# Evaluate and save
# -------------------------------------------------------------------

def evaluate_and_save(model, trainer, history, best: dict) -> None:
    result = trainer.evaluate()
    scores_np = result["scores"].numpy()
    labels_np = result["labels"].numpy()
    metrics = compute_all_metrics(scores_np, labels_np, CLASS_NAMES)

    total_params = sum(p.numel() for p in model.parameters())
    out = {
        "experiment": "exp1_extended",
        "hyperparams": best,
        "full_epochs": FULL_EPOCHS,
        "top1_accuracy": metrics["top1_accuracy"],
        "mAP": metrics["mAP"],
        "per_class": metrics["per_class"],
        "params": total_params,
        "percent_of_vgg16_fc": round(total_params / VGG16_FC_PARAMS * 100, 4),
        "sparsity": {
            "C_input": float((~model.C_input_mask).float().mean()),
            "C_hh":    float((~model.C_hh_mask).float().mean()),
            "C_ho":    float((~model.C_ho_mask).float().mean()),
        },
        "training_history": [
            {"epoch": i, "train_loss": h["train_loss"]} for i, h in enumerate(history)
        ],
    }
    os.makedirs("results", exist_ok=True)
    with open("results/exp1_ext_full.json", "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"Exp1-extended: top1={metrics['top1_accuracy']:.4f}, "
        f"mAP={metrics['mAP']:.4f}, params={total_params}, "
        f"sparsity={best['sparsity']}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    with h5py.File("data/store.h5", "r") as f:
        data = (
            torch.tensor(f["train/features"][:]),
            torch.tensor(f["train/soft_labels"][:]),
            torch.tensor(f["train/labels"][:]),
            torch.tensor(f["val/features"][:]),
            torch.tensor(f["val/soft_labels"][:]),
            torch.tensor(f["val/labels"][:]),
        )
    print(f"Loaded data: train={data[0].shape[0]}, val={data[3].shape[0]}")

    ga_result = run_ga(data, device)
    best = ga_result["best_config"]
    print(f"GA best config: {best}")

    os.makedirs("results", exist_ok=True)
    with open("results/exp1_ext_ga_results.json", "w") as f:
        json.dump(ga_result, f, indent=2)

    model, trainer, history = run_full_training(best, data, device)
    evaluate_and_save(model, trainer, history, best)


if __name__ == "__main__":
    main()
