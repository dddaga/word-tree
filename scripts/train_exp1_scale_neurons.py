"""Exp 1 (Stage B) neuron scaling sweep.

Reuses the GA best config from exp1_ga_results.json (K, lr_Wpos, lambda_safety,
batch_size) and sweeps N_hidden over [512, 1024, 2048, 4096] for 150 epochs each.

The original GA only searched [64, 128, 256]. This establishes whether
more hidden neurons improve accuracy given the same position-based learning.

Parameter counts with sparsity=0.90 (C masks stored as bool):
  N_hidden=512   →    2,088 params  buffers~0.05GB
  N_hidden=1024  →    4,136 params  buffers~0.11GB
  N_hidden=2048  →    8,232 params  buffers~0.22GB
  N_hidden=4096  →   16,424 params  buffers~0.12GB
  N_hidden=10000 →   40,040 params  buffers~0.35GB
  N_hidden=50000 →  200,040 params  buffers~3.75GB

Output per run:
  results/exp1_scale_{N}.json          -- metrics
  checkpoints/exp1_scale_{N}_best.pt   -- checkpoint
Summary across all runs:
  results/exp1_neuron_scaling.json
"""

from __future__ import annotations

import json
import os

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
N_HIDDEN_SWEEP = [512, 1024, 2048, 4096, 10000]


def load_data() -> tuple:
    with h5py.File("data/store.h5", "r") as f:
        return (
            torch.tensor(f["train/features"][:]),
            torch.tensor(f["train/soft_labels"][:]),
            torch.tensor(f["train/labels"][:]),
            torch.tensor(f["val/features"][:]),
            torch.tensor(f["val/soft_labels"][:]),
            torch.tensor(f["val/labels"][:]),
        )


def run_one(n_hidden: int, best: dict, data: tuple, device: str) -> dict:
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
        N_hidden=n_hidden, K=best["K"], D=4, N_in=train_f.shape[1],
        sparsity=0.90, use_proximity=True, use_wphase=False,
    )
    total_params = sum(p.numel() for p in model.parameters())

    # Scale lambda_safety by (256/N)^(1/D) so repulsion strength stays comparable
    # as neuron density increases. Also, safety_valve_loss is disabled above N=5000
    # (O(N²) cdist would OOM), so set to 0.0 there.
    BASE_N = 256
    D_geom = 4
    if n_hidden > 5000:
        lambda_safety = 0.0   # cdist OOM guard kicks in — no safety loss
    else:
        lambda_safety = best["lambda_safety"] * (BASE_N / n_hidden) ** (1.0 / D_geom)

    trainer = Trainer(
        model, train_loader, val_loader,
        lr_wpos=best["lr_Wpos"],
        lambda_safety=lambda_safety,
        device=device,
        use_amp=True,
    )

    print(f"  lambda_safety={lambda_safety:.4f} (scaled from {best['lambda_safety']:.3f})")

    def log(m):
        if m["epoch"] % 25 == 0 or m["epoch"] < 3:
            print(
                f"  [N={n_hidden}] epoch {m['epoch']:3d}  "
                f"train_loss={m['train_loss']:.4f}  val_loss={m['val_loss']:.4f}"
            )

    history = trainer.train(FULL_EPOCHS, log_fn=log)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/exp1_scale_{n_hidden}_best.pt")

    result = trainer.evaluate()
    metrics = compute_all_metrics(
        result["scores"].numpy(), result["labels"].numpy(), CLASS_NAMES
    )

    out = {
        "experiment": f"exp1_scale_{n_hidden}",
        "N_hidden": n_hidden,
        "hyperparams": {**best, "N_hidden": n_hidden},
        "full_epochs": FULL_EPOCHS,
        "top1_accuracy": metrics["top1_accuracy"],
        "mAP": metrics["mAP"],
        "params": total_params,
        "percent_of_vgg16_fc": round(total_params / VGG16_FC_PARAMS * 100, 4),
        "training_history": [
            {"epoch": i, "train_loss": h["train_loss"], "val_loss": h["val_loss"]}
            for i, h in enumerate(history)
        ],
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/exp1_scale_{n_hidden}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"[N={n_hidden}] top1={metrics['top1_accuracy']:.4f}  "
        f"mAP={metrics['mAP']:.4f}  params={total_params}"
    )
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Sweeping N_hidden={N_HIDDEN_SWEEP}  epochs={FULL_EPOCHS}")

    # Load GA best config (K, lr_Wpos, lambda_safety, batch_size)
    with open("results/exp1_ga_results.json") as f:
        ga_result = json.load(f)
    best = ga_result["best_config"]
    print(f"Base config (from GA): K={best['K']}, lr_Wpos={best['lr_Wpos']:.4e}, "
          f"lambda_safety={best['lambda_safety']:.3f}, batch_size={best['batch_size']}")

    data = load_data()
    print(f"Loaded data: train={data[0].shape[0]}, val={data[3].shape[0]}\n")

    results = []
    for n_hidden in N_HIDDEN_SWEEP:
        print(f"\n--- N_hidden={n_hidden} ---")
        r = run_one(n_hidden, best, data, device)
        results.append(r)

    # Summary table
    print("\n=== Neuron Scaling Summary ===")
    print(f"{'N_hidden':>8}  {'params':>8}  {'top-1':>8}  {'mAP':>8}")
    print(f"{'256 (orig)':>8}  {'1064':>8}  {'0.1197':>8}  {'0.1160':>8}  (Exp1 baseline)")
    for r in results:
        print(
            f"{r['N_hidden']:>8}  {r['params']:>8}  "
            f"{r['top1_accuracy']:>8.4f}  {r['mAP']:>8.4f}"
        )

    summary = {
        "base_config": best,
        "baseline_n256": {"N_hidden": 256, "params": 1064, "top1": 0.1197, "mAP": 0.1160},
        "sweep": [
            {"N_hidden": r["N_hidden"], "params": r["params"],
             "top1_accuracy": r["top1_accuracy"], "mAP": r["mAP"]}
            for r in results
        ],
    }
    with open("results/exp1_neuron_scaling.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: results/exp1_neuron_scaling.json")


if __name__ == "__main__":
    main()
