"""Routing dropout ablation.

Drops entire Z-vectors (not individual dims) during the routing loop with
probability p. Surviving neurons keep their l2-normalised directions; the
end-of-step l2-normalise handles rescaling automatically.

Config:  dynamic_z_geo + thresh=0.3, N=512, D=4, store_aug.h5, 120ep, MPS.
Same as aug baseline — only routing_dropout_p varies.

Sweep:
  p=0.0  — baseline (no dropout, should match aug baseline result)
  p=0.1  — 10% of neurons dropped per routing step
  p=0.2  — 20% of neurons dropped per routing step
  p=0.3  — 30% of neurons dropped per routing step

If p=0.0 here diverges from train_aug_baseline.json by >0.5%, something is
wrong — re-check the seed. Use p=0.0 as internal reference, not aug baseline.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py, numpy as np
import torch, torch.nn as nn, torch.utils.data

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
from src.training.trainer        import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--data",   default="data/store_aug.h5")
parser.add_argument("--sched",  default="plateau", choices=["plateau", "cosine"])
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Sched: {args.sched}")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    with h5py.File(args.data, "r") as f:
        return (
            torch.from_numpy(f["train/features"][:]),
            torch.from_numpy(f["train/soft_labels"][:]),
            torch.from_numpy(f["train/labels"][:]).long(),
            torch.from_numpy(f["val/features"][:]),
            torch.from_numpy(f["val/soft_labels"][:]),
            torch.from_numpy(f["val/labels"][:]).long(),
        )

def make_loaders(data):
    tf, tsl, tl, vf, vsl, vl = data
    g = torch.Generator().manual_seed(SEED)
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tf, tsl, tl),
        batch_size=BATCH, shuffle=True, generator=g, num_workers=0,
    )
    va = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(vf, vsl, vl),
        batch_size=BATCH, num_workers=0,
    )
    return tr, va


# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(dropout_p: float) -> nn.Module:
    torch.manual_seed(SEED)
    tk = topology_kwargs(512)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=512, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=4,
        encoding_mode="linear",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo",
        resonance_threshold=0.3,
        geo_gamma=1.0,
        routing_dropout_p=dropout_p,
    )


# ── Run ───────────────────────────────────────────────────────────────────────

def run(label: str, dropout_p: float, data) -> dict:
    print(f"\n{'='*62}\n{label}\n{'='*62}")
    tr_loader, va_loader = make_loaders(data)
    model = make_model(dropout_p).to(DEVICE)
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type=args.sched)
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best   = max(h.get("val_top1", 0.0) for h in history)
    last5  = history[-5:]
    result = {
        "label":           label,
        "dropout_p":       dropout_p,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "epochs_run":      len(history),
        "elapsed_s":       round(elapsed, 1),
        "lr_history":      [h["lr"] for h in history],
        "top1_history":    [round(h.get("val_top1", 0.0), 4) for h in history],
    }
    print(f"  top1_best={best:.4f}  top1_last={result['top1_last']:.4f}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

DROPOUT_SWEEP = [0.0, 0.1, 0.2, 0.3]

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded: train={len(data[0])}  val={len(data[3])}")

    results = {}
    for p in DROPOUT_SWEEP:
        key = f"p{int(p*10):02d}"
        results[key] = run(
            f"routing_dropout p={p:.1f}  dynamic_z_geo thresh=0.3  N=512 D=4  aug  120ep",
            dropout_p=p, data=data,
        )

    out = Path("results/train_routing_dropout.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    base_top1 = results["p00"]["top1_best"]
    print("\n── Routing dropout ablation ─────────────────────────────────────")
    print(f"  {'Config':<55}  {'top1_best':>9}  {'Δ vs p=0':>8}  {'t(s)':>6}")
    print(f"  {'-'*55}  {'-'*9}  {'-'*8}  {'-'*6}")
    for key, r in results.items():
        delta = r["top1_best"] - base_top1
        print(f"  {key:<55}  {r['top1_best']:>9.4f}  {delta:>+8.4f}  {r['elapsed_s']:>6.0f}")

    best_key = max(results, key=lambda k: results[k]["top1_best"])
    best_p   = results[best_key]["dropout_p"]
    print(f"\n  Best: p={best_p:.1f}  ({results[best_key]['top1_best']:.4f})")
    if best_p == 0.0:
        print("  → No dropout benefit. Keep routing_dropout_p=0.0.")
    else:
        print(f"  → Dropout helps. Use routing_dropout_p={best_p:.1f} in subsequent ablations.")
