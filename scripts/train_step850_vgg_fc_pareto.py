"""Step 850: VGG-FC Pareto baseline — accuracy vs FLOPs/params curve for dense FC heads.

MOTIVATION
==========
Paper needs: "what accuracy does a standard FC head achieve at SGNNET's compute budget?"
step403b covered MLP_37 (97.71% @ 1.86M FLOPs). Missing:
- Sub-SGNNET configs (MLP_16 @ 0.80M FLOPs) — does FC break below SGNNET budget?
- VGG-style 2-layer head at matched FLOPs — does depth help?
- Curve above SGNNET FLOPs (MLP_128, MLP_256, MLP_512) — FC saturation point?

Hypothesis: FC head accuracy collapses below SGNNET's 95.52% once FLOPs < ~6M.
SGNNET holds 95.52% at 1.85M FLOPs (step199) — if FC needs 6×+ more FLOPs to match,
that's the efficiency gap paper claims.

CONFIGS (FLOPs = 2 × MACs convention, consistent with step403b/step800)
========================================================================
  MLP_16    : 25088→16→10       0.804M FLOPs,   401K params  (sub-SGNNET — key)
  MLP_128   : 25088→128→10      6.426M FLOPs,   3.21M params
  MLP_256   : 25088→256→10      12.85M FLOPs,   6.42M params
  MLP_512   : 25088→512→10      25.70M FLOPs,   12.84M params
  VGG_37_2L : 25088→37→37→10   1.860M FLOPs,   0.93M params (matched SGNNET FLOPs, 2-layer)
  VGG_128_2L: 25088→128→128→10 6.458M FLOPs,   3.23M params (2-layer mid-range)

Reference (prior steps, not re-trained here):
  Linear    : step401  96.92%  ~0.50M FLOPs  (25088×10×2)
  MLP_37    : step403b 97.71%  1.86M FLOPs
  MLP_64    : step401  97.20%  3.21M FLOPs
  SGNNET    : step199  95.52%  1.85M FLOPs   ← efficiency champion reference
  SGNNET_K1 : step607  95.95%  0.20M FLOPs   ← K=1 distilled student

Tier: 2 equivalent (150ep, 100% data) — paper baseline quality.
Device: MPS or CPU (no SGNNET, no CUDA audit required).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.dataset import make_loaders

# --- CLI --------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=150)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--configs",  default="",
                    help="Comma-separated config keys to run. Empty = all.")
args = parser.parse_args()

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N_IN   = 25088
N_OUT  = 10

SLOT     = os.environ.get("SGN_SLOT", "local")
STEP_NAME = Path(__file__).stem
OUT_PATH  = ROOT / "results" / f"{STEP_NAME}__{SLOT}.json"


# --- Architectures ----------------------------------------------------------

class MLP1L(nn.Module):
    """Single hidden layer MLP: N_in → h → N_out."""
    def __init__(self, h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, h), nn.ReLU(),
            nn.Linear(h, N_OUT),
        )

    def forward(self, x):
        return self.net(x)


class MLP2L(nn.Module):
    """Two hidden layer MLP (VGG-style): N_in → h → h → N_out."""
    def __init__(self, h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, h), nn.ReLU(),
            nn.Linear(h, h),    nn.ReLU(),
            nn.Linear(h, N_OUT),
        )

    def forward(self, x):
        return self.net(x)


def count_flops_1l(h: int) -> int:
    """FLOPs = 2×MACs for N_in→h→N_out (each multiply-add = 2 FLOPs)."""
    return 2 * (N_IN * h + h * N_OUT)


def count_flops_2l(h: int) -> int:
    """FLOPs = 2×MACs for N_in→h→h→N_out."""
    return 2 * (N_IN * h + h * h + h * N_OUT)


def count_params_1l(h: int) -> int:
    return (N_IN + 1) * h + (h + 1) * N_OUT


def count_params_2l(h: int) -> int:
    return (N_IN + 1) * h + (h + 1) * h + (h + 1) * N_OUT


# --- Config definitions -----------------------------------------------------
# Each entry: (hidden_size, n_layers)
CONFIGS = {
    "MLP_16":     (16,  1),
    "MLP_128":    (128, 1),
    "MLP_256":    (256, 1),
    "MLP_512":    (512, 1),
    "VGG_37_2L":  (37,  2),
    "VGG_128_2L": (128, 2),
}


def build_model(h: int, n_layers: int) -> nn.Module:
    torch.manual_seed(SEED)
    if n_layers == 1:
        return MLP1L(h)
    return MLP2L(h)


def train_model(model: nn.Module, tr, va) -> list[dict]:
    model = model.to(DEVICE)
    opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    crit  = nn.CrossEntropyLoss()
    history = []

    for epoch in range(EPOCHS):
        model.train()
        for feats, _soft, labels in tr:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            crit(model(feats), labels).backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, _soft, labels in va:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                preds = model(feats).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_top1 = correct / max(total, 1)
        history.append({"epoch": epoch, "val_top1": val_top1})

        ep = epoch + 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={val_top1:.4f}", flush=True)

    return history


def main():
    run_keys = list(CONFIGS.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"{STEP_NAME} — VGG-FC Pareto baseline")
    print(f"Running: {run_keys}  on {DEVICE}")
    print(f"{'='*70}")

    tr, va = make_loaders(ROOT / args.data, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}
    for key in run_keys:
        h, n_layers = CONFIGS[key]
        flops  = count_flops_1l(h)  if n_layers == 1 else count_flops_2l(h)
        n_params = count_params_1l(h) if n_layers == 1 else count_params_2l(h)

        print(f"\n{'─'*60}")
        print(f"Config {key}: h={h} layers={n_layers}")
        print(f"  FLOPs={flops:,} ({flops/1e6:.2f}M)  params={n_params:,}")
        print(f"{'─'*60}")

        model = build_model(h, n_layers)
        t0 = time.time()
        history = train_model(model, tr, va)
        elapsed = time.time() - t0

        top1h = [round(e["val_top1"], 4) for e in history]
        best  = max(top1h)
        bep   = int(np.argmax(top1h)) + 1
        results[key] = {
            "top1_best":    best,
            "top1_last":    top1h[-1],
            "best_epoch":   bep,
            "top1_history": top1h,
            "elapsed_s":    round(elapsed, 1),
            "flops":        flops,
            "n_params":     n_params,
            "hidden":       h,
            "n_layers":     n_layers,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    # --- Summary with SGNNET reference numbers ---
    print(f"\n{'='*70}\n{STEP_NAME} SUMMARY\n{'='*70}")
    print(f"  {'Config':14s} {'FLOPs(M)':>10s} {'Params':>10s} {'Top1':>8s}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*8}")
    # Reference entries from prior steps
    ref_rows = [
        ("SGNNET_K1",  0.20,   34976,   0.9595),
        ("SGNNET",     1.85,   67744,   0.9552),
        ("Linear",     0.50,   250890,  0.9692),
        ("MLP_37",     1.86,   928673,  0.9771),
        ("MLP_64",     3.21,   1607050, 0.9720),
    ]
    for name, fm, np_, acc in ref_rows:
        print(f"  {name:14s} {fm:>10.2f} {np_:>10,} {acc:>8.4f}  [prior]")
    for key in run_keys:
        r = results[key]
        print(f"  {key:14s} {r['flops']/1e6:>10.2f} {r['n_params']:>10,} {r['top1_best']:>8.4f}  [NEW]")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
