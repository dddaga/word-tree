"""Step 615: Fair matched-params MLP baseline — best practices at ~67K param budget.

MOTIVATION
==========
step401b comparison: SGNNET (67K params) = 95.52% vs MLP_h3 (75K params) = 70.62%.
User concern: MLP comparison was not fair — basic ReLU, no regularization.

This script gives the matched-param MLP every advantage:
  - LeakyReLU(0.1) — avoids dead neurons, better gradient flow
  - He (Kaiming) initialization — correct for ReLU-family activations
  - Label smoothing 0.1 — regularizes soft targets, improves generalization
  - Dropout 0.3 after hidden layer — prevents overfitting with tiny hidden
  - Weight decay 1e-3 (stronger than default) — correct for tiny hidden layer
  - 150 epochs full data — same budget as SGNNET T2
  - Cosine annealing LR schedule

Also tests: what is the best achievable accuracy with ~67K params in an MLP,
regardless of architecture? Tries wider+shallower, deeper+narrow, skip-connection.

CONFIGS (~67K params each, Imagenette VGG16 features N_in=25088, N_out=10)
  A_fair_h3     : 25088→3→10  LeakyReLU, He init, dropout, label smooth  (75K params)
  B_fair_h2     : 25088→2→10  same best practices                         (50K params)
  C_deeper_h2   : 25088→2→2→10 deeper narrow (two hidden layers of 2)     (50K params)
  D_skip_h3     : 25088→3→10 with residual skip (25088→10 linear bypass)  (75K+250K — control)
  E_linear      : 25088→10 direct linear                                   (250K params)

E_linear tests whether a 250K linear probe beats 75K nonlinear — rules out
that the bottleneck is nonlinearity, not parameter count.

To run:
    python -u scripts/train_step615_fair_mlp_matched.py --device mps --epochs 150
"""
# CUDA-5060ti-validated — pure MLP, no SGNNET routing; no GradScaler; device-agnostic
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.training.dataset import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="A_fair_h3,B_fair_h2,C_deeper_h2,E_linear")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step615_fair_mlp_matched_seed{SEED}__{SLOT}.json"

# SGNNET reference for comparison
SGNNET_REF = {"accuracy": 0.9552, "params": 67_744, "label": "SGNNET N=2048 D=16 K=5 (step199)"}


# ─────────────────────────────────────────────────────────────────────────────
# Models — all use best practices
# ─────────────────────────────────────────────────────────────────────────────

def _init_linear(layer):
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class FairMLP_h3(nn.Module):
    """25088 → 3 → 10, LeakyReLU, He init, dropout."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.fc1 = _init_linear(nn.Linear(N_IN, 3))
        self.fc2 = _init_linear(nn.Linear(3, N_OUT))
        self.drop = nn.Dropout(dropout)
        self.act  = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class FairMLP_h2(nn.Module):
    """25088 → 2 → 10, LeakyReLU, He init, dropout."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.fc1 = _init_linear(nn.Linear(N_IN, 2))
        self.fc2 = _init_linear(nn.Linear(2, N_OUT))
        self.drop = nn.Dropout(dropout)
        self.act  = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class DeeperMLP_h2(nn.Module):
    """25088 → 2 → 2 → 10, two hidden layers."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.fc1 = _init_linear(nn.Linear(N_IN, 2))
        self.fc2 = _init_linear(nn.Linear(2, 2))
        self.fc3 = _init_linear(nn.Linear(2, N_OUT))
        self.drop = nn.Dropout(dropout)
        self.act  = nn.LeakyReLU(0.1)
    def forward(self, x):
        h1 = self.drop(self.act(self.fc1(x)))
        h2 = self.drop(self.act(self.fc2(h1)))
        return self.fc3(h2)


class LinearProbe(nn.Module):
    """25088 → 10 direct linear (250K params) — upper bound for linear approaches."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_IN, N_OUT)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, x): return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(model, tr, va, epochs, label_smoothing=0.1, weight_decay=1e-3, lr=1e-3):
    model = model.to(DEVICE)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    crit  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best  = 0.0; best_ep = 0
    for epoch in range(epochs):
        model.train()
        for x, _, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, _, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item(); total += y.size(0)
        v = correct / max(total, 1)
        if v > best: best, best_ep = v, epoch + 1
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  ep{epoch+1:4d}  val={v:.4f}  best={best:.4f}", flush=True)
    return best, best_ep


def main():
    torch.manual_seed(SEED)
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)
    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED)
    print(f"Step 615 — Fair matched-params MLP (best practices, ~67K params)")
    print(f"  Reference: SGNNET = {SGNNET_REF['accuracy']*100:.2f}% @ {SGNNET_REF['params']:,} params")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}\n")

    MODEL_MAP = {
        "A_fair_h3":   (FairMLP_h3,    "h=3 LeakyReLU He-init dropout=0.3 label-smooth"),
        "B_fair_h2":   (FairMLP_h2,    "h=2 LeakyReLU He-init dropout=0.3 label-smooth"),
        "C_deeper_h2": (DeeperMLP_h2,  "h=2×2 deeper LeakyReLU He-init dropout=0.3"),
        "E_linear":    (LinearProbe,   "linear probe 25088→10 (250K, upper-bound)"),
    }
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    for key in keys:
        if key not in MODEL_MAP: print(f"  skip {key}"); continue
        cls, desc = MODEL_MAP[key]
        model = cls()
        n_p = sum(p.numel() for p in model.parameters())
        print(f"{'─'*60}\n{key}: {desc}\n  params={n_p:,}")
        t0 = time.time()
        best, best_ep = train(model, tr, va, EPOCHS)
        elapsed = time.time() - t0
        delta = best - SGNNET_REF["accuracy"]
        print(f"  → best={best:.4f} @ep{best_ep}  Δ_vs_SGNNET={delta*100:+.2f}pp  elapsed={elapsed:.0f}s")
        results[key] = {"label": desc, "n_params": n_p, "best": best,
                        "best_ep": best_ep, "delta_vs_sgnnet": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 615 SUMMARY — Fair matched-params MLP vs SGNNET {SGNNET_REF['accuracy']*100:.2f}%")
    print(f"{'='*60}")
    print(f"  {'Config':<16} {'params':>8}  {'best':>7}  {'Δ vs SGNNET':>12}")
    for k, r in results.items():
        print(f"  {k:<16} {r['n_params']:>8,}  {r['best']:>7.4f}  {r['delta_vs_sgnnet']*100:>+11.2f}pp")
    print(f"\n  SGNNET ref:      {SGNNET_REF['params']:>8,}  {SGNNET_REF['accuracy']:>7.4f}  {'—':>12}")


if __name__ == "__main__":
    main()
