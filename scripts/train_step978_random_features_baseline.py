"""Step 978: Random features + linear baseline — isolate routing contribution.

MOTIVATION
==========
Claim 3 states "learning happens in routing dynamics, not weights." Supporting
evidence so far: SGNNET_RandProj (all params frozen random) = 10.04% ≈ chance.
But that conflates frozen W_pos with the random *projection topology itself*.

This experiment isolates three distinct questions:
  Q1: How much does the full VGG16 feature vector encode? (Lin_VGG: upper bound)
  Q2: How much do random sparse projections of VGG16 features encode with no
      routing — just project + mean-pool → linear? (RandProj_meanpool)
  Q3: How much does SGNNET add via iterative routing over those same random
      projections? (SGNNET_Ref: reported from step199, not rerun)

If RandProj_meanpool << SGNNET_Ref → routing adds the value, Claim 3 supported.
If RandProj_meanpool ≈ SGNNET_Ref → projections suffice, routing claim needs revision.

CONFIGS (50% data, 20ep, fast-confirm)
=======================================
  Lin_VGG           : Linear(25088, 10) — full VGG16 features → linear
  RandProj_N2048_D16 : K_in=25 random projections (N=2048, D=16) → mean-pool
                       → 16-dim → Linear(16, 10). No routing, no learning in
                       the projection layer (fixed random weights).
  RandProj_N2048_D64 : Same with D=64 (fatter per-neuron projection)
  RandProj_N256_D16  : Fewer neurons (N=256) — less ensemble averaging
  RandProj_concat    : K_in=25, N=256, D=16 — concat all N×D before linear
                       (upper bound on what projections alone can do, no pooling)

REFERENCE (not rerun — from step199/step401):
  Lin_VGG (step401):     97.12%  — full features, linear
  SGNNET step199 T2:     97.30%  — routing on same sparse projections

OUTPUT: results/train_step978_random_features_seed42__<host>_<device>.json
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

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="")
parser.add_argument("--batch", type=int, default=256)
args = parser.parse_args()

if args.device == "auto":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS = args.epochs
BATCH  = args.batch
SEED   = 42
DATA   = ROOT / "data" / "store.h5"

N_IN  = 25088  # VGG16 pool5 flat dim
N_OUT = 10     # Imagenette classes

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Host detection ────────────────────────────────────────────────────────────
import socket
HOST = socket.gethostname().split(".")[0].replace("-", "_")
DEV_TAG = str(DEVICE).replace(":", "")
OUT_PATH = ROOT / "results" / f"train_step978_random_features_seed{SEED}__{HOST}_{DEV_TAG}.json"


# ── Random projection model ───────────────────────────────────────────────────
class RandProjLinear(nn.Module):
    """
    Random sparse projection + mean-pool + linear classifier.

    For each of N neurons: select K_in indices from N_IN uniformly at random,
    project to D dims via a fixed random Gaussian matrix. Mean-pool over N
    neurons → D-dim representation → linear classifier.

    Projection weights are FIXED (not learned). Only the linear head is trained.
    """
    def __init__(self, n_in: int, n_out: int, n_neurons: int, k_in: int, d: int, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)

        # Random connectivity: N × K_in indices into [0, n_in)
        conn = np.stack([
            rng.choice(n_in, size=k_in, replace=False)
            for _ in range(n_neurons)
        ])  # (N, K_in)
        self.register_buffer("conn", torch.from_numpy(conn).long())

        # Random projection weights: N × D × K_in  (fixed, not learned)
        W = (rng.standard_normal((n_neurons, d, k_in)) / np.sqrt(k_in)).astype(np.float32)
        self.register_buffer("W_proj", torch.from_numpy(W))

        self.n_neurons = n_neurons
        self.d = d
        self.fc = nn.Linear(d, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_IN)
        B = x.size(0)
        # Gather inputs: (B, N, K_in)
        gathered = x[:, self.conn]       # (B, N, K_in)
        # Project: (B, N, D) via einsum
        z = torch.einsum("bnk,ndk->bnd", gathered, self.W_proj)   # (B, N, D)
        z = torch.nn.functional.normalize(z, dim=-1)
        # Mean pool over N neurons → (B, D)
        pooled = z.mean(dim=1)
        return self.fc(pooled)


class ConcatRandProj(nn.Module):
    """
    Like RandProjLinear but concatenates all N×D features before linear.
    Upper bound on what fixed projections can do without routing.
    """
    def __init__(self, n_in: int, n_out: int, n_neurons: int, k_in: int, d: int, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)
        conn = np.stack([
            rng.choice(n_in, size=k_in, replace=False)
            for _ in range(n_neurons)
        ])
        self.register_buffer("conn", torch.from_numpy(conn).long())
        W = (rng.standard_normal((n_neurons, d, k_in)) / np.sqrt(k_in)).astype(np.float32)
        self.register_buffer("W_proj", torch.from_numpy(W))
        self.n_neurons = n_neurons
        self.d = d
        self.fc = nn.Linear(n_neurons * d, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        gathered = x[:, self.conn]
        z = torch.einsum("bnk,ndk->bnd", gathered, self.W_proj)
        z = torch.nn.functional.normalize(z, dim=-1)
        flat = z.reshape(B, -1)
        return self.fc(flat)


# ── Training helper ───────────────────────────────────────────────────────────
def run_config(model: nn.Module, tr_loader, va_loader, tag: str) -> dict:
    model = model.to(DEVICE)
    # Freeze all params except fc (projection is not a param — it's a buffer)
    # For linear baseline, everything is trainable
    opt = Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_acc = 0.0
    best_ep  = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            xb, _, yb = batch  # (features, soft_labels, hard_labels)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in va_loader:
                xb, _, yb = batch
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total   += yb.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_ep  = ep
        print(f"  [{tag}] ep{ep:3d}/{EPOCHS} val={acc:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"  [{tag}] DONE — best={best_acc:.4f} @ ep{best_ep}, params={n_params}, {elapsed:.0f}s")
    return {"best": best_acc, "best_ep": best_ep, "n_params": n_params, "elapsed_s": round(elapsed)}


# ── Config registry ───────────────────────────────────────────────────────────
ALL_CONFIGS = {
    "Lin_VGG": lambda: nn.Linear(N_IN, N_OUT),
    "RandProj_N2048_D16": lambda: RandProjLinear(N_IN, N_OUT, n_neurons=2048, k_in=25, d=16, seed=SEED),
    "RandProj_N2048_D64": lambda: RandProjLinear(N_IN, N_OUT, n_neurons=2048, k_in=25, d=64, seed=SEED),
    "RandProj_N256_D16":  lambda: RandProjLinear(N_IN, N_OUT, n_neurons=256,  k_in=25, d=16, seed=SEED),
    "RandProj_concat":    lambda: ConcatRandProj(N_IN, N_OUT, n_neurons=256,  k_in=25, d=16, seed=SEED),
}

if args.configs:
    wanted = set(args.configs.split(","))
    CONFIG_LIST = [(k, v) for k, v in ALL_CONFIGS.items() if k in wanted]
else:
    CONFIG_LIST = list(ALL_CONFIGS.items())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    # 50% subsample (T0 budget)
    n_tr = len(tr_full.dataset)
    sub_idx = torch.randperm(n_tr, generator=torch.Generator().manual_seed(SEED))[:n_tr // 2]
    tr_sub = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True,
    )

    results = {}
    for tag, model_fn in CONFIG_LIST:
        print(f"\n=== {tag} ===")
        model = model_fn()
        results[tag] = run_config(model, tr_sub, va, tag)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {OUT_PATH}")

    # Print summary table
    print("\n--- Summary ---")
    print(f"{'Config':<28} {'best':>7} {'params':>10}")
    for k, v in results.items():
        print(f"{k:<28} {v['best']:.4f}  {v['n_params']:>10,}")
    print(f"\nREF (step401): Lin_VGG(T0)=97.12%, SGNNET step199 T2=97.30%")


if __name__ == "__main__":
    main()
