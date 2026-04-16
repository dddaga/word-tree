"""Step 403c: CIFAR-10 MLP baselines for cross-dataset comparison.

MOTIVATION
==========
step401b: SGNNET K=4 achieves 80.59% on CIFAR-10 (VGG16 features, 150ep).
This script establishes MLP baselines at multiple param budgets so we can
characterize SGNNET's parameter efficiency on CIFAR-10 (vs Imagenette).

On Imagenette: SGNNET 67K = MLP_h3 (75K) ≈ MLP_h2 (50K), outperforms linear.
Question: does SGNNET maintain same relative advantage on CIFAR-10?

CONFIGS
=======
  Linear    : 25088→10 (no hidden layer, 250K params)
  MLP_h1    : 25088→1→10 (trivial, ~25K)
  MLP_h3    : 25088→3→10 (~75K, matched to SGNNET param budget)
  MLP_h37   : 25088→37→10 (~930K, matched FLOPs to step403b Imagenette ref)
  MLP_h256  : 25088→256→10 (~6.4M, large baseline)

Scale: VGG16 CIFAR-10 features (25088-dim), 10 classes
Tier: T2 (150ep, 100% data)

To run:
    python -u scripts/train_step403c_cifar10_mlp_baselines.py --device cpu --epochs 150
    python -u scripts/train_step403c_cifar10_mlp_baselines.py --device mps --epochs 150
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.training.dataset import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
parser.add_argument("--configs", default="Linear,MLP_h1,MLP_h3,MLP_h37,MLP_h256")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step403c_cifar10_mlp_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Linear":   ([], "Linear (no hidden, 250K params)"),
    "MLP_h1":   ([1],  "MLP h=1 (~25K params, trivial bottleneck)"),
    "MLP_h3":   ([3],  "MLP h=3 (~75K params, matched SGNNET budget)"),
    "MLP_h37":  ([37], "MLP h=37 (~930K params, matched step403b FLOPs)"),
    "MLP_h256": ([256], "MLP h=256 (~6.4M params, large baseline)"),
}


class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        prev = N_IN
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, N_OUT))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(hidden_sizes):
    prev = N_IN; total = 0
    for h in hidden_sizes:
        total += prev * h + h
        prev = h
    total += prev * N_OUT + N_OUT
    return total


def train_model(model, tr, va):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    ce = nn.CrossEntropyLoss()
    best = 0.0; best_ep = 0; history = []

    for epoch in range(EPOCHS):
        model.train()
        for xb, _, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = ce(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, _, yb in va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(-1) == yb).sum().item()
                total += yb.size(0)
        v = correct / total
        history.append(v)
        if v > best: best, best_ep = v, epoch + 1
        if (epoch + 1) % 30 == 0:
            print(f"  ep{epoch+1:3d}  val={v:.4f}", flush=True)

    return best, best_ep


def main():
    torch.manual_seed(SEED)
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED)
    print(f"Step 403c — CIFAR-10 MLP baselines (T2)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  train={len(tr.dataset)}  val={len(va.dataset)}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    # SGNNET reference from step401b
    SGNNET_REF = 0.8059

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        hidden, desc = CONFIGS[key]
        n_p = count_params(hidden)
        model = MLP(hidden)
        print(f"{'─'*50}\n{key}: {desc}  params={n_p:,}")
        t0 = time.time()
        best, best_ep = train_model(model, tr, va)
        elapsed = time.time() - t0
        delta_sgnnet = best - SGNNET_REF
        print(f"  → best={best:.4f} @ep{best_ep}  Δ_SGNNET={delta_sgnnet*100:+.2f}pp  {elapsed:.0f}s")
        results[key] = {"label": desc, "hidden": hidden, "n_params": n_p,
                        "best": best, "best_ep": best_ep,
                        "delta_vs_sgnnet": round(delta_sgnnet, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*50}")
    print(f"STEP 403c SUMMARY — CIFAR-10 MLP baselines")
    print(f"  SGNNET K=4 reference: {SGNNET_REF:.4f} (step401b)")
    for k, r in results.items():
        print(f"  {k:<12} params={r['n_params']:>8,}  best={r['best']:.4f}  Δ_SGNNET={r['delta_vs_sgnnet']*100:+.2f}pp")


if __name__ == "__main__":
    main()
