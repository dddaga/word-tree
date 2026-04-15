"""Step 401c: Linear + MLP baselines on CIFAR-10 VGG16 features.

PURPOSE
=======
step401b SGNNET on CIFAR-10: K=5=80.44%, K=4=80.11%. Need baselines on SAME features
to interpret this result. Matches step401 Imagenette baselines methodology.

CONFIGS (100ep, 100% data, pure PyTorch — no SGNNET Trainer):
  Lin_direct : Linear(25088, 10)       — 250,890 params
  MLP_2      : 25088 → 2 → 10          — ~50K (matched-params to SGNNET)
  MLP_3      : 25088 → 3 → 10          — ~75K (matched-params to SGNNET=67K)
  MLP_64     : 25088 → 64 → 10         — ~1.6M (matched-FLOPs region)

Target interpretation:
  - If SGNNET ≫ MLP_3 (at matched params): cross-dataset generalization of parameter-efficiency claim
  - If SGNNET ≈ MLP_64 (at much fewer params): headroom for FLOPs efficiency
  - If Linear ≫ SGNNET: CIFAR-10 is too easy for VGG16 features, test is invalid
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.dataset import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--data", default="data/store_cifar10.h5")
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42
N_IN = 25088; N_OUT = 10

OUT_PATH = ROOT / "results" / "train_step401c_cifar10_baselines.json"


class LinearProbe(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(N_IN, N_OUT)
    def forward(self, x): return self.fc(x)


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, h), nn.ReLU(inplace=True),
            nn.Linear(h, N_OUT),
        )
    def forward(self, x): return self.net(x)


BASELINES = {
    "Lin_direct": lambda: LinearProbe(),
    "MLP_2":      lambda: MLP(h=2),
    "MLP_3":      lambda: MLP(h=3),
    "MLP_64":     lambda: MLP(h=64),
}


def train_model(model, tr, va, epochs, device):
    model.to(device)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0; hist = []
    for ep in range(epochs):
        model.train()
        for batch in tr:
            x = batch[0].to(device)
            y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        # validate
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for batch in va:
                x = batch[0].to(device)
                y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        hist.append(acc); best = max(best, acc)
        if (ep + 1) % 10 == 0:
            print(f"  ep{ep+1:3d}  val={acc:.4f}  best={best:.4f}", flush=True)
    return best, hist


def main():
    data_path = ROOT / args.data
    keys = list(BASELINES.keys())
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else keys

    print(f"Step 401c — CIFAR-10 baselines (Linear + MLPs)")
    print(f"  data={args.data}  epochs={EPOCHS}  device={DEVICE}")
    tr, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}
    for key in run_keys:
        if key not in BASELINES: continue
        print(f"\n{'─'*60}\nConfig {key}\n{'─'*60}")
        t0 = time.time()
        model = BASELINES[key]()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}")
        best, hist = train_model(model, tr, va, EPOCHS, DEVICE)
        elapsed = time.time() - t0
        print(f"  → best={best:.4f}  elapsed={elapsed:.0f}s")
        results[key] = {"n_params": n_params, "top1_best": best,
                        "history": hist, "elapsed_s": elapsed}
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n========== STEP 401c SUMMARY (CIFAR-10) ==========")
    for k, r in results.items():
        print(f"  {k:<12}  params={r['n_params']:>10,}  best={r['top1_best']:.4f}")
    print(f"\n(Compare vs SGNNET: step401b K=5=80.44%, K=4=80.11% @ 67,744 params)")


if __name__ == "__main__":
    main()
