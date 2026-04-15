"""Step 403b: Matched-FLOPs MLP baseline (h=37, ~1.86M FLOPs).

MOTIVATION
==========
step401 gives MLP_2 (50K p), MLP_3 (75K p, 52.41%), MLP_64 (1.6M p, 97.25%).
Reviewer will ask: "what about an MLP at the SAME FLOPs as SGNNET?"

SGNNET true FLOPs ≈ 1.85M (ncu-validated, step800).
MLP FLOPs = 2·h·(25088 + 10) = 50196·h.
  Setting = 1.85M → h = 36.85 → **h=37** (closest integer).

Params:  25088·37 + 37·10 = 928,256 + 370 + 37(bias) + 10 = 928,673 params
FLOPs:   2·37·(25088+10) ≈ 1,857,252 → 1.86M ≈ 1.85M match

Baseline name: MLP_37 — fair accuracy comparison at matched compute.

Expected: 55-65% (MLP_3 at 75K=52.41%, MLP_64 at 1.6M=97.25%, MLP_37 sits
between, likely in the 65-85% range given param count). If MLP_37 >= 95%,
SGNNET's accuracy-at-FLOPs claim is weaker than we thought. If MLP_37 ≤ 85%,
paper claim "SGNNET beats matched-FLOPs MLP" is validated.

CONFIG: Single MLP_37 at Tier-2 (150ep, full data), same protocol as step401.
Uses same train_mlp loop (plain Adam + CosineAnnealingLR, no SGNNET Trainer).

To run:
    python -u scripts/train_step403b_matched_flops_mlp.py
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

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--hidden", type=int, default=37,
                    help="Hidden size. Default 37 matches SGNNET 1.85M FLOPs.")
parser.add_argument("--data",   default="data/store.h5")
parser.add_argument("--full_data", action="store_true", default=True)
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10; H = args.hidden

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step403b_matched_flops_mlp_h{H}_seed{SEED}__{SLOT}.json"


class MLPBaseline(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_OUT),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(model, tr, va, epochs: int):
    model = model.to(DEVICE)
    opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    crit  = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)

    return history


def main():
    h = H
    flops = 2 * h * (N_IN + N_OUT)
    n_params = (N_IN + 1) * h + (h + 1) * N_OUT

    print(f"Step 403b — Matched-FLOPs MLP (h={h})")
    print(f"  FLOPs={flops:,} ({flops/1e6:.2f}M)  — target 1.85M")
    print(f"  params={n_params:,}")
    print(f"  data={args.data}  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")

    tr, va = make_loaders(ROOT / args.data, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    torch.manual_seed(SEED)
    model = MLPBaseline(hidden=h)

    t0 = time.time()
    history = train_mlp(model, tr, va, epochs=EPOCHS)
    elapsed = time.time() - t0

    top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
    best  = max(top1h)
    bep   = int(np.argmax(top1h)) + 1

    result = {
        "hidden":        h,
        "n_params":      n_params,
        "flops":         flops,
        "top1_best":     best,
        "top1_last":     top1h[-1],
        "best_epoch":    bep,
        "top1_history":  top1h,
        "elapsed_s":     round(elapsed, 1),
        "epochs":        EPOCHS,
        "seed":          SEED,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print(f"\n========== STEP 403b SUMMARY ==========")
    print(f"  MLP_{h}  params={n_params:,}  flops={flops/1e6:.2f}M  best={best:.4f} @ep{bep}")
    print(f"  Compare: SGNNET_Ref 95.52% at 1.85M FLOPs, MLP_3 52.41% at 75K params")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
