"""Step 610: Low-rank MLP sweep — does VGG feature space have rank ≤16?

MOTIVATION
==========
MLP_37 (Linear 25088→37→10) hits 97.71% @ 929K params. If r=16 matches that
accuracy, the feature space has intrinsic rank ≤16 for this task — meaning MLP_37
is doing an accidental rank-16 projection and any compression to r<37 is "free."
Highest-EV experiment from the FC-fission meditation (2026-04-15).
NeurIPS 2024 (arXiv:2406.16450) shows LowRank consistently beats BlockDiag at
matched param budgets.

CONFIGS (N_in=25088, N_out=10, VGG16 features, T0 20ep 50% data)
  LR_pure_r8  : Linear(25088,8)→Linear(8,10)  — no nonlinearity, 200K params (22%)
  LR_pure_r16 : Linear(25088,16)→Linear(16,10) — 401K params (43%)
  LR_pure_r32 : Linear(25088,32)→Linear(32,10) — 803K params (86%)
  LR_relu_r8  : Linear(25088,8)→ReLU→Linear(8,10)
  LR_relu_r16 : Linear(25088,16)→ReLU→Linear(16,10)
  LR_relu_r32 : Linear(25088,32)→ReLU→Linear(32,10)
  MLP_37_ref  : Linear(25088,37)→ReLU→Linear(37,10) — 929K params (100%), reference

Acceptance:
  STRONG : LR_relu_r16 ≥ 97.71% at 43% params — low-rank compression is free
  MEDIUM : LR_relu_r16 within 1pp of MLP_37 — viable Pareto point
  KILL   : r=32 (86% params) still >2pp below MLP_37 — feature space is high-rank

Paper implication if STRONG: MLP_37 accidentally finds a rank-16 projection;
SGNNET on S^{D-1} with D=16 is doing the same thing geometrically.
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="LR_pure_r8,LR_pure_r16,LR_pure_r32,"
                                         "LR_relu_r8,LR_relu_r16,LR_relu_r32,MLP_37_ref")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step610_lowrank_mlp_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def build_model(key: str) -> nn.Module:
    torch.manual_seed(SEED)
    if key == "MLP_37_ref":
        return nn.Sequential(nn.Linear(N_IN, 37), nn.ReLU(), nn.Linear(37, N_OUT))
    parts = key.split("_")            # e.g. ["LR", "pure", "r16"]
    nonlin = parts[1]                 # "pure" or "relu"
    r = int(parts[2][1:])             # strip leading "r"
    layers: list[nn.Module] = [nn.Linear(N_IN, r)]
    if nonlin == "relu":
        layers.append(nn.ReLU())
    layers.append(nn.Linear(r, N_OUT))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one(model: nn.Module, tr, va) -> list[dict]:
    model = model.to(DEVICE)
    opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    crit  = nn.CrossEntropyLoss()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        for x, _s, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, _s, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item()
                total   += y.numel()
        v = correct / max(total, 1)
        history.append({"epoch": epoch, "val_top1": v})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={v:.4f}", flush=True)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    tr_full, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=g)[:n // 2]
    tr  = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0)

    print(f"\nStep 610 — Low-rank MLP sweep (T0 {EPOCHS}ep 50% data)")
    print(f"  device={DEVICE}  seed={SEED}  data={args.data}")
    print(f"  Hypothesis: VGG features have rank ≤16 for 10-class task")
    print(f"  Reference: MLP_37 = 97.71% @ 929K params")

    results: dict = {}
    for key in keys:
        print(f"\n{'─'*60}\nConfig: {key}\n{'─'*60}")
        model = build_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # FLOPs: 2 × (N_in × r + r × N_out) for two linear layers
        if key == "MLP_37_ref":
            flops = 2 * (N_IN * 37 + 37 * N_OUT)
        else:
            r = int(key.split("_")[2][1:])
            flops = 2 * (N_IN * r + r * N_OUT)
        print(f"  params={n_p:,}  flops≈{flops/1e6:.2f}M  ({100*n_p/929_157:.0f}% of MLP_37)")

        t0 = time.time()
        hist = train_one(model, tr, va)
        elapsed = time.time() - t0

        top1h = [round(h["val_top1"], 4) for h in hist]
        best, bep = max(top1h), int(np.argmax(top1h)) + 1
        print(f"  → best={best:.4f} @ep{bep}  ({elapsed:.0f}s)")

        results[key] = {
            "top1_best": best, "best_epoch": bep, "top1_last": top1h[-1],
            "n_params": n_p, "flops": flops, "elapsed_s": round(elapsed, 1),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary + verdict
    ref = results.get("MLP_37_ref", {}).get("top1_best", 0.0)
    print(f"\n{'='*70}\nSTEP 610 SUMMARY  (ref MLP_37={ref:.4f})\n{'='*70}")
    print(f"  {'Config':<18} {'params':>10} {'FLOPs':>8} {'top1':>8} {'Δ vs ref':>10} {'% params':>10}")
    mlp37_params = 929_157
    for k, r in results.items():
        delta = f"{r['top1_best']-ref:+.4f}" if k != "MLP_37_ref" else "—"
        pct   = f"{100*r['n_params']/mlp37_params:.0f}%"
        print(f"  {k:<18} {r['n_params']:>10,} {r['flops']/1e6:>7.2f}M {r['top1_best']:>8.4f} {delta:>10} {pct:>10}")

    # Verdict
    lr16 = results.get("LR_relu_r16", {}).get("top1_best")
    lr32 = results.get("LR_relu_r32", {}).get("top1_best")
    if lr16 is not None and ref > 0:
        d16 = lr16 - ref
        print(f"\nVerdict: LR_relu_r16 Δ = {d16:+.4f}")
        if d16 >= 0:
            print("  → STRONG: rank-16 compression is FREE. Feature space rank ≤ 16.")
        elif d16 >= -0.01:
            print("  → MEDIUM: within 1pp at 43% params — viable Pareto point.")
        elif lr32 is not None and (lr32 - ref) < -0.02:
            print("  → KILL: even r=32 (86% params) >2pp below MLP_37. Feature space is high-rank.")
        else:
            print("  → EXPLORE: r=16 suboptimal; check r=32 for compression cliff.")


if __name__ == "__main__":
    main()
