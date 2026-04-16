"""Step 851: MLP param-threshold crossover — find minimum h where MLP first beats SGNNET.

MOTIVATION
==========
step850 found: MLP_16 (401K params, 0.80M FLOPs) = 97.50% > SGNNET (67K params, 1.85M FLOPs) = 95.52%.
step401 found: MLP_3 (75K params) = 45.91%, MLP_2 (50K params) = 46.98%.

The crossover must be between h=3 and h=16. This experiment finds the exact threshold:
what is the MINIMUM number of hidden units for a plain 1L MLP to match/beat SGNNET?

This is paper-critical: it sharpens the params-efficiency claim.
"SGNNET achieves VGG-FC-level accuracy at X params. A plain MLP needs Y×X params to match."

CONFIGS
=======
  h=4  : 25088→4→10   ~200K params
  h=5  : 25088→5→10   ~250K params
  h=6  : 25088→6→10   ~301K params
  h=7  : 25088→7→10   ~351K params
  h=8  : 25088→8→10   ~401K... wait, h=8→201K params
  h=10 : ~251K params
  h=12 : ~301K params
  h=14 : ~351K params

FLOPs and params (2×MACs convention):
  h=4:  FLOPs=200,784  params=100,402  (25088×4 + 4×10 + biases)
  h=6:  FLOPs=301,176  params=150,603
  h=8:  FLOPs=401,568  params=200,804
  h=10: FLOPs=501,960  params=251,005
  h=12: FLOPs=602,352  params=301,206

Tier: T2 (150ep, 100% data) — paper quality.
Device: any (networks are tiny, fast on all devices).
"""
# CUDA-5060ti-validated — pure MLP, no SGNNET routing; no GradScaler; pin_memory/non_blocking
# handled internally by make_loaders. CUDA audit inapplicable to MLP-only scripts.
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
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="",
                    help="Comma-separated hidden sizes, e.g. '4,6,8,10,12'. Empty = all.")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N_IN   = 25088
N_OUT  = 10

SLOT      = os.environ.get("SGN_SLOT", "local")
STEP_NAME = Path(__file__).stem
OUT_PATH  = ROOT / "results" / f"{STEP_NAME}__{SLOT}.json"

# Crossover targets
SGNNET_ACC   = 0.9552   # step199 T2
SGNNET_PARAMS = 67744   # step199

# All hidden sizes to probe (default)
ALL_H = [4, 6, 8, 10, 12, 14]


class MLP1L(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, h), nn.ReLU(),
            nn.Linear(h, N_OUT),
        )

    def forward(self, x):
        return self.net(x)


def count_flops(h: int) -> int:
    return 2 * (N_IN * h + h * N_OUT)


def count_params(h: int) -> int:
    return (N_IN + 1) * h + (h + 1) * N_OUT


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
        history.append({"epoch": epoch, "val_top1": correct / max(total, 1)})

        ep = epoch + 1
        if ep % 25 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={history[-1]['val_top1']:.4f}", flush=True)

    return history


def main():
    if args.configs:
        h_list = [int(x.strip()) for x in args.configs.split(",")]
    else:
        h_list = ALL_H

    print(f"\n{'='*70}")
    print(f"{STEP_NAME} — MLP param crossover probe")
    print(f"Testing h={h_list}  SGNNET target: {SGNNET_ACC:.4f} @ {SGNNET_PARAMS:,} params")
    print(f"device={DEVICE}  epochs={EPOCHS}")
    print(f"{'='*70}")

    tr, va = make_loaders(ROOT / args.data, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}
    if OUT_PATH.exists():
        try: results = json.loads(OUT_PATH.read_text())
        except Exception: pass

    for h in h_list:
        key = f"MLP_{h}"
        if key in results:
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f})")
            continue

        flops    = count_flops(h)
        n_params = count_params(h)
        beats    = "?"

        print(f"\n{'─'*60}")
        print(f"{key}: h={h}  FLOPs={flops/1e6:.3f}M  params={n_params:,}  "
              f"({'%.1f' % (n_params/SGNNET_PARAMS)}× SGNNET params)")

        torch.manual_seed(SEED)
        model = MLP1L(h)
        t0 = time.time()
        history = train_model(model, tr, va)
        elapsed = time.time() - t0

        top1h = [e["val_top1"] for e in history]
        best  = max(top1h)
        bep   = int(np.argmax(top1h)) + 1
        beats = best >= SGNNET_ACC

        results[key] = {
            "h":         h,
            "top1_best": round(best, 6),
            "top1_last": round(top1h[-1], 6),
            "best_epoch": bep,
            "flops":     flops,
            "n_params":  n_params,
            "elapsed_s": round(elapsed, 1),
            "beats_sgnnet": beats,
            "delta_vs_sgnnet": round(best - SGNNET_ACC, 4),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

        status = "BEATS SGNNET" if beats else f"trails SGNNET by {(SGNNET_ACC-best)*100:.2f}pp"
        print(f"  → best={best:.4f} @ep{bep}  {status}  ({elapsed:.0f}s)")

    # Summary
    print(f"\n{'='*70}\n{STEP_NAME} SUMMARY\n{'='*70}")
    print(f"  SGNNET ref: {SGNNET_ACC:.4f} @ {SGNNET_PARAMS:,} params")
    print(f"  {'h':>4} {'params':>10} {'FLOPs(M)':>10} {'top1':>8} {'vs SGNNET':>12} {'beats?':>7}")
    for key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        r = results[key]
        print(f"  {r['h']:>4} {r['n_params']:>10,} {r['flops']/1e6:>10.3f} "
              f"{r['top1_best']:>8.4f} {r['delta_vs_sgnnet']*100:>+11.2f}pp "
              f"{'✓' if r['beats_sgnnet'] else '✗':>7}")
    crossover = [r['h'] for r in results.values() if r['beats_sgnnet']]
    if crossover:
        min_h = min(crossover)
        r = results[f"MLP_{min_h}"]
        print(f"\n  Crossover at h={min_h}: {r['n_params']:,} params "
              f"({r['n_params']/SGNNET_PARAMS:.1f}× SGNNET) — {r['top1_best']:.4f}%")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
