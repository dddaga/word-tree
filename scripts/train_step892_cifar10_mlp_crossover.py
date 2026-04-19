"""Step 892: CIFAR-10 MLP h-sweep to find SGNNET crossover param budget.

MOTIVATION
==========
step891 T2 showed:
  MLP_h1 (25K params) = 14.31%  (-71.83pp vs Linear)
  MLP_h2 (50K params) = 17.05%  (-69.09pp vs Linear)
  Ref_SGNNET (35K params) = ~80.69% (from step882 T2)

Both matched-params MLPs catastrophically fail due to N_in=25088 bottleneck.
Q: at what param count (h=?) does MLP finally match SGNNET's 80.69%?
This finds the param-efficiency ratio: SGNNET vs minimum-competitive MLP on CIFAR-10.

Expected: similar to Imagenette crossover (4.3× at 150K params, h≈6).
MLP_h6: 25088*6+6*10=150,588 params. Check if h=6 crosses 80.69%.

CONFIGS (T1: 75ep, 50% data, seed=42)  ← T1 not T0; early stop artifacts at 20ep
  MLP_h4   : h=4   → 100,368 params
  MLP_h6   : h=6   → 150,548 params
  MLP_h8   : h=8   → 200,728 params
  MLP_h16  : h=16  → 401,446 params
  MLP_h32  : h=32  → 802,882 params
  MLP_h64  : h=64  → 1,605,754 params
  Ref_SGNNET: 34,976 params (~80.69% from step882)

Target: find smallest h where MLP_h ≥ SGNNET 80.69%.
Paper claim: "SGNNET achieves CIFAR-10 accuracy with Nx fewer params than minimum-viable MLP."
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

from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10

STEP882_SGNNET = 0.8069  # SGNNET crossover target

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step892_cifar10_mlp_crossover_seed{SEED}__{SLOT}.json"


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.W_pos = self.fc1.weight
    @property
    def W_phase(self): return None
    def tick_epoch(self): pass
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))


HIDDEN_SIZES = [4, 6, 8, 16, 32, 64]


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=False)
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False,
    )

    print(f"\n{'='*70}")
    print(f"step892 — CIFAR-10 MLP h-sweep crossover with SGNNET (T1: 75ep, 50%)")
    print(f"  device={DEVICE}  seed={SEED}")
    print(f"  SGNNET target: {STEP882_SGNNET:.4f} (step882 T2 = step891 Ref_SGNNET)")
    print(f"  Q: smallest h where MLP_h >= SGNNET on CIFAR-10?")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    results = {}
    crossover_h = None

    for h in HIDDEN_SIZES:
        torch.manual_seed(SEED)
        model = MLP(N_IN, h, N_OUT)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = n_p / 34976
        print(f"{'─'*60}\nMLP_h{h}: params={n_p:,} ({ratio:.1f}× SGNNET)")

        kw = trainer_kwargs(N_IN, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 25 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h_.get("val_top1", 0.0) if isinstance(h_, dict) else float(h_) for h_ in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        gap = best - STEP882_SGNNET
        if crossover_h is None and best >= STEP882_SGNNET:
            crossover_h = h
        print(f"  -> best={best:.4f} @ep{best_ep}  vs_SGNNET={gap*100:+.2f}pp  {elapsed:.0f}s")

        results[f"MLP_h{h}"] = {
            "h": h, "n_params": n_p, "ratio_vs_sgnnet": round(ratio, 2),
            "best": best, "best_ep": best_ep,
            "gap_vs_sgnnet": round(gap, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 892 SUMMARY — CIFAR-10 MLP crossover with SGNNET (T1)")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'Params':>10}  {'Ratio':>6}  {'Best':>7}  {'vs SGNNET':>10}")
    for k, r in results.items():
        marker = " ← CROSSOVER" if r["h"] == crossover_h else ""
        print(f"  {k:<12} {r['n_params']:>10,}  {r['ratio_vs_sgnnet']:>5.1f}×  "
              f"{r['best']:>7.4f}  {r['gap_vs_sgnnet']*100:>+9.2f}pp{marker}")
    if crossover_h:
        ratio = results[f"MLP_h{crossover_h}"]["ratio_vs_sgnnet"]
        print(f"\n  Crossover at h={crossover_h}: MLP needs {ratio:.1f}× SGNNET's params to match")
        print(f"  Paper claim: SGNNET achieves CIFAR-10 accuracy at {ratio:.1f}× fewer params than minimum-viable MLP")
    else:
        print(f"\n  No crossover found at h≤{HIDDEN_SIZES[-1]} — extend sweep")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
