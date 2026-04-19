"""Step 893: CIFAR-10 MLP crossover T2 confirmation — h=8,12,16 only.

MOTIVATION
==========
step892 T1 showed dramatic cliff:
  h=8  (200K, 5.7× SGNNET): 45.52%  ← catastrophic bottleneck
  h=16 (400K, 11.5× SGNNET): 81.29% ← crosses SGNNET 80.42%

Q: does the cliff hold at T2 (150ep, 100% data)?
   What is the exact h at crossover? Test h=8,12,16.

h=12 (300K): may be at/below crossover — worth testing.
  h=12: N_in→12→N_out: 25088*12+12*10 = 301,176 params (8.6× SGNNET)

SGNNET baseline: 80.42% (step891 T2, seed=42)
Linear baseline: 86.14% (step891 T2, seed=42)

CONFIGS (T2: 150ep, 100% data, seed=42)
  MLP_h8  : h=8  → 200,802 params (5.7× SGNNET)
  MLP_h12 : h=12 → 301,194 params (8.6× SGNNET)
  MLP_h16 : h=16 → 401,594 params (11.5× SGNNET)

Paper claim: "Crossover at h=X (Y× SGNNET params) — confirmed T2."
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
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10

STEP891_SGNNET = 0.8042  # step891 T2 canonical
STEP891_LINEAR = 0.8614

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step893_cifar10_mlp_crossover_t2_seed{SEED}__{SLOT}.json"


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


HIDDEN_SIZES = [8, 12, 16]


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step893 — CIFAR-10 MLP crossover T2 (150ep, 100%, h=8,12,16)")
    print(f"  device={DEVICE}  seed={SEED}")
    print(f"  SGNNET target: {STEP891_SGNNET:.4f} (step891 T2)")
    print(f"  Linear ref:    {STEP891_LINEAR:.4f} (step891 T2)")
    print(f"  step892 T1: h=8→45%, h=16→81.3% (cliff confirmed)")
    print(f"  Q: exact T2 crossover — is h=12 (300K) above or below SGNNET?")
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
        gap = best - STEP891_SGNNET
        if crossover_h is None and best >= STEP891_SGNNET:
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
    print(f"STEP 893 SUMMARY — CIFAR-10 MLP crossover T2")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'Params':>10}  {'Ratio':>6}  {'Best':>7}  {'vs SGNNET':>10}")
    for k, r in results.items():
        marker = " ← CROSSOVER" if r["h"] == crossover_h else ""
        print(f"  {k:<12} {r['n_params']:>10,}  {r['ratio_vs_sgnnet']:>5.1f}×  "
              f"{r['best']:>7.4f}  {r['gap_vs_sgnnet']*100:>+9.2f}pp{marker}")
    if crossover_h:
        ratio = results[f"MLP_h{crossover_h}"]["ratio_vs_sgnnet"]
        print(f"\n  T2 crossover at h={crossover_h}: {ratio:.1f}× SGNNET params required")
        print(f"  Paper: SGNNET achieves CIFAR-10 accuracy at {ratio:.1f}× fewer params (T2 CONFIRMED)")
    else:
        print(f"\n  No T2 crossover at h≤{HIDDEN_SIZES[-1]}; step892 h=16 T1 result may not hold T2")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
