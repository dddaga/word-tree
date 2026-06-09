"""Step 99: W_phase trained at N=4096 — scaling gap fix.

MOTIVATION
==========
step76 showed W_phase trained (with turing=0.0) gave +2.88pp at N=1024:
  Ref (turing=0.0, W_phase frozen): 83.67%
  A   (turing=0.1, W_phase trained): 86.55% ← WINNER

But this was NEVER tested at N=4096 where the current best is 97.38%.
W_phase is a learnable parameter but was never added to the optimizer
in any N=4096 experiment — it sits at random initialization.

This is a one-line fix: pass lr_wphase to Trainer.

Note: step76 A used turing=0.1 (not 0.0). At N=4096, turing=0.0 is
confirmed optimal (step70). We test both: turing=0.0 + W_phase trained
AND turing=0.1 + W_phase trained, to see if the turing finding changes
when W_phase actually learns.

CONFIGS (N=4096, K_hh=4, K_iter=8, AH=1.0, 50%/75ep)
=======================================================
  Ref : turing=0.0, W_phase frozen  (step86 A reproduction)
  A   : turing=0.0, W_phase trained at lr_wpos
  B   : turing=0.0, W_phase trained at 0.1×lr_wpos
  C   : turing=0.1, W_phase trained at lr_wpos  (step76 winner scaled)

To reproduce:
    python -u scripts/train_step99_wphase_trained_n4096.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, run_metadata, GA_BEST
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 4096; N_IN = 25088; N_OUT = 10; D = 64
K_LOCAL = 2; K_RANDOM = 2; K_ITER = 8; K_IN = 50
K_PHASE = 8; BEAM_SIZE = 16; GEO_GAMMA = 0.5
ALPHA_REFLECT = 0.5; ALPHA_AHEBB = 1.0
N_GROUPS = max(8, N // 8)
STEP86A = 0.9659  # step86 A: K_hh=4, turing=0.0, 50%/75ep


@dataclass
class Config:
    key: str; label: str; turing: float; lr_wphase_scale: float | None
    # lr_wphase_scale: None=frozen, 1.0=same as lr_wpos, 0.1=10× slower


CONFIGS = [
    Config("Ref", "Ref  turing=0.0  W_phase frozen (step86-A repro)", 0.0, None),
    Config("A",   "A    turing=0.0  W_phase trained (1×lr_wpos)",     0.0, 1.0),
    Config("B",   "B    turing=0.0  W_phase trained (0.1×lr_wpos)",   0.0, 0.1),
    Config("C",   "C    turing=0.1  W_phase trained (1×lr_wpos)",     0.1, 1.0),
]

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=10)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg: Config, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=K_LOCAL, K_random=K_RANDOM,
        K_in=K_IN, K_iter=K_ITER, n_groups=N_GROUPS,
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=cfg.turing, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg, model, meta):
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")

    # Key change: pass lr_wphase to enable W_phase training
    lr_wpos = tk.get("lr_wpos", GA_BEST["lr_Wpos"])
    if cfg.lr_wphase_scale is not None:
        tk["lr_wphase"] = lr_wpos * cfg.lr_wphase_scale
        print(f"  W_phase TRAINED: lr_wphase={tk['lr_wphase']:.6f} "
              f"({cfg.lr_wphase_scale}× lr_wpos)")
    else:
        print(f"  W_phase FROZEN (not in optimizer)")

    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0 = time.time(); history = trainer.train(n_epochs=EPOCHS); elapsed = time.time() - t0
    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist); best_ep = int(np.argmax(top1_hist)) + 1
    frac = best_ep / len(history)
    result = {
        "label": cfg.label, "turing": cfg.turing,
        "wphase_trained": cfg.lr_wphase_scale is not None,
        "lr_wphase_scale": cfg.lr_wphase_scale,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1), "best_epoch_frac": round(frac, 3),
        "step86a_ref": STEP86A, "delta_vs_step86a": round(best - STEP86A, 4),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  "
          f"vs_step86a={best-STEP86A:+.4f}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  Data: 50%")
    print(f"Step 99: W_phase trained at N=4096 (scaling gap)")
    print(f"Baseline: step86 A = {STEP86A:.4f}\n")
    for c in CONFIGS:
        wp = f"trained({c.lr_wphase_scale}×)" if c.lr_wphase_scale else "frozen"
        print(f"  {c.key:4s}  turing={c.turing}  W_phase={wp}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step99_wphase_trained_n4096.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": K_ITER, "K_hh": K_LOCAL + K_RANDOM,
                "turing": cfg.turing, "wphase_trained": cfg.lr_wphase_scale is not None,
                "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 99 COMPLETE\n")
    print(f"  {'Key':4s}  {'turing':>7s}  {'W_phase':>10s}  {'top1':>8s}  {'vs_86a':>8s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            wp = f"trained({c.lr_wphase_scale}×)" if c.lr_wphase_scale else "frozen"
            print(f"  {c.key:4s}  {c.turing:>7.1f}  {wp:>10s}  "
                  f"{r['top1_best']:.4f}  {r['delta_vs_step86a']:+.4f}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
