"""Step 72: N-scaling curve on patched arch with confirmed defaults.

MOTIVATION
==========
step56 mapped N-scaling on BUGGY arch: N=512(69.58%), 1024(80.92%),
2048(81.10%), 4096(84.36%), 10000(82.37%). But the step69 patch gave
+9.83pp — the entire curve is invalidated.

This is CRITICAL for the 1% FLOPs goal. At N=4096, minimum FLOPs ≈ 30M
(seed alone = 13M). To hit ≤1.2M FLOPs, we need N=256-512. But what
accuracy do small N achieve on the patched arch?

If N=512 on patched arch reaches ~85%+ (vs old 69.58%), the FLOPs path
becomes viable: N=512 + D=32 + K_hh=2 ≈ 1-2M FLOPs range.

CONFIGS (D=64, K_hh=4, K_iter=8, AH=1.0, turing=0.0, 50%/75ep)
=================================================================
  A : N=256   n_groups=32   (smallest — FLOPs target candidate)
  B : N=512   n_groups=64   (1% FLOPs candidate at D=32)
  C : N=1024  n_groups=128  (step69 Ref replication on K_hh=4)
  D : N=2048  n_groups=256
  E : N=4096  n_groups=512  (step86-A replication = 96.59%)

All use confirmed defaults: K_hh=4 (K_local=2, K_random=2), K_iter=8,
D=64, turing=0.0, reflect=0.5, AH=1.0 wpos.

FLOPs per config (K_hh=4, K_iter=8, D=64, K_in=50):
  A (N=256):   seed=0.8M + route=2.4M + read=0.2M = 3.4M   (2.8% VGG16)
  B (N=512):   seed=1.6M + route=4.7M + read=0.3M = 6.7M   (5.6%)
  C (N=1024):  seed=3.3M + route=9.4M + read=0.7M = 13.4M  (11.2%)
  D (N=2048):  seed=6.6M + route=18.9M + read=1.3M = 26.7M (22.3%)
  E (N=4096):  seed=13.1M + route=37.7M + read=2.6M = 53.5M (44.7%)

Wait — these FLOPs use the step86 formula. Let me recalculate with
the exact formula from step86 (K_hh=4):
  seed    = N * K_in * D
  step    = N * K_hh * D * 2 + N * D + N * D * 2  = N * D * (2*K_hh + 3)
  routing = K_iter * step
  readout = N * N_out * D
  total   = seed + routing + readout

KEY QUESTIONS
=============
  1. Is N-scaling monotone on patched arch? (step56 showed peak at N=4096)
  2. What is N=512 accuracy? If >85%, the FLOPs path is viable.
  3. Where does N=256 land? If >75%, even ultra-low FLOPs is possible.
  4. Does K_hh=4 change the scaling curve shape vs K_hh=6?

To reproduce:
    python -u scripts/train_step72_nscaling_patched.py --device mps
    python -u scripts/train_step72_nscaling_patched.py --device cpu
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

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs, run_metadata
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = 75
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N_IN      = 25088
N_OUT     = 10
D         = 64
K_IN      = 50
K_LOCAL   = 2        # K_hh=4 default
K_RANDOM  = 2
K_ITER    = 8
K_PHASE   = 8
BEAM_SIZE = 16
GEO_GAMMA = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

# Old step56 results (buggy arch, K_hh=6, turing=0.3) for comparison
STEP56 = {256: None, 512: 0.6958, 1024: 0.8092, 2048: 0.8110, 4096: 0.8436}
# step86-A (patched arch, K_hh=4, 50%/75ep)
STEP86A = 0.9659


@dataclass
class Config:
    key:   str
    label: str
    N:     int


CONFIGS = [
    Config("A", "A  N=256   (ultra-low FLOPs candidate)",   256),
    Config("B", "B  N=512   (1% FLOPs candidate at D=32)",  512),
    Config("C", "C  N=1024  (step69 Ref scale)",            1024),
    Config("D", "D  N=2048",                                2048),
    Config("E", "E  N=4096  (step86-A replication)",        4096),
]


_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0
        )
        _loaders = (tr, va)
    return _loaders


def compute_flops(N: int) -> int:
    K_hh    = K_LOCAL + K_RANDOM
    seed    = N * K_IN * D
    step    = N * K_hh * D * 2 + N * D + N * D * 2
    routing = K_ITER * step
    readout = N * N_OUT * D
    return seed + routing + readout


def make_model(N: int, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    n_groups = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=K_LOCAL, K_random=K_RANDOM,
        K_in=K_IN, K_iter=K_ITER, n_groups=n_groups,
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg: Config, model: nn.Module, meta: dict) -> dict:
    N     = cfg.N
    flops = compute_flops(N)
    print(f"\n{'='*70}")
    print(f"{cfg.label}")
    print(f"  N={N}  n_groups={max(8,N//8)}  params={count_params(model):,}"
          f"  FLOPs={flops/1e6:.1f}M  vs_VGG16={flops/119_578_624:.1%}")
    print(f"{'='*70}")

    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)

    old = STEP56.get(N)
    delta_old = round(best - old, 4) if old else None

    result = {
        "label":            cfg.label,
        "N":                N,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     top1_hist,
        "flops_per_sample": flops,
        "flops_M":          round(flops / 1e6, 2),
        "flops_vs_vgg16":   round(flops / 119_578_624, 3),
        "params":           count_params(model),
        "step56_old":       old,
        "delta_vs_step56":  delta_old,
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    old_str = f"  vs_step56={delta_old:+.4f}" if delta_old else ""
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  FLOPs={flops/1e6:.1f}M{old_str}  t={elapsed:.0f}s"
    )
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 72: N-scaling curve on patched arch, confirmed defaults")
    print(f"K_hh=4 K_iter=8 D=64 turing=0.0 reflect=0.5 AH=1.0")
    print(f"step86-A (N=4096, 50%/75ep): {STEP86A:.4f}")
    print()
    print(f"  {'Key':4s}  {'N':>6s}  {'FLOPs':>8s}  {'vs_VGG16':>9s}  {'step56':>8s}")
    for cfg in CONFIGS:
        f = compute_flops(cfg.N)
        old = STEP56.get(cfg.N)
        print(f"  {cfg.key:4s}  {cfg.N:>6d}  {f/1e6:>7.1f}M  {f/119_578_624:>8.1%}"
              f"  {f'{old:.2%}' if old else '—':>8s}")
    print()

    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step72_nscaling_patched.json"

    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg.N, seed_offset=i).to(DEVICE)
        meta  = {
            "N": cfg.N, "D": D, "K_iter": K_ITER,
            "K_hh": K_LOCAL + K_RANDOM, "n_groups": max(8, cfg.N // 8),
            "alpha_turing": ALPHA_TURING, "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
        }
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 72 COMPLETE — N-Scaling on Patched Arch")
    print()
    print(f"  {'N':>6s}  {'top1':>8s}  {'FLOPs':>8s}  {'vs_VGG16':>9s}"
          f"  {'step56':>8s}  {'delta':>8s}  {'params':>8s}")
    for cfg in CONFIGS:
        if cfg.key not in results:
            continue
        r   = results[cfg.key]
        old = STEP56.get(cfg.N)
        old_str   = "{:.2%}".format(old) if old else "—"
        delta_str = "{:+.4f}".format(r['delta_vs_step56']) if r['delta_vs_step56'] else "—"
        print(f"  {cfg.N:>6d}  {r['top1_best']:.4f}  {r['flops_M']:>7.1f}M"
              f"  {r['flops_vs_vgg16']:>8.1%}"
              f"  {old_str:>8s}"
              f"  {delta_str:>8s}"
              f"  {r['params']:>8,}")
    print()
    # N-scaling analysis
    ns   = [cfg.N for cfg in CONFIGS if cfg.key in results]
    accs = [results[cfg.key]["top1_best"] for cfg in CONFIGS if cfg.key in results]
    if len(ns) >= 2:
        monotone = all(accs[i] <= accs[i+1] for i in range(len(accs)-1))
        peak_n   = ns[accs.index(max(accs))]
        print(f"  Scaling: {'monotone' if monotone else 'non-monotone'}")
        print(f"  Peak at N={peak_n} ({max(accs):.4f})")
        # FLOPs efficiency: accuracy per M FLOPs
        for cfg in CONFIGS:
            if cfg.key in results:
                r = results[cfg.key]
                eff = r["top1_best"] / r["flops_M"]
                print(f"    N={cfg.N}: {eff:.4f} acc/M_FLOPs")
