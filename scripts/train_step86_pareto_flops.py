"""Step 86: FLOPs / accuracy Pareto sweep.

MOTIVATION
==========
Project best (step70 B): 97.38% at 529K params, 49.2M FLOPs/sample.
Parameter efficiency is strong (0.44% of VGG16 FC).
FLOPs efficiency can be improved — routing dominates at 68% of total.

FLOPs breakdown (N=4096, K_iter=8, turing=0.0):
    Seed    (N × K_in × D)           = 13.1M  (27%)
    Routing (K_iter × N × K_hh × D) = 33.6M  (68%)   ← lever
    Readout (N × N_out × D)          =  2.6M   (5%)

THREE LEVERS:
  K_hh   = K_local + K_random  (currently 6 = 4+2)
  K_iter (currently 8 at N=4096, turing=0.0)
  D      (currently 64; D=32 halves all terms)

KEY HYPOTHESIS: AH suppression already effectively zeros out the weakest
K_hh neighbours. Reducing K_hh from 6→3 may cost little accuracy while
saving ~45% of routing FLOPs. K_random=2 is the minimum for graph
connectivity — only K_local is reduced.

CONFIGS (N=4096, D=64, turing=0.0, AH=1.0, 50%/75ep)
=======================================================
  Ref : K_hh=6  K_iter=8  D=64  →  49.2M FLOPs  (reproduces step71 Ref)
  A   : K_hh=4  K_iter=8  D=64  →  38.0M FLOPs  (-23%)
  B   : K_hh=3  K_iter=8  D=64  →  32.4M FLOPs  (-34%)
  C   : K_hh=2  K_iter=8  D=64  →  26.8M FLOPs  (-45%)  [K_local=0: random only]
  D   : K_hh=6  K_iter=6  D=64  →  38.0M FLOPs  (K_iter lever, same FLOPs as A)
  E   : K_hh=4  K_iter=6  D=64  →  28.3M FLOPs  (compound: -42%)
  F   : K_hh=6  K_iter=8  D=32  →  24.6M FLOPs  (D lever: halves everything)

FLOPs formula (turing=0.0 per-step):
  seed           = N * K_in * D
  per_step       = N * K_hh * D * 2 + N * D + N * D * 2   (gather + gate + norm)
  routing_total  = K_iter * per_step
  readout        = N * N_out * D
  total          = seed + routing_total + readout

PARETO TARGET: find configs that achieve ≥95% accuracy at ≤30M FLOPs.

To reproduce:
    python -u scripts/train_step86_pareto_flops.py --device mps
    python -u scripts/train_step86_pareto_flops.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass

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
parser.add_argument("--only", nargs="*", default=None,
                    help="Run only these config keys, e.g. --only Ref A B")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = 75
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N         = 4096
N_IN      = 25088
N_OUT     = 10
K_IN      = 50          # input fan-in per neuron (fixed across all configs)
K_RANDOM  = 2           # minimum for graph connectivity — never reduced
K_PHASE   = 8
BEAM_SIZE = 16
GEO_GAMMA = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0     # turing=0.0 is the project best config
ALPHA_AHEBB   = 1.0

# Reference from step71 Ref (N=4096, 50%/75ep, patched arch)
STEP71_REF = 0.9587

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


def compute_flops(N: int, K_hh: int, K_iter: int, D: int,
                  K_in: int = K_IN, N_out: int = N_OUT) -> int:
    """Estimate FLOPs per sample (MACs counted as 1 FLOP).

    Components:
      seed    = N * K_in * D
      per_step = N * K_hh * D * 2  (gather weighted sum)
               + N * D              (gate: relu(Z - theta))
               + N * D * 2          (L2 normalize)
      routing = K_iter * per_step
      readout = N * N_out * D
    """
    seed    = N * K_in * D
    step    = N * K_hh * D * 2 + N * D + N * D * 2
    routing = K_iter * step
    readout = N * N_out * D
    return seed + routing + readout


@dataclass
class Config:
    key:     str
    label:   str
    K_local: int
    K_iter:  int
    D:       int
    K_in:    int = K_IN   # input fan-in (default 50)


CONFIGS = [
    Config("Ref", "Ref  K_hh=6  K_iter=8  D=64  (baseline)",  4, 8, 64),
    Config("A",   "A    K_hh=4  K_iter=8  D=64  (-23% FLOPs)", 2, 8, 64),
    Config("B",   "B    K_hh=3  K_iter=8  D=64  (-34% FLOPs)", 1, 8, 64),
    Config("C",   "C    K_hh=2  K_iter=8  D=64  (-45% FLOPs)", 0, 8, 64),  # random-only
    Config("D",   "D    K_hh=6  K_iter=6  D=64  (-23% FLOPs)", 4, 6, 64),
    Config("E",   "E    K_hh=4  K_iter=6  D=64  (-42% FLOPs)", 2, 6, 64),
    Config("F",   "F    K_hh=6  K_iter=8  D=32  (-50% FLOPs)", 4, 8, 32),
    Config("G",   "G    K_hh=6  K_iter=4  D=64  (-37% FLOPs)", 4, 4, 64),       # K_iter floor test
    Config("H",   "H    K_hh=6  K_iter=4  D=32  (-68% FLOPs)", 4, 4, 32),       # compound floor
    Config("I",   "I    K_in=25 K_iter=8  D=64  (-13% FLOPs)", 4, 8, 64, K_in=25),  # seed lever
]


def make_model(cfg: Config, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    K_local_eff = max(0, cfg.K_local)  # K_local=0 → random-only conn_hh
    # When K_local=0, pass K_local=1 and let random=1 give K_hh=2
    # (the _build_smallworld_conn handles edge cases)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=K_local_eff, K_random=K_RANDOM,
        K_in=cfg.K_in, K_iter=cfg.K_iter,
        n_groups=max(8, N // 8),  # matches topology_kwargs(N) used in step71
        norm_mode="l2", D=cfg.D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(cfg: Config, model: nn.Module, meta: dict) -> dict:
    K_hh   = cfg.K_local + K_RANDOM
    flops  = compute_flops(N, K_hh, cfg.K_iter, cfg.D, K_in=cfg.K_in)
    params = count_params(model)

    print(f"\n{'='*70}")
    print(f"{cfg.label}")
    print(f"  K_hh={K_hh}  K_iter={cfg.K_iter}  D={cfg.D}  "
          f"params={params:,}  FLOPs={flops/1e6:.1f}M  "
          f"vs_VGG16={flops/119_578_624:.2f}x")
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

    result = {
        "label":           cfg.label,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":      best_ep,
        "epochs_run":      len(history),
        "elapsed_s":       round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag":"training_too_short" if frac < 0.7 else "converged",
        "top1_history":    top1_hist,
        "step71_ref":      STEP71_REF,
        "delta_vs_ref":    round(best - STEP71_REF, 4),
        "flops_per_sample":flops,
        "flops_M":         round(flops / 1e6, 2),
        "flops_vs_vgg16":  round(flops / 119_578_624, 3),
        "params":          params,
        "params_vs_vgg16_pct": round(params / 119_578_624 * 100, 3),
        "_meta":           run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_ref={best-STEP71_REF:+.4f}  FLOPs={flops/1e6:.1f}M  t={elapsed:.0f}s"
    )
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  N={N}  Data: 50%")
    print(f"Step 86: FLOPs / accuracy Pareto sweep")
    print(f"Ref: step71 Ref (N=4096, K_hh=6, K_iter=8) = {STEP71_REF:.4f}")
    print()
    print(f"  {'Cfg':4s}  {'FLOPs':>8s}  {'vs_VGG16':>9s}  {'K_hh':>5s}  {'K_iter':>6s}  {'D':>3s}  Description")
    for cfg in CONFIGS:
        K_hh  = cfg.K_local + K_RANDOM
        flops = compute_flops(N, K_hh, cfg.K_iter, cfg.D, K_in=cfg.K_in)
        print(f"  {cfg.key:4s}  {flops/1e6:>7.1f}M  {flops/119_578_624:>8.2f}x"
              f"  {K_hh:>5d}  {cfg.K_iter:>6d}  {cfg.D:>3d}  {cfg.label}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step86_pareto_flops.json"

    active_configs = [c for c in CONFIGS if args.only is None or c.key in args.only]

    for i, cfg in enumerate(active_configs):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        K_hh  = cfg.K_local + K_RANDOM
        meta  = {
            "N": N, "D": cfg.D, "K_iter": cfg.K_iter,
            "K_hh": K_hh, "K_local": cfg.K_local, "K_random": K_RANDOM,
            "K_in": cfg.K_in,
            "alpha_turing": ALPHA_TURING,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "flops_per_sample": compute_flops(N, K_hh, cfg.K_iter, cfg.D, K_in=cfg.K_in),
        }
        results[cfg.key] = run(cfg, model, meta)  # type: ignore[index]
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 86 COMPLETE — FLOPs / Accuracy Pareto")
    print(f"Ref (step71): {STEP71_REF:.4f}  @  49.2M FLOPs")
    print()
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs_ref':>8s}  {'FLOPs':>8s}  {'vs_VGG16':>9s}  {'K_hh':>5s}  {'K_iter':>6s}  {'D':>3s}")
    for cfg in CONFIGS:
        if cfg.key not in results:
            continue
        r     = results[cfg.key]
        K_hh  = cfg.K_local + K_RANDOM
        flops = compute_flops(N, K_hh, cfg.K_iter, cfg.D, K_in=cfg.K_in)
        print(f"  {cfg.key:4s}  {r['top1_best']:.4f}    {r['delta_vs_ref']:+.4f}"
              f"  {flops/1e6:>7.1f}M  {flops/119_578_624:>8.2f}x"
              f"  {K_hh:>5d}  {cfg.K_iter:>6d}  {cfg.D:>3d}")
    print()
    print("  Pareto frontier: configs where accuracy drop < 1pp per 10M FLOPs saved")
    print("  → feeds into step87 (winner at full 100%/150ep) or step88 (compound with redistribution routing)")
