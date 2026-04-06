"""Step 71: K_iter sweep at N=4096 — patched arch, Gen4+ params, 50%/75ep.

MOTIVATION
==========
step70 Ref established 97.20% at K_iter=8, N=4096, 100%/150ep.
Key question: can we get similar accuracy at lower K_iter?

Lower K_iter = faster inference + smaller memory footprint per forward pass.
At N=4096 the graph is large — fewer routing steps may be sufficient to
propagate information across relevant neighbourhoods.

step68 (buggy arch, N=1024): K_iter=16 was optimal (+0.51pp). Non-monotone:
8→10→12 declining, 16 recovers, 24 cliffs hard. Shape may differ at N=4096
on patched arch where signal propagates cleanly (no alpha_reflect bug).

CONFIGS (N=4096, D=64, 50% data, 75ep)
=======================================
  Ref : K_iter=8  — Gen4+ reference (matches step70 params at lower compute)
  A   : K_iter=4  — can half the routing steps preserve accuracy?
  B   : K_iter=6  — intermediate
  C   : K_iter=12 — modest increase
  D   : K_iter=16 — step68 optimal on buggy N=1024 base

All other Gen4+ params fixed across all configs:
  alpha_turing=0.3, alpha_reflect=0.5, AH=1.0 (wpos), beam=16, geo_gamma=0.5
  K_phase=8 (phase inhibition neighbours — independent of routing depth)

KEY QUESTIONS
=============
  A/B vs Ref → efficiency: does accuracy hold at K_iter < 8 at N=4096?
  C/D vs Ref → depth: does more routing help at this scale?
  Curve shape → monotone or non-monotone (as in step68)?

COMPARISON POINTS
=================
  step70 Ref: 97.20% (N=4096, K_iter=8, 100%/150ep) — full-scale reference
  step69 A:   85.04% (N=1024, K_iter=8, 50%/75ep)  — same compute budget, smaller N

Note: these runs are 50%/75ep so absolute numbers below 97.20%.
Compare RELATIVE to each other (efficiency curve) and to step69A (N scaling gain).

To reproduce:
    python -u scripts/train_step71_kiter_n4096.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.training.experiment_config    import (
    trainer_kwargs, topology_kwargs, run_metadata,
)
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
N         = 4096
D         = 64

# Fixed Gen4+ params — only K_iter varies
K_PHASE       = 8
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5
ALPHA_TURING  = 0.3
ALPHA_REFLECT = 0.5

# Comparison baselines
STEP70_FULL   = 0.9720   # step70 Ref: N=4096, K_iter=8, 100%/150ep
STEP69A_50PCT = 0.8504   # step69 A:   N=1024, K_iter=8, 50%/75ep

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


def make_model(k_iter: int, seed_offset: int = 0) -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED + seed_offset)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=k_iter, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base,
        K_phase=K_PHASE,
        alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING,
        beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos")


def run(label: str, model: nn.Module, meta: dict,
        results: dict, out_path: Path) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
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
        "label":            label,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     top1_hist,
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    result.update(meta)

    d_full   = best - STEP70_FULL
    d_s69a   = best - STEP69A_50PCT
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_step70={d_full:+.4f}  vs_step69A={d_s69a:+.4f}  t={elapsed:.0f}s"
    )

    results[meta["key"]] = result
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  [saved {out_path.name}]")
    return result


# ── Configs ───────────────────────────────────────────────────────────────────
CONFIGS = [
    {"key": "Ref", "label": "Ref  K_iter=8  (Gen4+ reference, matches step70 params)", "K_iter": 8,  "seed_offset": 0},
    {"key": "A",   "label": "A    K_iter=4  (half routing depth)",                       "K_iter": 4,  "seed_offset": 1},
    {"key": "B",   "label": "B    K_iter=6  (3/4 routing depth)",                        "K_iter": 6,  "seed_offset": 2},
    {"key": "C",   "label": "C    K_iter=12 (1.5× routing depth)",                       "K_iter": 12, "seed_offset": 3},
    {"key": "D",   "label": "D    K_iter=16 (2× routing depth, step68 winner at N=1024)","K_iter": 16, "seed_offset": 4},
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%  N: {N}")
    print(f"Step 71: K_iter sweep at N=4096 (patched arch, Gen4+ params)")
    print(f"Fixed: turing={ALPHA_TURING}  reflect={ALPHA_REFLECT}  AH=1.0  beam={BEAM_SIZE}  geo={GEO_GAMMA}  K_phase={K_PHASE}")
    print(f"step70 full-scale ref (100%/150ep): {STEP70_FULL:.4f}")
    print(f"step69A at N=1024   (50%/75ep):     {STEP69A_50PCT:.4f}")

    tr, va = get_loaders()
    print(f"Dataset: train={len(tr.dataset)}  val={len(va.dataset)}")

    tk = topology_kwargs(N)
    print(f"Topology: N={N} K_local={tk['K_local']} K_random={tk['K_random']} "
          f"K_in={tk['K_in']} n_groups={tk['n_groups']}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step71_kiter_n4096.json"

    for cfg in CONFIGS:
        model = make_model(k_iter=cfg["K_iter"], seed_offset=cfg["seed_offset"]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Config {cfg['key']}: K_iter={cfg['K_iter']}  {n_params:,} trainable params")

        meta = {
            "key":           cfg["key"],
            "N":             N, "D": D, "K_iter": cfg["K_iter"],
            "K_phase":       K_PHASE,
            "alpha_ahebb":   1.0, "variant": "wpos",
            "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing":  ALPHA_TURING,
            "beam_size":     BEAM_SIZE, "geo_gamma": GEO_GAMMA,
            "data_frac":     0.5,
        }
        run(cfg["label"], model, meta, results, out_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STEP 71 COMPLETE — K_iter sweep at N=4096")
    print(f"step70 full-scale (100%/150ep, K_iter=8): {STEP70_FULL:.4f}")
    print(f"step69A at N=1024 (50%/75ep,  K_iter=8): {STEP69A_50PCT:.4f}")
    print()

    print(f"  {'Key':4s}  {'K_iter':>7s}  {'top1':>8s}  {'vs step70':>10s}  {'vs step69A':>10s}")
    print("  " + "-" * 55)
    ref_top1 = results.get("Ref", {}).get("top1_best", 0.0)
    for cfg in CONFIGS:
        key = cfg["key"]
        if key not in results:
            continue
        r      = results[key]
        d_full = r["top1_best"] - STEP70_FULL
        d_s69a = r["top1_best"] - STEP69A_50PCT
        print(f"  {key:4s}  {cfg['K_iter']:>7d}  {r['top1_best']:.4f}    {d_full:>+.4f}      {d_s69a:>+.4f}")

    if "Ref" in results:
        print(f"\n  K_iter efficiency curve (vs Ref={ref_top1:.4f}):")
        for cfg in CONFIGS:
            if cfg["key"] == "Ref" or cfg["key"] not in results:
                continue
            delta = results[cfg["key"]]["top1_best"] - ref_top1
            print(f"    K_iter={cfg['K_iter']:2d}: {delta:+.4f}")
