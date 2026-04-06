"""Step 64: Stacked SGNNET depth and parallel-stream ablation.

MOTIVATION
==========
All wave-1 mechanisms (steps 58-63) added excitation WITHIN a single routing
layer and failed — gate-death, safety-valve collapse, or -18pp vs Ref.
The root failure: additive excitation within 8 K_iter steps oversaturates the
threshold gate, collapsing routing diversity.

A fundamentally different hypothesis: instead of modifying mechanism inside a
layer, stack multiple independent Gen4 AntiHebb layers IN SERIES, or run two
streams IN PARALLEL and fuse before readout. Each layer retains exactly the
mechanism that works; depth comes from layer composition, not internal changes.

If the Gen4 routing already extracts all structure in one pass, series layers
will not help (or hurt). If depth helps, it suggests the routing is genuinely
iterative at a coarser grain than K_iter — more like multiple passes over the
representation, not more steps within a single pass.

ARCHITECTURE
============
SGNNET_Stacked  — layers in series: seed(x) → route_1 → route_2 → ... → readout
  series_mode='passthrough': Z flows directly between layers (no new params)
  series_mode='bridge': learned Linear(D,D,bias=False) between layers
  skip=True: Z_out = normalize(Z_out + Z_in) for each layer 2+

SGNNET_Parallel — two independent streams, fused before readout
  fusion_mode='sum': Z_fused = normalize(Z_a + Z_b)
  fusion_mode='concat': Z_cat=[Z_a,Z_b] [B,N,2D], proj Linear(2D,D), Z_fused=normalize(proj)

CONFIGS
=======
  Ref  — 1-layer AntiHebb baseline (same as step57 Ref)
  A    — 2-layer series, passthrough, no skip
  B    — 2-layer series, passthrough, skip (residual)
  C    — 2-layer series, bridge (Linear D→D), no skip
  D    — 3-layer series, passthrough, no skip
  E    — 2-parallel, sum fusion
  F    — 2-parallel, concat-project fusion

QUESTIONS
=========
  A vs Ref  — does any depth at all help?
  B vs A    — does skip connection stabilize deep routing?
  C vs A    — does learned bridge improve over raw Z pass-through?
  D vs A    — does 3 layers beat 2?
  E vs Ref  — does parallel diversity help?
  F vs E    — does richer fusion (concat-proj) beat simple sum?

To reproduce:
    python -u scripts/train_step64_stacked_sgnnet.py --device cpu
    python -u scripts/train_step64_stacked_sgnnet.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_stacked        import SGNNET_Stacked, SGNNET_Parallel
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 1024
D      = 64
K_ITER = 8

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_ah_layer(seed_offset: int = 0) -> SGNNET_AntiHebbian:
    """Factory: one SGNNET_AntiHebbian (Gen4 params, N=1024, D=64, K_iter=8, AH=1.0)."""
    torch.manual_seed(SEED + seed_offset)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base=base, K_phase=8, beam_size=16,
        theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=0.5,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos")


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# Load REF_BASELINE from step57
_ref_path = ROOT / "results" / "train_step57_benchmark_ablation.json"
try:
    REF_BASELINE = json.loads(_ref_path.read_text()).get("top1_best", 0.7353)
    print(f"Loaded REF_BASELINE from step57: {REF_BASELINE:.4f}")
except Exception:
    REF_BASELINE = 0.7353
    print(f"step57 result not found — using fallback REF_BASELINE={REF_BASELINE:.4f}")


# (key, label, factory_fn)
CONFIGS = [
    ("Ref", "Ref  1-layer AntiHebb α=1.0 baseline",
     lambda: make_ah_layer(0)),

    ("A",   "A    2-layer series passthrough no-skip",
     lambda: SGNNET_Stacked(
         [make_ah_layer(0), make_ah_layer(1)],
         series_mode="passthrough", skip=False)),

    ("B",   "B    2-layer series passthrough + skip",
     lambda: SGNNET_Stacked(
         [make_ah_layer(0), make_ah_layer(1)],
         series_mode="passthrough", skip=True)),

    ("C",   "C    2-layer series bridge no-skip",
     lambda: SGNNET_Stacked(
         [make_ah_layer(0), make_ah_layer(1)],
         series_mode="bridge", skip=False)),

    ("D",   "D    3-layer series passthrough no-skip",
     lambda: SGNNET_Stacked(
         [make_ah_layer(0), make_ah_layer(1), make_ah_layer(2)],
         series_mode="passthrough", skip=False)),

    ("E",   "E    2-parallel sum fusion",
     lambda: SGNNET_Parallel(
         make_ah_layer(0), make_ah_layer(1),
         fusion_mode="sum")),

    ("F",   "F    2-parallel concat-project fusion",
     lambda: SGNNET_Parallel(
         make_ah_layer(0), make_ah_layer(1),
         fusion_mode="concat")),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 64: Stacked SGNNET  |  REF_BASELINE={REF_BASELINE:.4f} (step57)")
    print("Testing series depth, skip connections, linear bridge, parallel fusion")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, factory in CONFIGS:
        model = factory().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        meta = {"N": N, "D": D, "K_iter": K_ITER, "config": key,
                "n_params": n_params, "data_frac": 0.5}
        print(f"\n  {label}  |  params={n_params:,}")
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step64_stacked_sgnnet.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Step 64: Stacked SGNNET (run-Ref={ref_val:.4f} / step57={REF_BASELINE:.4f}) --")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'vs_57':>8}  {'params':>10}  {'best_ep':>8}  {'t(s)':>6}")
    print("  " + "-"*110)
    for k, r in results.items():
        d_ref = r["top1_best"] - ref_val
        d_57  = r["top1_best"] - REF_BASELINE
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d_ref:>+8.4f}  {d_57:>+8.4f}"
              f"  {r.get('n_params', 0):>10,}  {r.get('best_epoch', 0):>7d}  {r['elapsed_s']:>6.0f}")

    print("\n  Key questions:")
    print("  A vs Ref  -> does depth at all help (any gain from 2 layers)?")
    print("  B vs A    -> does skip connection stabilize deep routing?")
    print("  C vs A    -> does learned bridge improve over raw Z pass-through?")
    print("  D vs A    -> does 3 layers beat 2?")
    print("  E vs Ref  -> does parallel diversity help?")
    print("  F vs E    -> does richer fusion (concat-proj) beat simple sum?")
