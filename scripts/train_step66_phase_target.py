"""Step 66: Phase-Target Routing with Local Plasticity.

Two weights per neuron — analogous to Q·K attention:
  W_pos        (Key/Value): signal operator, learned via gradient descent
  phase_target (Query):     routing query vector, updated by LOCAL plasticity rule only

MOTIVATION
==========
Wave-1 failures (steps 58-63) modified the ROUTING LOOP itself — adding attention,
gating, or interneurons into the forward pass. All failed via gate death.

This experiment is different: routing is attention-based but FIXED at forward time.
The phase_target query vectors learn WHERE to attend via a local Hebbian-like rule
OUTSIDE the gradient path. No gradient flows through phase_target.

The local plasticity rule (per-batch, no gradients):
  For each neuron i, rank K neighbors by contribution:
    contribution[j] = mean_batch(normalize(Z[i]) · normalize(Z[j]))
  Top-K/2 (helpful):     phase_target[i] += lr_phase · mean(W_pos[j])
  Bottom-K/2 (suppress): phase_target[i] -= lr_phase · mean(W_pos[j])
  Renormalize to unit sphere.

Diversity penalty (β > 0): adds -β·(phase_target[i]·phase_target[j]) to routing
scores, discouraging multiple neurons from querying the same direction.

CONFIGS (N=1024, D=64, K_iter=8, 50% data, 75ep)
=================================================
  Ref  : AntiHebb α=1.0 static routing (established REF_BASELINE ~73.53%)
  A    : PhaseTarget routing only — no plasticity, no diversity, no AH
         (tests: does attention-like routing over conn_hh help vs static routing?)
  B    : PhaseTarget + plasticity (K/2 split, lr_phase=0.01), no diversity, no AH
         (tests: does local plasticity improve over fixed phase_target?)
  C    : PhaseTarget + plasticity + diversity penalty (β=0.1)
         (tests: does anti-alignment in phase_target space add further?)
  D    : PhaseTarget + plasticity + AH suppression (ah_alpha=1.0)
         (tests: can phase plasticity compound with AntiHebb W_pos diversity?)

KEY QUESTIONS
=============
  A vs Ref  → does attention routing (random phase_target) beat static uniform?
  B vs A    → does local plasticity improve routing quality over fixed queries?
  C vs B    → does diversity penalty in phase_target space add signal?
  D vs B    → does AH suppress + phase plasticity compound?
  Best vs REF_BASELINE → how close to AH static routing can phase plasticity get?

Stage 2 (if any config wins): full data 150ep vs 80.08% baseline.

To reproduce:
    python -u scripts/train_step66_phase_target.py --device mps
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
from src.sgnnet.model_phase_target     import SGNNET_PhaseTarget
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS  = 75
BATCH   = 128
SEED    = 42
DATA    = "data/store.h5"
N       = 1024
D       = 64
K_ITER  = 8

# Gen4 calibrated parameters (from step22b + step29c)
K_PHASE      = 8
ALPHA_REFLECT = 0.5
BEAM_SIZE    = 16
GEO_GAMMA    = 0.5

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


def make_smallworld_base(seed_offset: int = 0) -> SGNNET_SmallWorld:
    """Bare SmallWorld — shared factory for PhaseTarget wrapping."""
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    return SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )


def make_ref() -> SGNNET_AntiHebbian:
    """Standard AH=1.0 Resonant baseline (REF_BASELINE config)."""
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base,
        K_phase=K_PHASE,
        alpha_reflect=ALPHA_REFLECT,
        alpha_turing=0.0,
        beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA,
        resonant_mode="dynamic_z_geo",
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos")


def make_phase_target(
    plasticity: bool = True,
    beta: float = 0.0,
    lr_phase: float = 0.01,
    ah_alpha: float = 0.0,
    tau: float = 1.0,
    seed_offset: int = 0,
) -> SGNNET_PhaseTarget:
    """PhaseTarget wrapper around bare SmallWorld."""
    base = make_smallworld_base(seed_offset=seed_offset)
    return SGNNET_PhaseTarget(
        base,
        tau=tau,
        beta=beta,
        lr_phase=lr_phase,
        plasticity=plasticity,
        ah_alpha=ah_alpha,
    )


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)
    last5     = history[-5:]

    result = {
        "label": label,
        "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep,
        "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  diag={result['convergence_diag']}  t={elapsed:.0f}s"
    )
    return result


# ── Load REF_BASELINE ────────────────────────────────────────────────────────
_ref_path = ROOT / "results" / "train_step57_benchmark_ablation.json"
try:
    REF_BASELINE = json.loads(_ref_path.read_text()).get("top1_best", 0.7353)
    print(f"Loaded REF_BASELINE from step57: {REF_BASELINE:.4f}")
except Exception:
    REF_BASELINE = 0.7353
    print(f"step57 not found — using fallback REF_BASELINE={REF_BASELINE:.4f}")


# ── Configs ──────────────────────────────────────────────────────────────────
CONFIGS = [
    (
        "Ref",
        "Ref   AntiHebb α=1.0 static routing (REF_BASELINE)",
        lambda: make_ref(),
        {"mechanism": "antihebb", "alpha_ahebb": 1.0, "data_frac": 0.5},
    ),
    (
        "A",
        "A     PhaseTarget routing only — no plasticity, no diversity, no AH",
        lambda: make_phase_target(plasticity=False, beta=0.0, ah_alpha=0.0, seed_offset=0),
        {"mechanism": "phase_target", "plasticity": False, "beta": 0.0,
         "lr_phase": 0.0, "ah_alpha": 0.0, "tau": 1.0, "data_frac": 0.5},
    ),
    (
        "B",
        "B     PhaseTarget + plasticity (lr=0.01, K/2 split), no diversity, no AH",
        lambda: make_phase_target(plasticity=True, beta=0.0, lr_phase=0.01,
                                   ah_alpha=0.0, seed_offset=1),
        {"mechanism": "phase_target", "plasticity": True, "beta": 0.0,
         "lr_phase": 0.01, "ah_alpha": 0.0, "tau": 1.0, "data_frac": 0.5},
    ),
    (
        "C",
        "C     PhaseTarget + plasticity + diversity (β=0.1), no AH",
        lambda: make_phase_target(plasticity=True, beta=0.1, lr_phase=0.01,
                                   ah_alpha=0.0, seed_offset=2),
        {"mechanism": "phase_target", "plasticity": True, "beta": 0.1,
         "lr_phase": 0.01, "ah_alpha": 0.0, "tau": 1.0, "data_frac": 0.5},
    ),
    (
        "D",
        "D     PhaseTarget + plasticity + AH suppression (ah_alpha=1.0)",
        lambda: make_phase_target(plasticity=True, beta=0.0, lr_phase=0.01,
                                   ah_alpha=1.0, seed_offset=3),
        {"mechanism": "phase_target", "plasticity": True, "beta": 0.0,
         "lr_phase": 0.01, "ah_alpha": 1.0, "tau": 1.0, "data_frac": 0.5},
    ),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 66: Phase-Target Routing with Local Plasticity  |  REF_BASELINE={REF_BASELINE:.4f}")
    print("W_pos=Key/Value (gradient), phase_target=Query (local plasticity only)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step66_phase_target.json"

    for key, label, model_fn, meta_extra in CONFIGS:
        model = model_fn().to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": K_ITER, **meta_extra}
        results[key] = run(label, model, meta)
        results[key].update(meta_extra)

        # Save incrementally so partial results survive crashes
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 66 COMPLETE  |  REF_BASELINE={REF_BASELINE:.4f}")
    for key, label, _, _ in CONFIGS:
        r = results[key]
        delta = r['top1_best'] - REF_BASELINE
        print(f"  {key:4s}  {r['top1_best']:.4f}  ({delta:+.4f} vs ref)  ep={r['best_epoch']}/{r['epochs_run']}")
