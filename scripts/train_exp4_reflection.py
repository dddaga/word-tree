"""Exp 4: Signal reflection routing ablation at N=1024, D=64.

QUESTION
========
Does sign-based conditional propagation (reflection routing) improve accuracy
over the SmallWorld + AntiHebb baseline?

4 configs at N=1024, D=64, K_iter=8, 100 epochs:
  A: SmallWorld + AntiHebb alpha=0.7 wpos           (reference, no reflection)
  B: SmallWorld + AntiHebb alpha=0.7 + leaky-reflect (alpha_reflect=0.1, theta=0.0)
  C: SmallWorld + AntiHebb alpha=0.7 + hard-reflect  (alpha_reflect=1.0, theta=0.5)
  D: SmallWorld + AntiHebb alpha=0.7 + leaky-reflect (alpha_reflect=0.3, theta=0.0)

Key metric: top1 accuracy + dead_neuron_frac per epoch.
Dead frac > 5% indicates dying neuron cascade risk.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.trainer import Trainer
from src.training.dataset import make_loaders
from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_reflection import SGNNET_Reflection


# -- CLI -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch", type=int, default=128)
ARGS = parser.parse_args()

DEVICE = ARGS.device
EPOCHS = ARGS.epochs
BATCH  = ARGS.batch
SEED   = 42
DATA   = "data/store.h5"
N      = 1024
D      = 64
K_ITER = 8

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


# -- Base model factory -------------------------------------------------------

def make_base_antihebb(alpha_ahebb: float = 0.7) -> SGNNET_AntiHebbian:
    """Build SmallWorld + Resonant + AntiHebb (the confirmed best stack)."""
    torch.manual_seed(SEED)
    topo = topology_kwargs(N)
    topo["K_iter"] = K_ITER
    topo["n_groups"] = min(128, N // 8)

    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=10, D=D, N_in=25088,
        K_in=topo["K_in"],
        K_local=topo["K_local"],
        K_random=topo["K_random"],
        n_groups=topo["n_groups"],
        K_iter=K_ITER,
        sparsity=0.90,
        box_size=1.0,
        norm_mode="l2",
        encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base=base,
        K_phase=8,
        beam_size=32,
        theta_init=0.1,
        alpha_reflect=0.5,
        alpha_turing=0.3,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
        geo_gamma=1.0,
    )
    return SGNNET_AntiHebbian(base=resonant, alpha_ahebb=alpha_ahebb, variant="wpos")


def make_reflection_model(
    alpha_reflect: float,
    theta: float,
    alpha_ahebb: float = 0.7,
) -> SGNNET_Reflection:
    """Build SmallWorld + Reflection (no Resonant wrapper; reflection replaces routing)."""
    torch.manual_seed(SEED)
    topo = topology_kwargs(N)
    topo["K_iter"] = K_ITER
    topo["n_groups"] = min(128, N // 8)

    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=10, D=D, N_in=25088,
        K_in=topo["K_in"],
        K_local=topo["K_local"],
        K_random=topo["K_random"],
        n_groups=topo["n_groups"],
        K_iter=K_ITER,
        sparsity=0.90,
        box_size=1.0,
        norm_mode="l2",
        encoding_mode="fourier",
    )
    return SGNNET_Reflection(base=base, alpha_reflect=alpha_reflect, theta=theta)


# -- Run helper ----------------------------------------------------------------

def run(label: str, model: torch.nn.Module, meta: dict, is_reflection: bool = False) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  N={N}  D={D}  K_iter={K_ITER}  params={n_params:,}")

    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)

    dead_fracs_by_epoch: list[float] = []

    def log_fn(metrics: dict):
        ep = metrics["epoch"] + 1
        if is_reflection and hasattr(model, "dead_neuron_report"):
            report = model.dead_neuron_report()
            dead_fracs_by_epoch.append(report["max_dead_frac"])
            if (ep % 10 == 0 or ep == 1) and report["warning"]:
                print(f"    [WARN] ep{ep}: dead_frac={report['max_dead_frac']:.1%} >5%")

    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS, log_fn=log_fn)
    elapsed = time.time() - t0

    best_top1 = max(h["val_top1"] for h in history)
    best_ep   = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac      = best_ep / EPOCHS
    ms_per_ep = 1000 * elapsed / max(len(history), 1)

    # Compute dead neuron summary
    if dead_fracs_by_epoch:
        max_dead  = max(dead_fracs_by_epoch)
        mean_dead = sum(dead_fracs_by_epoch) / len(dead_fracs_by_epoch)
        dead_warning = max_dead > 0.05
    else:
        max_dead = mean_dead = 0.0
        dead_warning = False

    print(f"  top1_best={best_top1:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})"
          f"  elapsed={elapsed:.0f}s")
    if is_reflection:
        print(f"  dead_neuron max={max_dead:.1%}  mean={mean_dead:.1%}"
              f"  {'[DYING CASCADE RISK]' if dead_warning else '[OK]'}")

    return {
        "label":           label,
        "N":               N,
        "D":               D,
        "K_iter":          K_ITER,
        "params":          n_params,
        "top1":            best_top1,
        "best_ep":         best_ep,
        "ep_frac":         frac,
        "ms_per_epoch":    ms_per_ep,
        "total_time_s":    elapsed,
        "epochs_run":      len(history),
        "is_reflection":   is_reflection,
        "dead_max_frac":   max_dead,
        "dead_mean_frac":  mean_dead,
        "dead_warning":    dead_warning,
        "_meta":           run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }


# -- Configs -------------------------------------------------------------------
# Config A: pure AntiHebb reference (no reflection)
# Config B: leaky-reflect (low strength, all negatives)
# Config C: hard-reflect (full strength, only strongly negative)
# Config D: medium-reflect (alpha=0.3, theta=0.0 — intermediate)

CONFIGS = [
    # (key, label, builder_fn, is_reflection)
    (
        "A",
        "Config A: SmallWorld + AntiHebb alpha=0.7 (reference, no reflection)",
        lambda: make_base_antihebb(alpha_ahebb=0.7),
        False,
        {"mechanism": "antihebb_wpos", "alpha_ahebb": 0.7, "reflection": False},
    ),
    (
        "B",
        "Config B: SmallWorld + leaky-reflect (alpha_reflect=0.1, theta=0.0)",
        lambda: make_reflection_model(alpha_reflect=0.1, theta=0.0),
        True,
        {"mechanism": "reflection", "alpha_reflect": 0.1, "theta": 0.0},
    ),
    (
        "C",
        "Config C: SmallWorld + hard-reflect (alpha_reflect=1.0, theta=0.5)",
        lambda: make_reflection_model(alpha_reflect=1.0, theta=0.5),
        True,
        {"mechanism": "reflection", "alpha_reflect": 1.0, "theta": 0.5},
    ),
    (
        "D",
        "Config D: SmallWorld + medium-reflect (alpha_reflect=0.3, theta=0.0)",
        lambda: make_reflection_model(alpha_reflect=0.3, theta=0.0),
        True,
        {"mechanism": "reflection", "alpha_reflect": 0.3, "theta": 0.0},
    ),
]


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Goal: ablate reflection routing vs AntiHebb baseline at N=1024 D=64")
    print("Key metric: top1 accuracy + dead neuron fraction (must stay < 5%)")

    results = {}
    for key, label, build_fn, is_refl, meta in CONFIGS:
        model = build_fn()
        results[key] = run(label, model, meta, is_reflection=is_refl)

    # -- Save results --
    out_path = ROOT / "results" / "exp4_reflection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # -- Summary table --
    print(f"\n-- Reflection Ablation Results -----------------------------------------")
    print(f"  {'Config':<60}  {'top1':>6}  {'dead_max':>8}  {'dead_warn':>9}")
    print("  " + "-" * 90)
    for key, label, _, _, _ in CONFIGS:
        r = results[key]
        warn = "[WARN]" if r["dead_warning"] else "OK"
        print(f"  {label:<60}  {r['top1']:.4f}  {r['dead_max_frac']:>8.1%}  {warn:>9}")
