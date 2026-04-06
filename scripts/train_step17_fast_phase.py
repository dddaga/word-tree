"""Step 17: Fast W_phase — proper intra-forward adaptation of phase anchors.

MECHANISM
=========
W_phase is never adapted in the dynamic_z_geo routing loop (confirmed dead path).
Here we treat W_phase as a 'slow prior': cloned to local A at each forward pass,
A adapted by a Hebbian rule within K_iter routing steps, then discarded.
A is input-specific; gradient descent on W_phase is unchanged (shapes the prior).

This is the Ba et al. (2016) fast-weight architecture applied to routing.

fast_rule controls the update:
  oja_shared     : A[N,D] updated toward dominant Z directions (Oja, batch-shared)
  oja_per_sample : A[B,N,D] per-sample — fully input-specific
  hopfield       : Z attracted toward stored A directions (no A update)
  attention      : A updated to Z-cluster centroids; Z retrieved via attention

Reference: step9A 29.22% (D=16 Fourier N=512 dynamic_z_geo 150ep)

All configs: D=16 Fourier N=512 dynamic_z_geo 120ep plateau store.h5 MPS.

To reproduce:
    python -u scripts/train_step17_fast_phase.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.mechanisms_excitatory import SGNNET_FastPhase
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset            import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 120
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 16
N      = 512

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_base_resonant() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":           label,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch":      int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1,
        "epochs_run":      len(history),
        "elapsed_s":       round(elapsed, 1),
        "top1_history":    [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta":           run_metadata(__file__, {"D": D, "N": N, "epochs": EPOCHS, **meta}),
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# (label, fast_rule, alpha_fast, tau, beam_att, meta)
CONFIGS = [
    ("Ref. dynamic_z_geo no fast-phase  [baseline]",
     None, 0.0, 0.25, 0, {"rule": "none"}),
    ("A. Oja shared  α=0.1  [stable Hebbian]",
     "oja_shared",     0.1, 0.25,  0, {"rule": "oja_shared",     "alpha": 0.1}),
    ("B. Oja shared  α=0.3  [stronger Oja]",
     "oja_shared",     0.3, 0.25,  0, {"rule": "oja_shared",     "alpha": 0.3}),
    ("C. Oja per-sample  α=0.1  [input-specific]",
     "oja_per_sample", 0.1, 0.25,  0, {"rule": "oja_per_sample", "alpha": 0.1}),
    ("D. Hopfield attract  α=0.1  [Z pulled to A]",
     "hopfield",       0.1, 0.25,  0, {"rule": "hopfield",       "alpha": 0.1}),
    ("E. Hopfield attract  α=0.3  [stronger attract]",
     "hopfield",       0.3, 0.25,  0, {"rule": "hopfield",       "alpha": 0.3}),
    ("F. Attention update  τ=0.25  beam=32  [sparse attention]",
     "attention",      0.1, 0.25, 32, {"rule": "attention",      "alpha": 0.1, "tau": 0.25}),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}  encoding=fourier")
    print("Goal: fast W_phase — intra-forward Hebbian adaptation of phase anchors")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys = ["Ref", "A", "B", "C", "D", "E", "F"]
    for key, (label, rule, alpha, tau, beam_att, meta) in zip(keys, CONFIGS):
        resonant = make_base_resonant().to(DEVICE)
        model    = resonant if rule is None else \
                   SGNNET_FastPhase(resonant, fast_rule=rule, alpha_fast=alpha,
                                    tau=tau, beam_att=beam_att)
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step17_fast_phase.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results.get("Ref", {}).get("top1_best", 0.2922)
    print(f"\n-- Fast W_phase experiments (ref={ref:.4f}) --------------------------")
    print("  %-55s  %9s  %+8s  %6s" % ("Config", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*82)
    for k, r in results.items():
        d = r["top1_best"] - ref
        print("  %-55s  %9.4f  %+8.4f  %6.0f" % (k, r["top1_best"], d, r["elapsed_s"]))
