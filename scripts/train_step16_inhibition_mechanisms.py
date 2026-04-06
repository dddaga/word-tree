"""Step 16: Divisive normalization, refractory inhibition, anti-Hebbian suppression.

Three lateral inhibition mechanisms tested at D=16 Fourier, each built as a wrapper
around SGNNET_Resonant. Mechanism implementations in:
    src/sgnnet/mechanisms_inhibitory.py

CONFIGS (all: D=16 Fourier N=512 dynamic_z_geo 120ep plateau store.h5 MPS):
  A. DivNorm  alpha=0.5  — mild lateral shunting
  B. DivNorm  alpha=1.0  — moderate shunting
  C. DivNorm  alpha=2.0  — strong shunting
  D. Refract  beta=0.9  alpha_r=1.0  — slow decay, mild suppression
  E. Refract  beta=0.7  alpha_r=2.0  — medium decay, strong suppression
  F. Refract  beta=0.5  alpha_r=3.0  — fast decay, strong suppression
  G. AntiHebb alpha=0.3 wpos — W_pos spatial surround suppression
  H. AntiHebb alpha=0.5 wpos — stronger spatial surround
  I. AntiHebb alpha=0.3 zact — current-Z feature decorrelation

Reference: step9A 29.22% (D=16 Fourier N=512 dynamic_z_geo 150ep)

To reproduce:
    python -u scripts/train_step16_inhibition_mechanisms.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import (
    SGNNET_DivisiveNorm, SGNNET_Refractory, SGNNET_AntiHebbian
)
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

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


def run(label: str, model: nn.Module, extra_meta: dict) -> dict:
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
        "_meta":           run_metadata(__file__, {"D": D, "N": N, "epochs": EPOCHS, **extra_meta}),
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# ── Config table ──────────────────────────────────────────────────────────────
# (label, ModelClass, model_kwargs, result_meta)
CONFIGS = [
    ("A. DivNorm  alpha=0.5  [mild shunting]",
     SGNNET_DivisiveNorm, {"alpha_div": 0.5},                   {"mech": "divnorm", "alpha_div": 0.5}),
    ("B. DivNorm  alpha=1.0  [moderate shunting]",
     SGNNET_DivisiveNorm, {"alpha_div": 1.0},                   {"mech": "divnorm", "alpha_div": 1.0}),
    ("C. DivNorm  alpha=2.0  [strong shunting]",
     SGNNET_DivisiveNorm, {"alpha_div": 2.0},                   {"mech": "divnorm", "alpha_div": 2.0}),
    ("D. Refract  beta=0.9  alpha_r=1.0  [slow+mild]",
     SGNNET_Refractory,   {"beta": 0.9, "alpha_refract": 1.0},  {"mech": "refractory", "beta": 0.9, "alpha_r": 1.0}),
    ("E. Refract  beta=0.7  alpha_r=2.0  [medium+strong]",
     SGNNET_Refractory,   {"beta": 0.7, "alpha_refract": 2.0},  {"mech": "refractory", "beta": 0.7, "alpha_r": 2.0}),
    ("F. Refract  beta=0.5  alpha_r=3.0  [fast+strong]",
     SGNNET_Refractory,   {"beta": 0.5, "alpha_refract": 3.0},  {"mech": "refractory", "beta": 0.5, "alpha_r": 3.0}),
    ("G. AntiHebb  alpha=0.3  wpos  [spatial surround]",
     SGNNET_AntiHebbian,  {"alpha_ahebb": 0.3, "variant": "wpos"}, {"mech": "ahebb", "alpha": 0.3, "variant": "wpos"}),
    ("H. AntiHebb  alpha=0.5  wpos  [stronger spatial]",
     SGNNET_AntiHebbian,  {"alpha_ahebb": 0.5, "variant": "wpos"}, {"mech": "ahebb", "alpha": 0.5, "variant": "wpos"}),
    ("I. AntiHebb  alpha=0.3  zact  [feature decorrelation]",
     SGNNET_AntiHebbian,  {"alpha_ahebb": 0.3, "variant": "zact"}, {"mech": "ahebb", "alpha": 0.3, "variant": "zact"}),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}  encoding=fourier")
    print("Goal: divisive normalization / refractory inhibition / anti-Hebbian")

    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    results = {}
    for key, (label, Model, mkwargs, meta) in zip("ABCDEFGHI", CONFIGS):
        resonant = make_base_resonant().to(DEVICE)
        model    = Model(resonant, **mkwargs)
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step16_inhibition_mechanisms.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = 0.2922
    print(f"\n-- All inhibition mechanisms  (ref step9A={ref:.4f}) ---------------------")
    print("  %-55s  %9s  %+8s  %6s" % ("Config", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*82)
    for k, r in results.items():
        d = r["top1_best"] - ref
        print("  %-55s  %9.4f  %+8.4f  %6.0f" % (k, r["top1_best"], d, r["elapsed_s"]))

    for desc, keys in [("Divisive Normalization", "ABC"),
                       ("Refractory Inhibition",  "DEF"),
                       ("Anti-Hebbian",            "GHI")]:
        print(f"\n  {desc}:")
        for k in keys:
            if k in results:
                r = results[k]
                print(f"    {k}: top1={r['top1_best']:.4f}  (delta={r['top1_best']-ref:+.4f})"
                      f"  {r['label']}")
