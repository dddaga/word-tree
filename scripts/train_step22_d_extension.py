"""Step 22: D (encoding dimension) extension sweep at optimal K_iter=8.

MOTIVATION
==========
Results so far:
  D sweep  (N=512,  K_iter=3, 120ep): D=4→21%, D=8→26%, D=16→27.54% — rising, no plateau
  N=1024   (D=16,   K_iter=3, 120ep): 34.32% (N=512: 28.56%)
  K_iter=8 (D=16,   N=512,   120ep): 36.69% (K_iter=3: 29.04%)

The D sweep was done at K_iter=3 only. K_iter=8 × N=1024 × D>16 is completely untested.
Key insight: larger D gives each neuron richer state space; K_iter=8 allows that state
to interact across the routing graph long enough to matter.

HYPOTHESIS
==========
D=32 or D=64 at K_iter=8 × N=1024 will break 38% — the encoding dimension has been the
bottleneck that K_iter=3 couldn't exploit.

Also tests training duration: K_iter=8 peaked at epoch 63/120 with LR-plateau schedule.
Using 150ep + more patient plateau gives the model time to fully converge.

CONFIGS
=======
  Ref : D=16, N=512,  K_iter=8  (reproduces step13 K_iter=8 result — 36.69% expected)
  A   : D=16, N=1024, K_iter=8  (first time this combo is tested)
  B   : D=32, N=512,  K_iter=8
  C   : D=32, N=1024, K_iter=8  ← primary hypothesis
  D   : D=64, N=512,  K_iter=8
  E   : D=64, N=1024, K_iter=8  ← primary hypothesis
  F   : D=16, N=2048, K_iter=8  (double-wide, same depth as A)

All: Fourier encoding, dynamic_z_geo, 150ep, plateau, store.h5.

Reference: step13 K_iter=8 = 36.69%  (best_epoch=63 → training was ending too soon)

To reproduce:
    python -u scripts/train_step22_d_extension.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 150
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model(N: int, D: int, K_iter: int) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


def run(label: str, model: SGNNET_Resonant, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    N       = meta["N"]
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    # Training-too-short diagnostic
    frac = best_ep / len(history)
    diag = "training_too_short" if frac < 0.7 else "converged"
    result["convergence_diag"] = diag
    result["best_epoch_frac"]  = round(frac, 3)
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})  "
          f"diag={diag}  task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# (label, N, D, K_iter)
CONFIGS = [
    ("Ref. D=16 N=512  K_iter=8  [step13 baseline]", 512,  16, 8),
    ("A.  D=16 N=1024 K_iter=8  [wider, same depth]", 1024, 16, 8),
    ("B.  D=32 N=512  K_iter=8  [deeper encoding]",   512,  32, 8),
    ("C.  D=32 N=1024 K_iter=8  [primary hypothesis]",1024, 32, 8),
    ("D.  D=64 N=512  K_iter=8  [very deep encoding]",512,  64, 8),
    ("E.  D=64 N=1024 K_iter=8  [primary hypothesis]",1024, 64, 8),
    ("F.  D=16 N=2048 K_iter=8  [double-wide]",       2048, 16, 8),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}")
    print("Goal: D extension at optimal K_iter=8 — is D=16 a ceiling?")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys    = ["Ref", "A", "B", "C", "D", "E", "F"]
    for key, (label, N, D, K_iter) in zip(keys, CONFIGS):
        model = make_model(N, D, K_iter).to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": K_iter}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step22_d_extension.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results.get("Ref", {}).get("top1_best", 0.3669)
    print(f"\n-- D extension sweep at K_iter=8  (ref={ref:.4f}) -------------------")
    print("  %-52s  %5s  %5s  %6s  %9s  %+8s  %8s  %6s" % (
        "Config", "N", "D", "K_it", "top1_best", "vs_ref", "best_ep%", "t(s)"))
    print("  " + "-"*105)
    for k, r in results.items():
        d = r["top1_best"] - ref
        print("  %-52s  %5d  %5d  %6d  %9.4f  %+8.4f  %7.1f%%  %6.0f" % (
            r["label"][:52], r.get("N", 0), r.get("D", 0), r.get("K_iter", 0),
            r["top1_best"], d, r.get("best_epoch_frac", 0) * 100, r["elapsed_s"]))
