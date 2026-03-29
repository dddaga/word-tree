"""Step 5 diagnostic: plateau LR + no grad-clip to reproduce iter1 (26.52%).

Hypothesis:
  All post-refactor runs used cosine or constant LR. Iter1 (train_resonant.py)
  used the trainer_kwargs() default: sched_type="plateau". This experiment
  isolates whether plateau + no-clip recovers iter1 performance.

Reference:
  iter1 (train_resonant.py, plateau, no-clip, 120ep, store.h5):
    baseline_N512:    24.38%
    resonant_N512:    25.32%
    dynamic_gate_N512:25.07%
    dynamic_z_N512:   26.52%  ← target for config D here

Prior diagnostics that failed to recover:
  clip=1.0  + cosine  + 120ep: D=19.62%
  clip=1.0  + sched=none + 90ep: D=20.08%
  clip=inf  + cosine  + 120ep: (not tested)
  clip=inf  + sched=none + 90ep: D=19.87%
  clip=inf  + plateau + 120ep: THIS RUN ← expected to match iter1

Configs (same 4-config sweep as train_thresh_sweep.py):
  A. thresh=0.0  geo=True   dynamic_z_geo  — add-on to iter1 control
  B. thresh=0.1  geo=True   dynamic_z_geo
  C. thresh=0.3  geo=True   dynamic_z_geo
  D. thresh=0.0  geo=False  dynamic_z      ← iter1 control; expect ~26%+

All configs: N=512, D=4, linear encoding, store.h5, MPS, 120ep, plateau LR.
experiment_config.trainer_kwargs now has grad_clip_norm=inf (no clipping).

To reproduce:
    python -u scripts/train_step5_plateau_noclip.py --device mps
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
from src.training.trainer        import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs
from src.training.dataset        import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

# Fixed hyperparameters — not CLI args to ensure exact reproducibility
EPOCHS = 120
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
SCHED  = "plateau"   # must match iter1 default

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Sched: {SCHED}  Data: {DATA}")
print(f"grad_clip_norm: inf (no clipping) — set via experiment_config")

# Load full dataset into RAM once — reused across all configs
_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model(thresh: float, use_geo: bool) -> nn.Module:
    torch.manual_seed(SEED)
    tk = topology_kwargs(512)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=512, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=4,
        encoding_mode="linear",
    )
    mode = "dynamic_z_geo" if use_geo else "dynamic_z"
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode=mode,
        resonance_threshold=thresh,
        geo_gamma=1.0,
    )


def run(label: str, thresh: float, use_geo: bool) -> dict:
    print(f"\n{'='*65}\n{label}\n{'='*65}")
    tr_loader, va_loader = get_loaders()
    model = make_model(thresh, use_geo).to(DEVICE)
    # sched_type="plateau" to match iter1 — no cosine/none
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type=SCHED)
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":           label,
        "thresh":          thresh,
        "use_geo":         use_geo,
        "sched":           SCHED,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch":      int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1,
        "epochs_run":      len(history),
        "elapsed_s":       round(elapsed, 1),
        "top1_history":    [round(h.get("val_top1", 0.0), 4) for h in history],
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


CONFIGS = [
    ("A. thresh=0.0  geo=True   dynamic_z_geo  N=512 D=4 plateau 120ep",  0.0, True),
    ("B. thresh=0.1  geo=True   dynamic_z_geo  N=512 D=4 plateau 120ep",  0.1, True),
    ("C. thresh=0.3  geo=True   dynamic_z_geo  N=512 D=4 plateau 120ep",  0.3, True),
    ("D. thresh=0.0  geo=False  dynamic_z      N=512 D=4 plateau 120ep",  0.0, False),
]

if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    results = {}
    for key, (label, thresh, use_geo) in zip("ABCD", CONFIGS):
        results[key] = run(label, thresh, use_geo)

    out = Path("results/train_step5_plateau_noclip.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    base = results["D"]["top1_best"]
    print("\n-- Plateau + no-clip sweep (vs iter1 target: D~26.5%) ------------------")
    print("  %-65s  %9s  %8s  %6s" % ("Config", "top1_best", "vs_iter1", "t(s)"))
    print("  " + "-"*65 + "  " + "-"*9 + "  " + "-"*8 + "  " + "-"*6)
    iter1_d = 0.2652
    for k, r in results.items():
        delta = r["top1_best"] - iter1_d
        print("  %-65s  %9.4f  %+8.4f  %6.0f" % (
            k, r["top1_best"], delta, r["elapsed_s"]))

    print(f"\n  Config D (pure dynamic_z): {base:.4f}")
    if base >= 0.255:
        print(f"  -> MATCHES iter1 ({iter1_d:.4f}). plateau+no-clip is the fix.")
        print(f"  -> Use sched_type='plateau' + no clip for all ablations.")
    elif base >= 0.23:
        print(f"  -> Partial recovery (target {iter1_d:.4f}). Improved significantly.")
    else:
        print(f"  -> Still stuck at {base:.2%}. Plateau is not the explanation.")
        print(f"  -> Investigate: model architecture change, data pipeline, or seed.")
