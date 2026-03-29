"""Step 6 diagnostic: plain SGNNET_SmallWorld baseline (no Resonant wrapper).

Hypothesis:
  All Step 5 diagnostics used SGNNET_Resonant(dynamic_z). None tested whether
  the ~20% ceiling is in the SmallWorld base model or in the Resonant routing layer.

  Iter1 (train_resonant.py) ran BOTH:
    baseline_N512 (plain SmallWorld):  24.38%
    dynamic_z_N512 (with Resonant):    26.52%

  If this run gives ~24%, the SmallWorld base is intact and regression is in Resonant.
  If this run gives ~20%, the regression is in SmallWorld or training setup (data, optimizer).

Config: exact iter1 baseline_N512 replication.
  Model: SGNNET_SmallWorld(N_in=25088, N_hidden=512, N_out=10, D=4, norm_mode="l2")
  Trainer: trainer_kwargs(512, sched_type="plateau"), 120ep, store.h5
  Seed: 42 (same as all other diagnostics)

To reproduce:
    python -u scripts/train_step6_baseline_control.py --device mps
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.training.trainer        import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs
from src.training.dataset        import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 120
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Seed: {SEED}")
print(f"Model: plain SGNNET_SmallWorld (no Resonant wrapper) — iter1 baseline replication")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model() -> nn.Module:
    """Exact replica of iter1 baseline_N512 — SmallWorld, no routing."""
    torch.manual_seed(SEED)
    tk = topology_kwargs(512)
    return SGNNET_SmallWorld(
        N_in=25088, N_hidden=512, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=4,
        encoding_mode="linear",
    )


if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    label = "SmallWorld baseline  N=512 D=4 plateau 120ep  (iter1 baseline_N512 replica)"
    print(f"\n{'='*65}\n{label}\n{'='*65}")

    tr_loader, va_loader = get_loaders()
    model = make_model().to(DEVICE)
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
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
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")

    out = Path("results/train_step6_baseline_control.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"baseline_control": result}, indent=2))
    print(f"\nSaved -> {out}")

    iter1_ref = 0.2438   # iter1 baseline_N512
    dyn_z_ref = 0.2652   # iter1 dynamic_z_N512
    delta = result["top1_best"] - iter1_ref
    print(f"\n  SmallWorld baseline (no routing): {result['top1_best']:.4f}")
    print(f"  Iter1 baseline_N512 reference:    {iter1_ref:.4f}")
    print(f"  Iter1 dynamic_z reference:        {dyn_z_ref:.4f}")
    if delta >= -0.02:
        print(f"  -> MATCHES iter1 (delta={delta:+.4f}). SmallWorld is intact.")
        print(f"  -> Regression is in SGNNET_Resonant dynamic_z routing layer.")
        print(f"  -> Investigate: model_resonant.py, wave_routing.py changes.")
    elif delta >= -0.05:
        print(f"  -> Partial recovery (delta={delta:+.4f}). Regression is mixed.")
    else:
        print(f"  -> Still stuck (delta={delta:+.4f}). Regression is in SmallWorld or trainer.")
        print(f"  -> Investigate: model_smallworld.py, trainer.py, or seed differences.")
