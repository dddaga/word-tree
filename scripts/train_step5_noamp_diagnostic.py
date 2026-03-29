"""Step 5 diagnostic: disable AMP/GradScaler to match iter1 training regime.

Hypothesis:
  Iter1 (train_resonant.py, 2026-03-27) achieved 26.52% on dynamic_z.
  All post-refactor runs are stuck at ~19-20%.

  _check_grad_scaler_support() returns True for PyTorch >= 2.3.
  If Mac Studio has PyTorch >= 2.3 → GradScaler is active (fp16 + scaling).
  If iter1 ran on Mac Mini with PyTorch < 2.3 → scaler=None (fp32, no GradScaler).

  With GradScaler active, any float16 overflow/nan in MPS ops causes GradScaler
  to reduce scale and skip update steps. Repeated skipping effectively kills learning.
  Training without GradScaler (use_amp=False) forces fp32 throughout — identical to
  what iter1 would have done if PyTorch was < 2.3.

  This test isolates whether GradScaler on MPS is the hidden cause of the ~19-20% ceiling.

Config: config D only (dynamic_z, no geo, no thresh) — iter1 control at 26.52%.
  Sched: plateau (iter1 default)
  Clip: inf (no clipping — already set in experiment_config)
  use_amp: False (force fp32, no GradScaler) ← key change vs all previous runs

To reproduce:
    python -u scripts/train_step5_noamp_diagnostic.py --device mps
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

# Fixed hyperparameters
EPOCHS = 120
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
SCHED  = "plateau"

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Sched: {SCHED}")
print(f"use_amp: False (fp32 only, no GradScaler — testing iter1 AMP hypothesis)")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model_d() -> nn.Module:
    """Config D: pure dynamic_z, no geo, no thresh — iter1 control."""
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
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z",
        resonance_threshold=0.0,
    )


def run_config_d() -> dict:
    label = "D. thresh=0.0  geo=False  dynamic_z  N=512 D=4  fp32 plateau 120ep"
    print(f"\n{'='*65}\n{label}\n{'='*65}")
    tr_loader, va_loader = get_loaders()
    model = make_model_d().to(DEVICE)
    # Override use_amp=False to disable GradScaler — key change
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type=SCHED)
    tk["use_amp"] = False   # force fp32, no GradScaler
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":           label,
        "use_amp":         False,
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


if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    result = run_config_d()

    out = Path("results/train_step5_noamp_diagnostic.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"D_noamp": result}, indent=2))
    print(f"\nSaved -> {out}")

    iter1_ref = 0.2652
    print(f"\n  Config D (fp32, no AMP, plateau): {result['top1_best']:.4f}")
    print(f"  Iter1 reference (dynamic_z):      {iter1_ref:.4f}")
    delta = result["top1_best"] - iter1_ref
    if delta >= -0.02:
        print(f"  -> MATCHES iter1 (delta={delta:+.4f}). GradScaler was the hidden issue.")
        print(f"  -> Fix: set use_amp=False or investigate MPS float16 stability.")
    elif delta >= -0.05:
        print(f"  -> Partial recovery (delta={delta:+.4f}). AMP is a factor but not the only one.")
    else:
        print(f"  -> Still stuck (delta={delta:+.4f}). AMP is NOT the primary cause.")
        print(f"  -> Investigate: data pipeline diff, model arch change, or optimizer config.")
