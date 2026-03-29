"""Step 9: D=16 Fourier encoding deep run — establish new performance ceiling.

Motivation:
  Encoding sweep (train_encoding_D_sweep.py, 90ep) found D=16 Fourier achieves 27.54%
  at N=512, surpassing iter1's best (26.52%). This 150-epoch deep run:
    1. Confirms the D=16 Fourier advantage at full convergence
    2. Compares dynamic_z vs dynamic_z_geo to pick the routing mode for Group A/B/C
    3. Tests D=8 N=1024 with more epochs (90ep may have underconverged)

Reference:
  Encoding sweep (90ep):
    fourier_D16_N512  (dynamic_z_geo): 27.54%
    fourier_D8_N1024  (dynamic_z_geo): 26.88%
    fourier_D8_N512   (dynamic_z_geo): 25.86%
    linear_D4_N512    (dynamic_z_geo): 20.99%

  Iter1 rerun (120ep, dynamic_z, linear D=4):
    dynamic_z_N512: 23.18%

Configs:
  A. D=16 Fourier N=512  dynamic_z_geo  (encoding sweep winner, more epochs)
  B. D=16 Fourier N=512  dynamic_z      (no geo bias — cleaner routing)
  C. D=8  Fourier N=1024 dynamic_z_geo  (scale check at more epochs)

All: 150ep, plateau LR, store.h5, MPS.

To reproduce:
    python -u scripts/train_step9_d16_deep.py --device mps
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

EPOCHS = 150
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Seed: {SEED}")
print(f"Goal: establish D=16 Fourier ceiling and compare dynamic_z vs dynamic_z_geo")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model(n_hidden: int, D: int, mode: str, encoding: str = "fourier") -> nn.Module:
    torch.manual_seed(SEED)
    tk = topology_kwargs(n_hidden)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=n_hidden, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=D,
        encoding_mode=encoding,
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode=mode,
        resonance_threshold=0.0,
        geo_gamma=1.0,
    )


def run(label: str, model: nn.Module, n_hidden: int) -> dict:
    print(f"\n{'='*65}\n{label}\n{'='*65}")
    tr_loader, va_loader = get_loaders()
    tk = trainer_kwargs(n_hidden, n_epochs=EPOCHS, sched_type="plateau")
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
    return result


CONFIGS = [
    ("A. Fourier D=16 N=512  dynamic_z_geo  150ep  [encoding sweep winner]",
     512, 16, "dynamic_z_geo"),
    ("B. Fourier D=16 N=512  dynamic_z      150ep  [no geo bias]",
     512, 16, "dynamic_z"),
    ("C. Fourier D=8  N=1024 dynamic_z_geo  150ep  [scale check]",
     1024, 8, "dynamic_z_geo"),
]

if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    results = {}
    for key, (label, n_hidden, D, mode) in zip("ABC", CONFIGS):
        model = make_model(n_hidden, D, mode).to(DEVICE)
        results[key] = run(label, model, n_hidden)

    out = Path("results/train_step9_d16_deep.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    enc_ref = 0.2754   # fourier D=16 N=512 at 90ep
    print("\n-- D=16 deep run (vs encoding sweep 90ep ref: 27.54%) ----------------------")
    print("  %-67s  %9s  %8s  %6s" % ("Config", "top1_best", "vs_enc", "t(s)"))
    print("  " + "-"*67 + "  " + "-"*9 + "  " + "-"*8 + "  " + "-"*6)
    for k, r in results.items():
        delta = r["top1_best"] - enc_ref
        print("  %-67s  %9.4f  %+8.4f  %6.0f" % (
            k, r["top1_best"], delta, r["elapsed_s"]))

    best = max(results.values(), key=lambda r: r["top1_best"])
    print(f"\n  New best: {best['label']}")
    print(f"  top1_best={best['top1_best']:.4f}")
    print(f"  Encoding sweep 90ep ref: {enc_ref:.4f}  iter1 ref: 0.2652")
