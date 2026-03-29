"""Resonance threshold sweep — isolate regression from iter1 26.52%.

Iter1 used dynamic_z (no threshold, no geo). Iter2 added geo+threshold together
and got 24.25% on CPU/original data. Aug baseline (18.7%) uses geo+thresh=0.3
on aug data. This sweep isolates: is the threshold, or the geo bias, or the
combination causing the regression?

Configs (dynamic_z_geo, N=512, D=4, cosine, 120ep, store_aug.h5, MPS):
  A. thresh=0.0  geo=True   — geo bias only,  no threshold gate
  B. thresh=0.1  geo=True   — light threshold
  C. thresh=0.3  geo=True   — current default (already ~18.7%, sanity check)
  D. thresh=0.0  geo=False  — pure dynamic_z (closest to iter1 winner)

D is the critical control: if it recovers to ~25%+ we know it's the geo or threshold
causing the regression. If D is also ~18-19%, the regression is from the aug data itself
(hflip features shifting the resonance distribution).
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch, torch.nn as nn

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
from src.training.trainer        import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs
from src.training.dataset        import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--data",   default="data/store_aug.h5")
parser.add_argument("--sched",  default="cosine", choices=["cosine", "plateau", "none"])
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
print(f"Device: {DEVICE}  Epochs: {EPOCHS}")

# Loaded once at startup — reused across all configs (no repeated disk I/O)
_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(args.data, batch_size=BATCH, seed=SEED)
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
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type=args.sched)
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
    ("A. thresh=0.0  geo=True   dynamic_z_geo  N=512 D=4 aug 120ep",  0.0, True),
    ("B. thresh=0.1  geo=True   dynamic_z_geo  N=512 D=4 aug 120ep",  0.1, True),
    ("C. thresh=0.3  geo=True   dynamic_z_geo  N=512 D=4 aug 120ep",  0.3, True),   # sanity check
    ("D. thresh=0.0  geo=False  dynamic_z      N=512 D=4 aug 120ep",  0.0, False),  # iter1 control
]

if __name__ == "__main__":
    # Load full dataset into RAM once before the sweep
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, 256GB RAM)")

    results = {}
    for key, (label, thresh, use_geo) in zip("ABCD", CONFIGS):
        results[key] = run(label, thresh, use_geo)

    out = Path("results/train_thresh_sweep.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    base = results["D"]["top1_best"]   # pure dynamic_z control
    print("\n-- Threshold / geo sweep ------------------------------------------")
    print("  %-65s  %9s  %8s  %6s" % ("Config", "top1_best", "vs iter1-ctrl", "t(s)"))
    print("  " + "-"*65 + "  " + "-"*9 + "  " + "-"*8 + "  " + "-"*6)
    for k, r in results.items():
        delta = r["top1_best"] - base
        print("  %-65s  %9.4f  %+8.4f  %6.0f" % (
            k, r["top1_best"], delta, r["elapsed_s"]))

    print("\n  Key question — config D (pure dynamic_z, no thresh, no geo):")
    if results["D"]["top1_best"] >= 0.24:
        print("  -> Recovers to %.2f%%. Regression IS from geo/thresh, NOT aug data." %
              (results["D"]["top1_best"] * 100))
        best_k = max(results, key=lambda k: results[k]["top1_best"])
        print("  -> Best overall: config %s (%.4f). Use this as new baseline." %
              (best_k, results[best_k]["top1_best"]))
    else:
        print("  -> Still %.2f%%. Regression is from aug data distribution shift." %
              (results["D"]["top1_best"] * 100))
        print("  -> hflip augmentation may be changing VGG feature distribution.")
        print("  -> Consider: train on original data only, or reduce hflip fraction.")
