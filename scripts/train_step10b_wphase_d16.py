"""Step 10b: W_phase receiver re-test at D=16 Fourier.

At D=4, resonant mode (W_phase-based phase graph) gave -0.2% to -1.1% vs baseline.
Conclusion: static random phase graph = noise at D=4.

At D=16, W_phase lives in a 16-dim space — genuinely distinct directions are possible.
Two tests:
  E. resonant mode, W_phase static  (NOT in optimizer, stays at random init)
     → Controls for "is D=16 enough to make a random phase graph useful?"
  F. resonant mode, W_phase learned (lr_wphase=2.36e-3, same as W_pos)
     → "Can W_phase learn a meaningful phase graph at D=16?"
  G. dynamic_z_geo, D=16, full routing  (vs step10a config D — adds geo bias)
     → "Does geo bias help/hurt at D=16 dynamic_z with full routing?"

All: D=16 Fourier N=512, 120ep, plateau, store.h5, MPS.
Reference: encoding sweep D=16 dynamic_z_geo = 27.54% (90ep).

To reproduce:
    python -u scripts/train_step10b_wphase_d16.py --device mps
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
from src.training.experiment_config import trainer_kwargs, topology_kwargs, GA_BEST
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
D      = 16

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  encoding=fourier")
print(f"Goal: W_phase static vs learned at D=16; geo bias effect at D=16")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_base() -> SGNNET_SmallWorld:
    tk = topology_kwargs(512)
    return SGNNET_SmallWorld(
        N_in=25088, N_hidden=512, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=D,
        encoding_mode="fourier",
    )


def run(label: str, model: nn.Module, lr_wphase: float | None = None) -> dict:
    print(f"\n{'='*65}\n{label}\n{'='*65}")
    tr_loader, va_loader = get_loaders()
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type="plateau")
    if lr_wphase is not None:
        tk["lr_wphase"] = lr_wphase
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":           label,
        "lr_wphase":       lr_wphase,
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

    results = {}

    # E. Resonant mode, W_phase static (not learned — original rejected config)
    torch.manual_seed(SEED)
    model_e = SGNNET_Resonant(
        base=make_base(),
        K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="resonant",   # uses W_phase-built graph, not dynamic Z
        resonance_threshold=0.0,
    ).to(DEVICE)
    results["E"] = run(
        "E. resonant mode  D=16 fourier  W_phase=static  (lr_wphase=None)",
        model_e, lr_wphase=None,
    )

    # F. Resonant mode, W_phase learned
    torch.manual_seed(SEED)
    model_f = SGNNET_Resonant(
        base=make_base(),
        K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="resonant",
        resonance_threshold=0.0,
    ).to(DEVICE)
    results["F"] = run(
        "F. resonant mode  D=16 fourier  W_phase=learned (lr_wphase=2.36e-3)",
        model_f, lr_wphase=GA_BEST["lr_Wpos"],
    )

    # G. dynamic_z_geo, full routing — geo bias check at D=16
    torch.manual_seed(SEED)
    model_g = SGNNET_Resonant(
        base=make_base(),
        K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
        geo_gamma=1.0,
    ).to(DEVICE)
    results["G"] = run(
        "G. dynamic_z_geo  D=16 fourier  full routing  (geo_gamma=1.0)",
        model_g, lr_wphase=None,
    )

    out = Path("results/train_step10b_wphase_d16.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    enc_ref = 0.2754
    print("\n-- W_phase + geo re-test at D=16 (ref: enc-sweep 27.54%) --------")
    print("  %-65s  %9s  %+8s  %6s" % ("Config", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*65 + "  " + "-"*9 + "  " + "-"*8 + "  " + "-"*6)
    for k, r in results.items():
        delta = r["top1_best"] - enc_ref
        print("  %-65s  %9.4f  %+8.4f  %6.0f" % (
            k, r["top1_best"], delta, r["elapsed_s"]))

    print(f"\n  W_phase static (E):  {results['E']['top1_best']:.4f}")
    print(f"  W_phase learned (F): {results['F']['top1_best']:.4f}")
    print(f"  geo dynamic_z (G):   {results['G']['top1_best']:.4f}")
    ef_delta = results["F"]["top1_best"] - results["E"]["top1_best"]
    if ef_delta >= 0.005:
        print(f"  -> W_phase LEARNS useful structure at D=16 (Δ={ef_delta:+.4f})")
    else:
        print(f"  -> W_phase provides no additional signal (Δ={ef_delta:+.4f})")
