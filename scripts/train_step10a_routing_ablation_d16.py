"""Step 10a: Routing component ablation at D=16 Fourier.

Which routing mechanisms actually contribute at D=16?
At D=4 (S³), theta (+6.6%) and turing (+6.5%) helped; reflection (+2.6%) was marginal.
At D=16 (S¹⁵) neurons have genuinely distinct directions — contributions may shift.

Ablation: isolate each component by zeroing out the others.
  A. theta only      — alpha_reflect=0.0, alpha_turing=0.0  (baseline: threshold gating)
  B. theta + reflect — alpha_reflect=0.3, alpha_turing=0.0
  C. theta + turing  — alpha_reflect=0.0, alpha_turing=0.3
  D. full routing    — alpha_reflect=0.3, alpha_turing=0.3  (current best)

All: D=16 Fourier N=512, dynamic_z, 120ep, plateau, store.h5, MPS.
Reference: encoding sweep D=16 dynamic_z_geo = 27.54% (90ep).

To reproduce:
    python -u scripts/train_step10a_routing_ablation_d16.py --device mps
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

EPOCHS = 120
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 16

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  encoding=fourier")
print(f"Goal: isolate theta / reflection / turing contributions at D=16")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model(alpha_reflect: float, alpha_turing: float) -> nn.Module:
    torch.manual_seed(SEED)
    tk = topology_kwargs(512)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=512, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=D,
        encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1,
        alpha_reflect=alpha_reflect,
        alpha_turing=alpha_turing,
        mode="dynamic_z",
        resonance_threshold=0.0,
    )


def run(label: str, model: nn.Module) -> dict:
    print(f"\n{'='*65}\n{label}\n{'='*65}")
    tr_loader, va_loader = get_loaders()
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
    return result


CONFIGS = [
    ("A. theta only       D=16 fourier dynamic_z  (reflect=0.0 turing=0.0)", 0.0, 0.0),
    ("B. theta+reflect    D=16 fourier dynamic_z  (reflect=0.3 turing=0.0)", 0.3, 0.0),
    ("C. theta+turing     D=16 fourier dynamic_z  (reflect=0.0 turing=0.3)", 0.0, 0.3),
    ("D. full routing     D=16 fourier dynamic_z  (reflect=0.3 turing=0.3)", 0.3, 0.3),
]

if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    results = {}
    for key, (label, ar, at) in zip("ABCD", CONFIGS):
        results[key] = run(label, make_model(ar, at).to(DEVICE))

    out = Path("results/train_step10a_routing_ablation_d16.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    enc_ref = 0.2754
    print("\n-- Routing component ablation at D=16 (ref: enc-sweep 27.54%) ------")
    print("  %-65s  %9s  %+8s  %6s" % ("Config", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*65 + "  " + "-"*9 + "  " + "-"*8 + "  " + "-"*6)
    for k, r in results.items():
        delta = r["top1_best"] - enc_ref
        print("  %-65s  %9.4f  %+8.4f  %6.0f" % (
            k, r["top1_best"], delta, r["elapsed_s"]))

    base_d = results["A"]["top1_best"]
    print(f"\n  theta-only baseline (A): {base_d:.4f}")
    print(f"  reflect contribution:    {results['B']['top1_best'] - base_d:+.4f}")
    print(f"  turing contribution:     {results['C']['top1_best'] - base_d:+.4f}")
    print(f"  both together (D):       {results['D']['top1_best'] - base_d:+.4f}")
