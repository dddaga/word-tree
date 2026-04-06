"""Step 29: AntiHebb inhibition at D=64, with K_iter diagnostic.

QUESTION
========
Step28 Ref showed: D=64 N=1024 K_iter=3 + signed = 32.15%
Step22E showed:    D=64 N=1024 K_iter=8  no signed = 56.28%

Two variables changed simultaneously. This step isolates them:

  Ref0  D=64 N=1024 K_iter=3  NO signed  → tells us how much K_iter=3 alone costs at D=64
  Ref   D=64 N=1024 K_iter=8  NO signed  → reproduced step22E baseline (should ≈56.28%)

Then tests whether AntiHebb inhibition (step16 winner, +8pp at D=16) compounds with D=64:

  A     D=64 K_iter=8  + AntiHebb α=0.5 wpos  [strongest variant from step16]
  B     D=64 K_iter=8  + AntiHebb α=0.3 wpos  [weaker variant]
  C     D=64 K_iter=8  + AntiHebb α=0.7 wpos  [stronger variant — test ceiling]

Key diagnostics:
  Ref0 vs Ref  → cost of K_iter=3 at D=64 (isolates K_iter effect)
  Ref0 vs 32.15%  → effect of signed coupling at D=64+K_iter=3
  A vs Ref     → AntiHebb gain at D=64
"""

import argparse, json, time, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.trainer            import Trainer
from src.training.dataset            import make_loaders
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian


# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
DEVICE = parser.parse_args().device

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


# ── Model factory ─────────────────────────────────────────────────────────────
def make_resonant(N: int = 1024, D: int = 64, K_iter: int = 8) -> SGNNET_Resonant:
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


# ── Run helper ────────────────────────────────────────────────────────────────
def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va   = get_loaders()
    tk       = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer  = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0       = time.time()
    history  = trainer.train(n_epochs=EPOCHS)
    elapsed  = time.time() - t0
    best     = max(h["val_top1"] for h in history)
    best_ep  = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac     = best_ep / EPOCHS
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})"
          f"  t={elapsed:.0f}s")
    return {
        "label": label, "top1_best": best, "best_ep": best_ep,
        "ep_frac": frac, "t": elapsed,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
        **meta,
    }


# ── Configs ───────────────────────────────────────────────────────────────────
# (key, label, K_iter, factory)
CONFIGS = [
    ("Ref0",
     "Ref0  D=64 N=1024 K_iter=3  NO signed  [diagnostic: K_iter=3 cost at D=64]",
     3, lambda r: r),
    ("Ref",
     "Ref   D=64 N=1024 K_iter=8  NO signed  [step22E replication, expect ≈56.28%]",
     8, lambda r: r),
    ("A",
     "A     D=64 K_iter=8  + AntiHebb α=0.5 wpos  [step16 winner at D=64]",
     8, lambda r: SGNNET_AntiHebbian(r, alpha_ahebb=0.5, variant="wpos")),
    ("B",
     "B     D=64 K_iter=8  + AntiHebb α=0.3 wpos",
     8, lambda r: SGNNET_AntiHebbian(r, alpha_ahebb=0.3, variant="wpos")),
    ("C",
     "C     D=64 K_iter=8  + AntiHebb α=0.7 wpos  [stronger — test ceiling]",
     8, lambda r: SGNNET_AntiHebbian(r, alpha_ahebb=0.7, variant="wpos")),
]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}")
    print("Goal: isolate K_iter vs signed, test AntiHebb at D=64 K_iter=8")
    print("step28 Ref (signed+K_iter=3) = 32.15%.  step22E (no signed+K_iter=8) = 56.28%")

    results = {}
    ref_top1 = None

    for key, label, K_iter, factory in CONFIGS:
        resonant = make_resonant(N=1024, D=64, K_iter=K_iter).to(DEVICE)
        model    = factory(resonant).to(DEVICE)
        meta     = {"N": 1024, "D": 64, "K_iter": K_iter,
                    "mechanism": "antihebb_d64"}
        results[key] = run(label, model, meta)
        if key == "Ref":
            ref_top1 = results[key]["top1_best"]

    # ── Summary ───────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "train_step29_antihebb_d64.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    ref0  = results["Ref0"]["top1_best"]
    ref8  = results["Ref"]["top1_best"]
    signed_k3 = 0.3215   # step28 Ref

    print(f"\n-- AntiHebb @ D=64  -----------------------------------------------")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref8':>8}  {'ep%':>5}  {'t(s)':>6}")
    print("  " + "-"*80)
    for key, label, _, _ in CONFIGS:
        r  = results[key]
        vs = f"{r['top1_best'] - ref8:+.4f}" if key != "Ref" else "   base"
        print(f"  {label:<55}  {r['top1_best']:.4f}  {vs:>8}  "
              f"{r['ep_frac']:>4.0%}  {r['t']:>6.0f}")

    print(f"\n  Diagnostics:")
    print(f"  K_iter=3 no-signed (Ref0):     {ref0:.4f}")
    print(f"  K_iter=3 + signed  (step28):   {signed_k3:.4f}")
    print(f"  K_iter=8 no-signed (Ref):      {ref8:.4f}")
    print(f"  → K_iter cost at D=64:         {ref0 - ref8:+.4f}pp")
    print(f"  → Signed effect at K_iter=3:   {signed_k3 - ref0:+.4f}pp")
    print(f"\n  If A > Ref: AntiHebb compounds with D=64 deep routing → include in Gen4")
    print(f"  If A ≈ Ref: AntiHebb doesn't survive K_iter=8 (interaction with depth)")
