"""Step 33: D=128 dimension extension — does the D trend continue beyond D=64?

QUESTION
========
Step22 showed a strong D trend at K_iter=8:

  D=16  N=512   K_iter=8 = 36.74%  (baseline)
  D=16  N=1024  K_iter=8 = 39.29%  (+2.55pp)
  D=32  N=512   K_iter=8 = 49.38%  (+12.64pp)
  D=32  N=1024  K_iter=8 = 51.29%  (+14.55pp)
  D=64  N=512   K_iter=8 = 49.61%  (+12.87pp)
  D=64  N=1024  K_iter=8 = 56.28%  (+19.54pp)  ← current best

The D=64 winner has been the ceiling ever since. This step asks whether D=128
continues the trend (D=16→32→64 was +12pp each step) or whether D=64 is a
genuine plateau.

Secondary question: does N=2048 at D=64 recover more signal than D=128 at N=1024?
That tells us whether we should widen N or deepen D next.

CONFIGS
=======
  Ref   D=64  N=1024  K_iter=8  [step22E replication, expect ≈56.28%]
  A     D=128 N=512   K_iter=8  [D=128 moderate N]
  B     D=128 N=1024  K_iter=8  [D=128 same N as step22 winner]
  C     D=128 N=2048  K_iter=8  [D=128 larger N — does N still help at D=128?]
  D     D=64  N=2048  K_iter=8  [N=2048 at D=64 — N scaling from step22 baseline]

All: Fourier encoding, dynamic_z_geo, 150ep, plateau, store.h5.

To reproduce:
    python -u scripts/train_step33_d128.py --device mps
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.trainer            import Trainer
from src.training.dataset            import make_loaders
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.model_smallworld     import SGNNET_SmallWorld


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
def make_resonant(N: int, D: int, K_iter: int = 8) -> SGNNET_Resonant:
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
def run(label: str, model: SGNNET_Resonant, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    diag    = "training_too_short" if frac < 0.7 else "converged"
    result  = {
        "label": label,
        "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep,
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": diag,
        "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={diag}  task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


# ── Configs ───────────────────────────────────────────────────────────────────
# (key, label, N, D, K_iter)
CONFIGS = [
    ("Ref",
     "Ref   D=64  N=1024  K_iter=8  [step22E replication, expect ≈56.28%]",
     1024, 64,  8),
    ("A",
     "A     D=128 N=512   K_iter=8  [D=128 moderate N]",
     512,  128, 8),
    ("B",
     "B     D=128 N=1024  K_iter=8  [D=128 same N as step22 winner]",
     1024, 128, 8),
    ("C",
     "C     D=128 N=2048  K_iter=8  [D=128 larger N — does N still help at D=128?]",
     2048, 128, 8),
    ("D",
     "D     D=64  N=2048  K_iter=8  [N=2048 at D=64 — N scaling from step22 baseline]",
     2048, 64,  8),
]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}")
    print("Goal: does D=128 continue D=16→32→64 trend (+12pp/step)?")
    print("      secondary: is D=128 N=1024 better or worse than D=64 N=2048?")
    print("step22E best: D=64 N=1024 K_iter=8 = 56.28%")
    tr, va = get_loaders()
    print(f"Dataset: train={len(tr.dataset)}  val={len(va.dataset)}")

    results  = {}
    ref_top1 = None

    for key, label, N, D, K_iter in CONFIGS:
        model = make_resonant(N=N, D=D, K_iter=K_iter).to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": K_iter, "experiment": "d128_extension"}
        results[key] = run(label, model, meta)
        results[key].update(meta)
        if key == "Ref":
            ref_top1 = results[key]["top1_best"]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "train_step33_d128.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    D64_N1024_ref = ref_top1 if ref_top1 is not None else 0.5628

    print(f"\n-- D=128 extension sweep at K_iter=8  (D64_N1024_ref={D64_N1024_ref:.4f}) --")
    print(f"  {'Config':<60}  {'top1':>6}  {'vs_D64_N1024_ref':>17}  {'best_ep':>8}  {'t(s)':>6}")
    print("  " + "-"*106)
    for key, label, N, D, K_iter in CONFIGS:
        r  = results[key]
        vs = f"{r['top1_best'] - D64_N1024_ref:+.4f}"
        ep = f"{r['best_epoch']}/{r['epochs_run']} ({r['best_epoch_frac']:.0%})"
        print(f"  {label:<60}  {r['top1_best']:.4f}  {vs:>17}  {ep:>8}  {r['elapsed_s']:>6.0f}")

    # ── D trend line ──────────────────────────────────────────────────────────
    print(f"\n-- Full D trend line (N=1024, K_iter=8) --------------------------------")
    print(f"  {'Config':<35}  {'top1':>6}  {'vs_D16_baseline':>16}")
    print("  " + "-"*62)
    D_TREND = [
        ("D=16  N=512   K_iter=8  [step22 Ref]", 0.3674),
        ("D=16  N=1024  K_iter=8  [step22 A]",   0.3929),
        ("D=32  N=512   K_iter=8  [step22 B]",   0.4938),
        ("D=32  N=1024  K_iter=8  [step22 C]",   0.5129),
        ("D=64  N=512   K_iter=8  [step22 D]",   0.4961),
        ("D=64  N=1024  K_iter=8  [step22 E]",   0.5628),
    ]
    d16_base = 0.3674
    for row_label, known_top1 in D_TREND:
        vs = known_top1 - d16_base
        print(f"  {row_label:<35}  {known_top1:.4f}  {vs:>+15.4f}")

    # Append this step's results to the trend
    new_rows = [
        ("D=64  N=2048  K_iter=8  [step33 D]", "D"),
        ("D=128 N=512   K_iter=8  [step33 A]", "A"),
        ("D=128 N=1024  K_iter=8  [step33 B]", "B"),
        ("D=128 N=2048  K_iter=8  [step33 C]", "C"),
    ]
    for row_label, key in new_rows:
        r  = results[key]
        vs = r["top1_best"] - d16_base
        print(f"  {row_label:<35}  {r['top1_best']:.4f}  {vs:>+15.4f}  ← this run")

    print(f"\n  If B > step22E ({D64_N1024_ref:.4f}): D trend continues → D=256 is next frontier")
    print(f"  If B ≈ step22E:              D=64 is encoding ceiling → focus on N / K_iter")
    print(f"  If D (D=64 N=2048) > B:      N scaling outperforms D scaling → widen N for Gen4")
