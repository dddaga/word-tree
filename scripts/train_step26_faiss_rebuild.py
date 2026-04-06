"""Step 26: FAISS Flat for conn_phase rebuild — does rebuild frequency matter?

MOTIVATION
==========
W_phase is updated every training step via gradient descent. The conn_phase K-NN graph
is currently rebuilt once per epoch (tick_epoch). Two questions:

  1. SPEED: Does FAISS Flat replace the O(N²) numpy brute-force with exact recall at
     much lower latency? (Benchmark: yes — 25× faster at N=512 D=16)

  2. ACCURACY: Does rebuilding conn_phase more frequently (every K steps vs every epoch)
     improve training accuracy?

     Hypothesis: At lr_eff≈0.003, W_phase drifts slowly — stale K-NN maintains >97%
     recall after 100 steps (bench_hnsw_phase.py result). So epoch-level rebuild should
     be sufficient. BUT: early in training, W_phase changes faster, and a fresh graph
     might guide early routing more effectively.

CONFIGS
=======
  Ref  : rebuild_interval=0  (epoch-only — current default)
  A    : rebuild_interval=10 (every 10 optimizer steps)
  B    : rebuild_interval=5  (every 5 steps)
  C    : rebuild_interval=1  (every step — maximum freshness)
  D    : rebuild_interval=0  + K_phase=16 (wider graph, epoch-only)
  E    : rebuild_interval=5  + K_phase=16 (wider + fresher)

All: D=16 N=512 Fourier dynamic_z_geo 150ep plateau store.h5.
Baseline: step19 best ≈ 28.56%  |  step18 signed ceiling ≈ 40.15%

To reproduce:
    python -u scripts/train_step26_faiss_rebuild.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

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
D      = 16
N      = 512

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model(K_phase: int = 8, rebuild_interval: int = 0) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=3,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=K_phase, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
        rebuild_interval=rebuild_interval,
    )


def run(label: str, model: torch.nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})  "
          f"diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# (label, K_phase, rebuild_interval)
CONFIGS = [
    ("Ref. dynamic_z_geo  rebuild=epoch  K_phase=8   [current default]",  8, 0),
    ("A.  dynamic_z_geo  rebuild=10step  K_phase=8   [every 10 steps]",   8, 10),
    ("B.  dynamic_z_geo  rebuild=5step   K_phase=8   [every 5 steps]",    8, 5),
    ("C.  dynamic_z_geo  rebuild=1step   K_phase=8   [every step]",       8, 1),
    ("D.  dynamic_z_geo  rebuild=epoch   K_phase=16  [wider graph]",      16, 0),
    ("E.  dynamic_z_geo  rebuild=5step   K_phase=16  [wider+fresher]",    16, 5),
]


if __name__ == "__main__":
    # Verify FAISS is available
    try:
        import faiss
        print(f"FAISS version: {faiss.__version__}  (exact K-NN, 25× faster than numpy)")
    except ImportError:
        print("WARNING: faiss-cpu not installed — will use torch O(N²) fallback")
        print("Install: pip install faiss-cpu")

    print(f"\nDevice: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}")
    print("Question: Does fresher conn_phase (sub-epoch FAISS rebuild) improve accuracy?")
    print(f"Expected baseline: ~28.56%  |  signed coupling ceiling: ~40.15%")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys    = ["Ref", "A", "B", "C", "D", "E"]
    for key, (label, K_phase, rebuild_interval) in zip(keys, CONFIGS):
        model = make_model(K_phase=K_phase, rebuild_interval=rebuild_interval).to(DEVICE)
        meta  = {
            "K_phase": K_phase, "rebuild_interval": rebuild_interval,
            "D": D, "N": N,
            "rebuild_freq": "epoch" if rebuild_interval == 0 else f"every_{rebuild_interval}_steps",
        }
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step26_faiss_rebuild.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results.get("Ref", {}).get("top1_best", 0.2856)
    print(f"\n-- FAISS rebuild frequency sweep  (ref_epoch={ref:.4f}) ---")
    print("  %-55s  %9s  %9s  %8s  %6s" % (
        "Config", "top1", "vs_epoch", "ep_frac", "t(s)"))
    print("  " + "-"*95)
    for k, r in results.items():
        d = r["top1_best"] - ref
        print("  %-55s  %9.4f  %+9.4f  %7.1f%%  %6.0f" % (
            r["label"][:55], r["top1_best"], d,
            r.get("best_epoch_frac", 0) * 100, r["elapsed_s"]))

    # Per-step rebuild adds K steps × ~0.4ms overhead — check if this slows training
    print("\n  Note: rebuild_interval=1 adds ~22ms/epoch at N=512 (55 batches × 0.4ms)")
    print("  If t(C) ≈ t(Ref), FAISS overhead is negligible as expected.")
