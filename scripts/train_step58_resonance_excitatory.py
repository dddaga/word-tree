"""Step 58: Resonance excitatory mechanism ablation.

Tests SGNNET_ResonanceExcitatory variants vs AntiHebbian baseline.
50% data, 75 epochs. REF_BASELINE from step57; fallback to 0.8008.

Configs:
  Ref : AntiHebbian alpha=1.0 wpos (quick sanity vs step57)
  A   : ResonanceExcitatory(rebuild_per_batch=True,  use_activation_gate=True,  alpha_exc=0.3)
  B   : ResonanceExcitatory(rebuild_per_batch=False, use_activation_gate=True,  alpha_exc=0.3)
  C   : ResonanceExcitatory(rebuild_per_batch=True,  use_activation_gate=False, alpha_exc=0.3)

Questions:
  A vs B : does per-batch rebuild of conn_phase help (cost: graph rebuild each forward)?
  A vs C : does the activation gate (resonance gate) improve over plain excitation?
  A vs Ref: does phase excitation improve over phase inhibition (AntiHebb)?

To reproduce:
    python -u scripts/train_step58_resonance_excitatory.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld        import SGNNET_SmallWorld
from src.sgnnet.model_resonant          import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory   import SGNNET_AntiHebbian
from src.sgnnet.mechanisms_resonance    import SGNNET_ResonanceExcitatory
from src.training.trainer               import Trainer
from src.training.experiment_config     import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset               import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_resonant(N=1024, D=64, K_iter=8) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=16,
        theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=0.5,
    )


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
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
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


N = 1024

# Load REF_BASELINE from step57 result if available
_ref_path = ROOT / "results" / "train_step57_benchmark_ablation.json"
try:
    REF_BASELINE = json.loads(_ref_path.read_text()).get("top1_best", 0.8008)
    print(f"Loaded REF_BASELINE from step57: {REF_BASELINE:.4f}")
except Exception:
    REF_BASELINE = 0.8008
    print(f"step57 result not found — using fallback REF_BASELINE={REF_BASELINE:.4f}")

CONFIGS = [
    ("Ref", "Ref   AntiHebb alpha=1.0 wpos",                                  "antihebb",   {}),
    ("A",   "A     ResonanceExc rebuild_batch=T gate=T alpha_exc=0.3",         "exc",
     dict(rebuild_per_batch=True,  use_activation_gate=True,  alpha_exc=0.3)),
    ("B",   "B     ResonanceExc rebuild_batch=F gate=T alpha_exc=0.3",         "exc",
     dict(rebuild_per_batch=False, use_activation_gate=True,  alpha_exc=0.3)),
    ("C",   "C     ResonanceExc rebuild_batch=T gate=F alpha_exc=0.3",         "exc",
     dict(rebuild_per_batch=True,  use_activation_gate=False, alpha_exc=0.3)),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 58: Resonance excitatory ablation  |  REF_BASELINE={REF_BASELINE:.4f} (step57)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, kind, kwargs in CONFIGS:
        resonant = make_resonant(N=N, D=64, K_iter=8).to(DEVICE)
        if kind == "antihebb":
            model = SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos").to(DEVICE)
            meta  = {"N": N, "D": 64, "K_iter": 8, "mechanism": "antihebb",
                     "alpha_ahebb": 1.0, "data_frac": 0.5}
        else:
            model = SGNNET_ResonanceExcitatory(resonant, **kwargs).to(DEVICE)
            meta  = {"N": N, "D": 64, "K_iter": 8, "mechanism": "resonance_excitatory",
                     "data_frac": 0.5, **kwargs}

        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step58_resonance_excitatory.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Step 58: ResonanceExcitatory ablation (ref={ref_val:.4f} / step57={REF_BASELINE:.4f}) --")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'best_ep':>8}  {'t(s)':>6}")
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r.get('best_epoch', 0):>7d}    {r['elapsed_s']:>6.0f}")

    print("\n  Key questions:")
    print("  A vs Ref  → does excitation beat inhibition as long-range phase signal?")
    print("  A vs B    → does per-batch conn_phase rebuild help?")
    print("  A vs C    → does resonance activation gate add value over plain excitation?")
