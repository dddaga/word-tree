"""Step 60: Phase routing magnitude ablation.

Tests SGNNET_PhaseRouting — a new routing architecture where Z[0] is the
activation magnitude and Z[1:] are 63 cyclic phase channels.
Activation travels via conn_hh with a distance-dependent phase shift;
magnitudes accumulate via one of 6 gate modes x 3 coherence refs x 2 freq modes.

NOTE: step57 REF was trained with standard SGNNET_AntiHebbian on SGNNET_Resonant.
      PhaseRouting uses a bare SGNNET_SmallWorld base (no Resonant wrapper) — it is
      a different architecture, not a mechanism on top of Resonant.

Configs:
  Ref      : AntiHebbian alpha=1.0 (from step57 Resonant — architecture baseline)
  A        : independent      ah_alpha=0.0  freq=capped_exp
  B        : coherent         ah_alpha=0.0  freq=capped_exp  ref=dynamic
  C        : independent      ah_alpha=1.0  freq=capped_exp
  D        : coherent         ah_alpha=1.0  freq=capped_exp  ref=dynamic
  A_wt     : weighted_phase   ah_alpha=0.0  freq=capped_exp
  B_wt     : coherent_weighted ah_alpha=0.0 freq=capped_exp  ref=dynamic
  B_anchor : coherent         ah_alpha=0.0  freq=capped_exp  ref=anchor
  B_abs    : coherent         ah_alpha=0.0  freq=capped_exp  ref=absolute
  E        : decay_coherence  ah_alpha=0.0  freq=capped_exp  ref=dynamic  lambda=1.0
  F        : decay_coherence  ah_alpha=1.0  freq=capped_exp  ref=dynamic  lambda=1.0
  A_prime  : independent      ah_alpha=0.0  freq=harmonic_primes
  B_prime  : coherent         ah_alpha=0.0  freq=harmonic_primes  ref=dynamic

Questions:
  A/B vs Ref    -> does phase-shift routing compete with AntiHebb Resonant?
  B vs A        -> does phase-coherent interference improve over independent magnitude?
  C/D vs A/B    -> does AntiHebb suppression help PhaseRouting?
  A_wt vs A     -> does mag-weighted phase update help?
  B_wt vs B     -> coherent gate + mag-weighted phase: additive gain?
  B_anchor/abs  -> does choice of coherence reference matter?
  E vs B        -> does decay weighting over coherent gate help?
  F vs E        -> decay_coherence + AH: best combined config?
  A_prime vs A  -> harmonic primes vs capped_exp frequencies?
  B_prime vs B  -> same but with coherent gate?

To reproduce:
    python -u scripts/train_step60_phase_routing_magnitude.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_phase_routing   import SGNNET_PhaseRouting
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

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
    """Resonant model — used only for Ref (AntiHebb baseline)."""
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


def make_phase_base(N=1024, D=64, K_iter=8) -> SGNNET_SmallWorld:
    """Bare SmallWorld base for PhaseRouting — no Resonant wrapper."""
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    return SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )


def make_phase_routing(magnitude_mode="independent", coherence_ref="dynamic",
                       freq_mode="capped_exp", lambda_decay=1.0, ah_alpha=0.0,
                       N=1024, D=64, K_iter=8) -> SGNNET_PhaseRouting:
    """Factory: bare SmallWorld base wrapped in SGNNET_PhaseRouting."""
    base = make_phase_base(N=N, D=D, K_iter=K_iter)
    return SGNNET_PhaseRouting(
        base, K_phase=8, beam_size=16,
        theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
        magnitude_mode=magnitude_mode,
        coherence_ref=coherence_ref,
        freq_mode=freq_mode,
        lambda_decay=lambda_decay,
        ah_alpha=ah_alpha,
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

# Load REF_BASELINE from step57
_ref_path = ROOT / "results" / "train_step57_benchmark_ablation.json"
try:
    REF_BASELINE = json.loads(_ref_path.read_text()).get("top1_best", 0.8008)
    print(f"Loaded REF_BASELINE from step57: {REF_BASELINE:.4f}")
except Exception:
    REF_BASELINE = 0.8008
    print(f"step57 result not found — using fallback REF_BASELINE={REF_BASELINE:.4f}")

# (key, label, kind, kwargs)
# kind="antihebb" -> Resonant+AntiHebb baseline
# kind="phase"    -> SGNNET_PhaseRouting via make_phase_routing(**kwargs)
CONFIGS = [
    # ---- Baseline ----
    ("Ref",      "Ref   AntiHebb alpha=1.0 wpos (Resonant backbone)",
     "antihebb", {}),
    # ---- Original 4 configs (freq_mode now explicit) ----
    ("A",        "A     independent no AH  freq=capped_exp",
     "phase", dict(magnitude_mode="independent",       freq_mode="capped_exp",
                   coherence_ref="dynamic",              ah_alpha=0.0)),
    ("B",        "B     coherent dynamic-ref no AH  freq=capped_exp",
     "phase", dict(magnitude_mode="coherent",           freq_mode="capped_exp",
                   coherence_ref="dynamic",              ah_alpha=0.0)),
    ("C",        "C     independent AH=1.0  freq=capped_exp",
     "phase", dict(magnitude_mode="independent",       freq_mode="capped_exp",
                   coherence_ref="dynamic",              ah_alpha=1.0)),
    ("D",        "D     coherent dynamic-ref AH=1.0  freq=capped_exp",
     "phase", dict(magnitude_mode="coherent",           freq_mode="capped_exp",
                   coherence_ref="dynamic",              ah_alpha=1.0)),
    # ---- New configs ----
    ("A_wt",     "A_wt  independent+mag-weighted-phase no AH",
     "phase", dict(magnitude_mode="weighted_phase",    freq_mode="capped_exp",
                   coherence_ref="dynamic",              ah_alpha=0.0)),
    ("B_wt",     "B_wt  coherent+mag-weighted-phase no AH",
     "phase", dict(magnitude_mode="coherent_weighted", freq_mode="capped_exp",
                   coherence_ref="dynamic",              ah_alpha=0.0)),
    ("B_anchor", "B_anc coherent anchor-ref no AH",
     "phase", dict(magnitude_mode="coherent",           freq_mode="capped_exp",
                   coherence_ref="anchor",               ah_alpha=0.0)),
    ("B_abs",    "B_abs coherent absolute-ref no AH",
     "phase", dict(magnitude_mode="coherent",           freq_mode="capped_exp",
                   coherence_ref="absolute",             ah_alpha=0.0)),
    ("E",        "E     decay_coherence no AH lambda=1.0",
     "phase", dict(magnitude_mode="decay_coherence",   freq_mode="capped_exp",
                   coherence_ref="dynamic",              lambda_decay=1.0, ah_alpha=0.0)),
    ("F",        "F     decay_coherence AH=1.0 lambda=1.0",
     "phase", dict(magnitude_mode="decay_coherence",   freq_mode="capped_exp",
                   coherence_ref="dynamic",              lambda_decay=1.0, ah_alpha=1.0)),
    ("A_prime",  "A_prm independent harmonic-primes no AH",
     "phase", dict(magnitude_mode="independent",       freq_mode="harmonic_primes",
                   coherence_ref="dynamic",              ah_alpha=0.0)),
    ("B_prime",  "B_prm coherent harmonic-primes no AH",
     "phase", dict(magnitude_mode="coherent",           freq_mode="harmonic_primes",
                   coherence_ref="dynamic",              ah_alpha=0.0)),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 60: PhaseRouting magnitude ablation  |  REF_BASELINE={REF_BASELINE:.4f} (step57)")
    print("NOTE: PhaseRouting uses bare SmallWorld base, not Resonant — new architecture test")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, kind, kwargs in CONFIGS:
        if kind == "antihebb":
            resonant = make_resonant(N=N, D=64, K_iter=8).to(DEVICE)
            model    = SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos").to(DEVICE)
            meta     = {"N": N, "D": 64, "K_iter": 8, "mechanism": "antihebb",
                        "alpha_ahebb": 1.0, "data_frac": 0.5}
        else:
            model = make_phase_routing(N=N, D=64, K_iter=8, **kwargs).to(DEVICE)
            meta  = {"N": N, "D": 64, "K_iter": 8, "mechanism": "phase_routing",
                     "data_frac": 0.5, **kwargs}

        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step60_phase_routing_magnitude.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Step 60: PhaseRouting magnitude (ref={ref_val:.4f} / step57={REF_BASELINE:.4f}) --")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'best_ep':>8}  {'t(s)':>6}")
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r.get('best_epoch', 0):>7d}    {r['elapsed_s']:>6.0f}")

    print("\n  Key questions:")
    print("  A/B vs Ref    -> does phase-shift routing compete with AntiHebb Resonant?")
    print("  B vs A        -> does phase-coherent interference beat independent magnitude?")
    print("  C/D vs A/B    -> does AntiHebb suppression help PhaseRouting?")
    print("  A_wt vs A     -> does mag-weighted phase update help?")
    print("  B_wt vs B     -> coherent gate + mag-weighted phase: additive gain?")
    print("  B_anchor/abs  -> does choice of coherence reference matter?")
    print("  E vs B        -> does decay weighting over coherent gate help?")
    print("  F vs E        -> decay_coherence + AH: best combined config?")
    print("  A_prime vs A  -> harmonic primes vs capped_exp frequencies?")
    print("  B_prime vs B  -> same but with coherent gate?")
