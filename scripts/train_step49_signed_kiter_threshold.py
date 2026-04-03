"""Step 49: Signed coupling × K_iter threshold sweep at D=64.

QUESTION
========
All prior signed coupling failures at D=64 used either K_iter=3 (too shallow)
or K_iter=8 (collapse). Is there a viable K_iter range (4–7) where:
  - More routing steps have built up non-trivial cosine similarities on S^63
    (at init, E[cos²] ≈ 1/64 ≈ 0.016 — near noise. After routing, similarities GROW
    as neurons average over neighbors → coupling signal becomes meaningful)
  - The power-iteration collapse threshold has NOT been reached yet

EVIDENCE FOR AND AGAINST
========================
FOR (sweet spot might exist):
  - At D=16 K_iter=3, cos-sim is already meaningful → coupling works at K_iter=3
  - At D=64, cos-sim at init is near-zero → coupling at K_iter=3 is pure noise
  - After 4–6 routing steps, Z_h ≈ average of neighborhood → cos-sim grows to O(0.3–0.5)
  - AntiHebb at K_iter=8 proves the routing builds up structured patterns — coupling
    signal should exist by K_iter=4+
  - The collapse at K_iter=8 is a power-iteration effect on the FULL signed coupling
    loop; lower K_iter may avoid it

AGAINST (probably still fails):
  - step42 (K_iter=3) already showed coupling HURTS at D=64 at ANY α
  - AntiHebb helps by REDUCING coupling between similar neurons → diversity matters more
  - If directional diversity is the signal, signed coupling (which ENFORCES similarity)
    is the wrong mechanism regardless of K_iter
  - Even if K_iter=4-5 avoids full collapse, partial coupling may just add noise

DESIGN
======
Two-phase design: 40-epoch calibration per K_iter config to find if ANY K_iter is viable.
Use cosine schedule to prevent LR collapse (plateau scheduler failed in step42).

Phase 1: Find the collapse threshold — K_iter values {3,4,5,6,7,8}
Phase 2: If any K_iter in 4–7 shows >Ref at 40ep, run 150ep full experiment

CONFIGS (D=64 N=1024, cosine LR, 40ep each — fast viability check):
  Ref    K_iter=8   no signed  [~56% but in 40ep: ~42%, per step22b trajectory]
  A      K_iter=3   + signed α=0.3  [known: ~25% at 40ep, step42 confirmation]
  B      K_iter=4   + signed α=0.3  [first test above known-fail threshold]
  C      K_iter=5   + signed α=0.3  [mid-range]
  D      K_iter=6   + signed α=0.3  [approaching collapse zone]
  E      K_iter=7   + signed α=0.3  [one below collapse threshold]
  F      K_iter=8   + signed α=0.3  [known: collapse, step23 — sanity check]

  G      K_iter=5   + signed α=0.1  [if C shows promise, lower α might stabilize]
  H      K_iter=5   + signed α=0.5  [stronger coupling at sweet spot candidate]
  I      K_iter=6   + signed α=0.1  [if D shows promise]

DECISION RULE (after 40ep):
  Any config > Ref_40ep (~42%) → viable, deserves 150ep full run
  Monotone decline B→C→D→E     → sweet spot doesn't exist; close the question
  Collapse signature (top1 < 15%) at config X → X is the collapse threshold

BROADER VALUE:
  This directly answers: "is signed coupling dead at D=64 (mechanism wrong),
  or was it just tested at the wrong K_iter (K_iter=3 too shallow, K_iter=8 collapse)?"

To reproduce:
    python -u scripts/train_step49_signed_kiter_threshold.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
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

# 40-epoch calibration runs — cosine schedule to avoid LR collapse (step42 lesson)
EPOCHS = 40
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 1024
D      = 64

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant(K_iter: int) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


class SGNNET_SignedCoupling(nn.Module):
    """Signed coupling: Z_new = normalize(Z_struct + alpha*(Z Z^T)Z + alpha_turing*Z_inh).

    (Z Z^T)Z is the all-pairs cosine-weighted message — identical neurons reinforce,
    anti-correlated neurons suppress. At K_iter≥8 this is a power iteration that
    converges to the dominant eigenvector of the N×N Gram matrix (collapse).

    This wrapper tracks the mean cosine similarity at each epoch for diagnostics.
    """

    def __init__(self, base_model: SGNNET_Resonant, alpha_signed: float = 0.3):
        super().__init__()
        self.m            = base_model
        self.alpha_signed = alpha_signed
        self.last_mean_cos = 0.0

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh

        for step_i in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Signed coupling: O(N²·D) per step
            Z_norm   = F.normalize(Z, dim=-1)             # [B, N, D]
            gram     = torch.bmm(Z_norm, Z_norm.transpose(1, 2))  # [B, N, N]
            Z_signed = torch.bmm(gram, Z_norm) / N        # [B, N, D]

            # Track mean cosine similarity as collapse diagnostic
            if step_i == self.m.base.K_iter - 1:
                with torch.no_grad():
                    # Upper triangle of gram (excluding diagonal)
                    mask = torch.triu(torch.ones(N, N, device=Z.device), diagonal=1).bool()
                    self.last_mean_cos = gram[0][mask].abs().mean().item()

            Z = F.normalize(
                (Z_struct + self.alpha_signed * Z_signed
                 + self.m.alpha_turing * Z_inh).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    # cosine schedule — avoids plateau LR collapse that killed step42
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="cosine")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)

    # Collapse diagnostic: if model has last_mean_cos, record it
    mean_cos = getattr(model, "last_mean_cos", None)

    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "collapse_signature": best < 0.15,  # <15% = collapse
        "mean_cos_final": round(mean_cos, 4) if mean_cos else None,
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  {'COLLAPSE!' if result['collapse_signature'] else ''}"
          f"  mean_cos={f'{mean_cos:.4f}' if mean_cos is not None else 'N/A'}  t={elapsed:.0f}s")
    return result


# Phase 1: K_iter threshold sweep (all at α=0.3 for direct comparison to step42)
CONFIGS_P1 = [
    ("Ref", "Ref   K_iter=8  no-signed  [D=64 base, ~42% at 40ep]",  8, None,  0.0),
    ("A",   "A     K_iter=3  signed α=0.3  [step42: ~25%, confirms fail]", 3, 0.3, 0.3),
    ("B",   "B     K_iter=4  signed α=0.3  [first test above K_iter=3]",   4, 0.3, 0.3),
    ("C",   "C     K_iter=5  signed α=0.3  [mid-range sweet spot?]",        5, 0.3, 0.3),
    ("D",   "D     K_iter=6  signed α=0.3  [approaching collapse zone]",    6, 0.3, 0.3),
    ("E",   "E     K_iter=7  signed α=0.3  [one below collapse]",           7, 0.3, 0.3),
    ("F",   "F     K_iter=8  signed α=0.3  [known collapse, step23]",       8, 0.3, 0.3),
]

# Phase 2: α sweep at best K_iter from P1 (if any config > Ref_40ep)
# (conditionally appended below after P1 completes)
CONFIGS_P2 = [
    ("G",   "G     K_iter=5  signed α=0.1  [lower α at sweet spot candidate]",  5, 0.1, 0.1),
    ("H",   "H     K_iter=5  signed α=0.5  [stronger α at sweet spot candidate]",5, 0.5, 0.5),
    ("I",   "I     K_iter=6  signed α=0.1  [lower α if K_iter=6 promising]",    6, 0.1, 0.1),
]


if __name__ == "__main__":
    REF_40EP = 0.42   # expected D=64 base at 40ep (step22b trajectory)
    COLLAPSE  = 0.15  # <15% = power-iteration collapse confirmed

    print(f"Device: {DEVICE}  Epochs: {EPOCHS} (cosine LR)  Batch: {BATCH}")
    print("Step 49: Signed coupling × K_iter threshold sweep at D=64")
    print("Question: Is there a K_iter sweet spot (4-7) where signed coupling works at D=64?")
    print("Known: K_iter=3 → 25% (below ref). K_iter=8 → collapse (<15%). Gap untested.")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}

    # --- Phase 1 ---
    print("\n" + "="*70)
    print("PHASE 1: K_iter threshold sweep (α=0.3, 40ep cosine)")
    print("="*70)
    for key, label, K_iter, alpha_s, alpha_ref in CONFIGS_P1:
        resonant = make_resonant(K_iter=K_iter).to(DEVICE)
        if alpha_s is None:
            model = resonant
        else:
            model = SGNNET_SignedCoupling(resonant, alpha_signed=alpha_s).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": K_iter,
                "alpha_signed": alpha_s, "phase": 1}
        results[key] = run(label, model, meta)
        results[key].update(meta)

    # Determine if Phase 2 is warranted
    ref_40ep = results["Ref"]["top1_best"]
    p1_best_key  = max((k for k in ["A","B","C","D","E","F"]),
                       key=lambda k: results[k]["top1_best"])
    p1_best_val  = results[p1_best_key]["top1_best"]
    run_phase2 = p1_best_val > ref_40ep
    print(f"\nPhase 1 best: {p1_best_key} = {p1_best_val:.4f}  "
          f"Ref_40ep = {ref_40ep:.4f}  → Phase 2: {'YES' if run_phase2 else 'NO (no sweet spot found)'}")

    # --- Phase 2 (conditional) ---
    if run_phase2:
        print("\n" + "="*70)
        print(f"PHASE 2: α sweep at best K_iter={results[p1_best_key]['K_iter']}")
        print("="*70)
        for key, label, K_iter, alpha_s, alpha_ref in CONFIGS_P2:
            resonant = make_resonant(K_iter=K_iter).to(DEVICE)
            model    = SGNNET_SignedCoupling(resonant, alpha_signed=alpha_s).to(DEVICE)
            meta     = {"N": N, "D": D, "K_iter": K_iter,
                        "alpha_signed": alpha_s, "phase": 2}
            results[key] = run(label, model, meta)
            results[key].update(meta)

    out = ROOT / "results" / "train_step49_signed_kiter_threshold.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    print(f"\n-- Signed coupling × K_iter threshold (Ref_40ep={ref_40ep:.4f}) ---")
    print(f"  {'Config':<58}  {'top1':>6}  {'vs_Ref':>8}  {'collapse':>8}  {'t(s)':>6}")
    print("  " + "-"*100)
    for k, r in results.items():
        d    = r["top1_best"] - ref_40ep
        coll = "COLLAPSE" if r.get("collapse_signature") else "ok"
        print(f"  {r['label'][:58]:<58}  {r['top1_best']:>6.4f}  {d:>+8.4f}"
              f"  {coll:>8}  {r['elapsed_s']:>6.0f}")

    print("\n  Interpretation:")
    print("  B/C/D/E > Ref → sweet spot exists; signed coupling viable at that K_iter")
    print("  Monotone A→B→C→D→E < Ref → mechanism dead at D=64 regardless of K_iter")
    print("  Collapse at config X → X is the power-iteration threshold at D=64")
    print("  If no sweet spot: signed coupling is architecturally incompatible with D=64")
    print("  If sweet spot found: run 150ep full experiment + AntiHebb compound")
