"""Step 29b: Gen1 mechanism winners at D=64 N=1024 K_iter=8 with calibration phase.

MOTIVATION
==========
Gen1 mechanism winners (phase_exc, interneurons, fast_W_phase) were tuned at D=16.
At D=64 cosine similarities concentrate near zero on S^63 vs S^15 — the same alpha
value has very different effective strength. We cannot assume D=16 hyperparameters
transfer. This script adds a calibration sweep before the full validation runs.

CALIBRATION PHASE (CALIB_EPOCHS=40 epochs each)
=================================================
Fast sweep to find the correct alpha / tau range at D=64:
  - Phase excitatory:          alpha in {0.1, 0.3, 0.5, 1.0}
  - Fast W_phase (attention):  alpha in {0.1, 0.3} x tau in {0.25, 0.5, 1.0}
  - Interneurons (fraction):   n_input in {N*3//4 (25% int), N//2 (50% int)}

Best alpha/tau per mechanism is selected programmatically after calibration,
then used in the full Phase 2 runs.

FULL PHASE (FULL_EPOCHS=150 epochs)
=====================================
  Ref   D=64 K_iter=8 vanilla                         [sanity check ~56.28%]
  A     + phase_exc   alpha=<best_from_calib>
  B     + interneurons <best_frac>
  C     + fast_W_phase alpha=<best_from_calib> tau=<best_from_calib>
  D     + phase_exc + interneurons                     [compound]
  E     + phase_exc + interneurons + fast_W_phase      [triple compound]

All: D=64 Fourier N=1024 K_iter=8 plateau SEED=42 store.h5.

To reproduce:
    python -u scripts/train_step29b_mechanisms_d64.py --device mps
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

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset            import make_loaders
from src.sgnnet.mechanisms_excitatory import SGNNET_FastPhase


# ── CLI & constants ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

CALIB_EPOCHS = 40
FULL_EPOCHS  = 150
BATCH        = 128
SEED         = 42
DATA         = "data/store.h5"
D            = 64
N            = 1024
K_ITER       = 8

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


# ── Model factory ─────────────────────────────────────────────────────────────

def make_resonant() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


# ── Mechanism classes (inlined exactly from original scripts) ─────────────────

class SGNNET_PhaseExcitatory(nn.Module):
    """W_phase K-NN graph used for EXCITATORY routing.

    conn_phase (rebuilt by tick_epoch from W_phase similarity) provides
    learned long-range excitatory connections that bypass structural distance.
    The 'teleportation portal': any neuron can directly excite its K_phase
    most phase-similar partners in a single routing step.

    Parameters
    ----------
    base_model      : SGNNET_Resonant (any mode; we override routing)
    alpha_exc       : weight for phase excitatory contribution
    with_inhibition : if True, keep dynamic_z_geo inhibitory beam too
    """

    def __init__(self, base_model: SGNNET_Resonant, alpha_exc: float = 0.3,
                 with_inhibition: bool = True):
        super().__init__()
        self.m               = base_model
        self.alpha_exc       = alpha_exc
        self.with_inhibition = with_inhibition

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        self.m.tick_epoch()  # rebuilds conn_phase from W_phase K-NN each epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)

            # W_phase teleportation: excite via learned long-range graph
            Z_exc = Z_fwd[:, self.m.conn_phase, :].sum(2)   # [B, N, D]

            if self.with_inhibition:
                Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            else:
                Z_inh = torch.zeros_like(Z)

            Z_new = (Z_struct
                     + self.alpha_exc * Z_exc
                     + self.m.alpha_turing * Z_inh)
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


class SGNNET_Interneuron(nn.Module):
    """Interneuron fraction wrapper for SGNNET_Resonant.

    Parameters
    ----------
    base_model    : SGNNET_Resonant
    n_input       : number of neurons that receive input (first n_input)
                    remaining N - n_input are interneurons (zero seed)
    readout_from  : 'all' or 'interneurons'
    """

    def __init__(self, base_model: SGNNET_Resonant, n_input: int,
                 readout_from: str = "all"):
        super().__init__()
        self.m            = base_model
        self.n_input      = n_input
        self.readout_from = readout_from

        if readout_from == "interneurons":
            N_h = base_model.base.N_hidden
            mask = torch.zeros(N_h, dtype=torch.float32)
            mask[n_input:] = 1.0
            self.register_buffer("readout_mask", mask)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Seed all neurons, then zero interneurons
        Z         = self.m.base._seed(x)                              # [B, N, D]
        Z[:, self.n_input:, :] = 0.0                                  # interneurons blank

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z_new    = Z_struct + self.m.alpha_turing * Z_inh
            Z        = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        if self.readout_from == "interneurons":
            C_ho     = self.m.base.C_ho_mask.float()
            C_masked = C_ho * self.readout_mask.unsqueeze(1)
            A_out    = torch.einsum("bhd,ho->bod", Z, C_masked)
            W_out    = self.m.base.W_pos[self.m.base.N_hidden:]
            return (A_out * F.normalize(W_out, dim=-1).unsqueeze(0)).sum(dim=-1)
        else:
            return self.m.base._readout(Z)


# SGNNET_FastPhase imported from src.sgnnet.mechanisms_excitatory above.


# ── Compound: phase_exc + interneurons (no fast_phase) ───────────────────────

class SGNNET_PhaseExcInterneuron(nn.Module):
    """Phase excitatory + interneurons in a single routing loop.

    Combines in one forward pass:
      - Interneuron masking: first n_input neurons seeded; remaining start at Z=0.
      - W_phase teleportation excitation: excites via K-NN phase graph.
      - Dynamic Z geo inhibition (alpha_turing): retained from base.

    NOT nested wrappers — one loop, all mechanisms combined.
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        n_input:    int,
        alpha_exc:  float = 0.3,
    ):
        super().__init__()
        self.m         = base_model
        self.n_input   = n_input
        self.alpha_exc = alpha_exc

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)                                      # [B, N, D]

        # Interneurons: neurons [n_input:] receive no direct input
        Z[:, self.n_input:] = 0.0

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)       # [B, N, D]

            # W_phase teleportation: phase-similar neurons excite each other
            Z_exc = Z_fwd[:, self.m.conn_phase, :].sum(2)            # [B, N, D]

            # Dynamic Z geo inhibition
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            Z = F.normalize(
                (Z_struct
                 + self.alpha_exc * Z_exc
                 + self.m.alpha_turing * Z_inh
                ).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


# ── Triple compound: phase_exc + interneurons + fast_W_phase ─────────────────

class SGNNET_TripleCompound(nn.Module):
    """Triple compound: phase_exc + interneurons + fast_W_phase in one forward pass.

    Single routing loop combines all three Gen1 mechanisms (no nested wrappers):
      1. Z[:, n_input:] = 0.0 at start  (interneurons)
      2. Z_struct from conn_hh           (structural aggregation)
      3. Z_exc = Z_fwd[:, conn_phase, :].sum(2) * alpha_exc  (phase teleportation)
      4. Fast attention update on A      (attention rule, tau, beam=32)
      5. Z_retrieved * alpha_fast
      6. Z_inh from _phase_inhibit
      7. Z = normalize(Z_struct + Z_exc + Z_retrieved + Z_inh)
      8. Update A to track Z cluster centroids

    Parameters
    ----------
    base_model  : SGNNET_Resonant
    n_input     : interneuron split (first n_input neurons receive direct input)
    alpha_exc   : phase excitatory strength
    alpha_fast  : fast W_phase attention contribution weight
    tau_fast    : attention softmax temperature
    beam_att    : beam size for sparse attention (0 = full O(N^2))
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        n_input:    int,
        alpha_exc:  float = 0.3,
        alpha_fast: float = 0.1,
        tau_fast:   float = 0.25,
        beam_att:   int   = 32,
    ):
        super().__init__()
        self.m          = base_model
        self.n_input    = n_input
        self.alpha_exc  = alpha_exc
        self.alpha_fast = alpha_fast
        self.tau_fast   = tau_fast
        self.beam_att   = beam_att

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        B, N_, D_ = Z.shape

        # 1. Interneurons: zero out non-input neurons before routing starts
        Z[:, self.n_input:] = 0.0

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]

        # Fast W_phase: per-forward-pass copy (slow prior initialises fast weight A)
        A = W_ph_norm.clone()                                         # [N, D]

        for _ in range(self.m.base.K_iter):
            # 2. Z_struct: aggregate from structural conn_hh graph
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)       # [B, N, D]

            # 3. Z_exc: phase teleportation via conn_phase K-NN excitation
            Z_exc = Z_fwd[:, self.m.conn_phase, :].sum(2)            # [B, N, D]

            # 4+5. Fast W_phase attention: sparse self-attention with A as key matrix
            Z_n    = F.normalize(Z, dim=-1)                           # [B, N, D]
            scores = torch.einsum('bnd,md->bnm', Z_n, A) / self.tau_fast  # [B, N, N]
            if 0 < self.beam_att < N_:
                topk_v, topk_i = scores.topk(self.beam_att, dim=-1)
                att_mask = torch.full_like(scores, float('-inf'))
                att_mask.scatter_(-1, topk_i, topk_v)
                weights = F.softmax(att_mask, dim=-1)
            else:
                weights = F.softmax(scores, dim=-1)
            Z_retrieved = torch.bmm(weights, Z)                       # [B, N, D]

            # 6. Z_inh: dynamic Z geo inhibition
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # 7. Combine all contributions and normalise
            Z = F.normalize(
                (Z_struct
                 + self.alpha_exc  * Z_exc
                 + self.alpha_fast * Z_retrieved
                 + self.m.alpha_turing * Z_inh
                ).clamp(-10, 10),
                dim=-1,
            )

            # 8. Update A to track Z cluster centroids (batch-averaged attention output)
            A = F.normalize(
                torch.einsum('bnm,bmd->bnd', weights,
                             F.normalize(Z, dim=-1)).mean(0),
                dim=-1,
            )

        return self.m.base._readout(Z)


# ── Training helper ───────────────────────────────────────────────────────────

def run(label: str, model: nn.Module, meta: dict, n_epochs: int) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=n_epochs, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=n_epochs)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    result  = {
        "label":            label,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta":            run_metadata(__file__, {**meta, "epochs": n_epochs}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})  "
          f"diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# ── Phase 1: Calibration ──────────────────────────────────────────────────────

def run_calibration() -> tuple[dict, dict]:
    """Run calibration sweeps (CALIB_EPOCHS each) and return (results, best_params).

    Returns
    -------
    calib_results : dict of key -> result dict
    best_params   : {
        "exc_alpha":   best alpha for phase_exc,
        "fp_alpha":    best alpha for fast_W_phase,
        "fp_tau":      best tau   for fast_W_phase,
        "int_n_input": best n_input for interneurons,
        "int_frac":    float fraction label (0.25 or 0.50),
    }
    """
    print("\n" + "#" * 70)
    print("  PHASE 1: CALIBRATION  (D=64 hyperparameter range search)")
    print(f"  Each config runs {CALIB_EPOCHS} epochs — just ranking, not final numbers.")
    print(f"  Rationale: cosine sims concentrate near zero on S^63 vs S^15;")
    print(f"  D=16 alpha values are NOT valid at D=64 without recalibration.")
    print("#" * 70)

    calib_results: dict[str, dict] = {}

    # ── Phase excitatory alpha sweep ──────────────────────────────────────────
    print("\n--- Phase excitatory alpha sweep ---")
    exc_alphas  = [0.1, 0.3, 0.5, 1.0]
    exc_keys    = ["calib_exc_01", "calib_exc_03", "calib_exc_05", "calib_exc_10"]
    exc_scores: dict[str, float] = {}

    for key, alpha in zip(exc_keys, exc_alphas):
        label = (f"calib phase_exc  alpha={alpha:.1f}  D=64  K_iter={K_ITER}"
                 f"  [{CALIB_EPOCHS}ep]")
        base  = make_resonant().to(DEVICE)
        model = SGNNET_PhaseExcitatory(base, alpha_exc=alpha, with_inhibition=True)
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "mechanism": "phase_exc",
                 "alpha_exc": alpha, "phase": "calibration"}
        r = run(label, model, meta, n_epochs=CALIB_EPOCHS)
        r.update(meta)
        calib_results[key] = r
        exc_scores[key] = r["top1_best"]

    best_exc_key   = max(exc_scores, key=lambda k: exc_scores[k])
    best_exc_alpha = exc_alphas[exc_keys.index(best_exc_key)]

    # ── Fast W_phase alpha/tau sweep ──────────────────────────────────────────
    print("\n--- Fast W_phase (attention) alpha/tau sweep ---")
    fp_configs = [
        ("calib_fp_a01", 0.1, 0.25),   # D=16 winner
        ("calib_fp_a03", 0.3, 0.25),
        ("calib_fp_t05", 0.1, 0.5),
        ("calib_fp_t10", 0.1, 1.0),
    ]
    fp_scores: dict[str, float] = {}

    for key, alpha, tau in fp_configs:
        label = (f"calib fast_W_phase  alpha={alpha:.1f}  tau={tau:.2f}"
                 f"  D=64  K_iter={K_ITER}  [{CALIB_EPOCHS}ep]")
        base  = make_resonant().to(DEVICE)
        model = SGNNET_FastPhase(base, fast_rule="attention",
                                 alpha_fast=alpha, tau=tau, beam_att=32)
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "mechanism": "fast_W_phase",
                 "alpha_fast": alpha, "tau": tau, "phase": "calibration"}
        r = run(label, model, meta, n_epochs=CALIB_EPOCHS)
        r.update(meta)
        calib_results[key] = r
        fp_scores[key] = r["top1_best"]

    best_fp_key = max(fp_scores, key=lambda k: fp_scores[k])
    best_fp_cfg = next((a, t) for (k, a, t) in fp_configs if k == best_fp_key)
    best_fp_alpha, best_fp_tau = best_fp_cfg

    # ── Interneuron fraction sweep ────────────────────────────────────────────
    print("\n--- Interneuron fraction sweep (25% vs 50%) ---")
    int_configs = [
        ("calib_int_25", N * 3 // 4, 0.25),   # 25% interneurons
        ("calib_int_50", N // 2,     0.50),   # 50% interneurons — D=16 winner
    ]
    int_scores: dict[str, float] = {}

    for key, n_input, frac in int_configs:
        label = (f"calib interneurons  frac={frac:.0%}  n_input={n_input}"
                 f"  readout=all  D=64  K_iter={K_ITER}  [{CALIB_EPOCHS}ep]")
        base  = make_resonant().to(DEVICE)
        model = SGNNET_Interneuron(base, n_input=n_input, readout_from="all")
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "mechanism": "interneurons",
                 "n_input": n_input, "int_frac": frac, "readout": "all",
                 "phase": "calibration"}
        r = run(label, model, meta, n_epochs=CALIB_EPOCHS)
        r.update(meta)
        calib_results[key] = r
        int_scores[key] = r["top1_best"]

    best_int_key     = max(int_scores, key=lambda k: int_scores[k])
    best_int_cfg     = next((ni, f) for (k, ni, f) in int_configs if k == best_int_key)
    best_int_n_input = best_int_cfg[0]
    best_int_frac    = best_int_cfg[1]

    # ── Calibration summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CALIBRATION SUMMARY — best hyperparameters found at D=64")
    print("=" * 70)

    print(f"\n  Phase excitatory alpha sweep  (D=64  K_iter={K_ITER}  {CALIB_EPOCHS}ep):")
    print(f"  {'Key':<18}  {'alpha':>6}  {'top1_best':>10}")
    print("  " + "-" * 40)
    for k, a in zip(exc_keys, exc_alphas):
        marker = "  <-- BEST" if k == best_exc_key else ""
        print(f"  {k:<18}  {a:6.1f}  {exc_scores[k]:10.4f}{marker}")
    print(f"\n  => best_exc_alpha = {best_exc_alpha}")

    print(f"\n  Fast W_phase alpha/tau sweep  (D=64  K_iter={K_ITER}  {CALIB_EPOCHS}ep):")
    print(f"  {'Key':<18}  {'alpha':>6}  {'tau':>6}  {'top1_best':>10}")
    print("  " + "-" * 48)
    for key, alpha, tau in fp_configs:
        marker = "  <-- BEST" if key == best_fp_key else ""
        print(f"  {key:<18}  {alpha:6.1f}  {tau:6.2f}  {fp_scores[key]:10.4f}{marker}")
    print(f"\n  => best_fp_alpha = {best_fp_alpha}  best_fp_tau = {best_fp_tau}")

    print(f"\n  Interneurons fraction sweep  (D=64  K_iter={K_ITER}  {CALIB_EPOCHS}ep  readout=all):")
    print(f"  {'Key':<18}  {'frac':>6}  {'n_input':>8}  {'top1_best':>10}")
    print("  " + "-" * 52)
    for key, n_input, frac in int_configs:
        marker = "  <-- BEST" if key == best_int_key else ""
        print(f"  {key:<18}  {frac:6.0%}  {n_input:8d}  {int_scores[key]:10.4f}{marker}")
    print(f"\n  => best_int_n_input = {best_int_n_input}"
          f"  ({best_int_frac:.0%} interneurons)")

    best_params = {
        "exc_alpha":   best_exc_alpha,
        "fp_alpha":    best_fp_alpha,
        "fp_tau":      best_fp_tau,
        "int_n_input": best_int_n_input,
        "int_frac":    best_int_frac,
    }
    return calib_results, best_params


# ── Phase 2: Full runs ────────────────────────────────────────────────────────

def run_full(best_params: dict) -> dict:
    """Run full FULL_EPOCHS validation using calibrated hyperparameters."""
    exc_alpha   = best_params["exc_alpha"]
    fp_alpha    = best_params["fp_alpha"]
    fp_tau      = best_params["fp_tau"]
    int_n_input = best_params["int_n_input"]
    int_frac    = best_params["int_frac"]

    print("\n" + "#" * 70)
    print("  PHASE 2: FULL RUNS  (D=64  K_iter=8  calibrated hyperparameters)")
    print(f"  exc_alpha={exc_alpha}  fp_alpha={fp_alpha}  fp_tau={fp_tau}")
    print(f"  int_n_input={int_n_input}  ({int_frac:.0%} interneurons)")
    print(f"  Each config runs {FULL_EPOCHS} epochs.")
    print("#" * 70)

    full_results: dict[str, dict] = {}

    configs = [
        (
            "Ref",
            f"Ref  D=64 K_iter={K_ITER} vanilla              [sanity check ~56.28%]",
            lambda: make_resonant(),
            {"mechanism": "vanilla"},
        ),
        (
            "A",
            f"A    + phase_exc alpha={exc_alpha}                    [calibrated for D=64]",
            lambda: SGNNET_PhaseExcitatory(
                make_resonant(), alpha_exc=exc_alpha, with_inhibition=True),
            {"mechanism": "phase_exc", "alpha_exc": exc_alpha},
        ),
        (
            "B",
            f"B    + interneurons {int_frac:.0%}  n_input={int_n_input}  readout=all"
            f"  [calibrated for D=64]",
            lambda: SGNNET_Interneuron(
                make_resonant(), n_input=int_n_input, readout_from="all"),
            {"mechanism": "interneurons", "n_input": int_n_input, "int_frac": int_frac},
        ),
        (
            "C",
            f"C    + fast_W_phase alpha={fp_alpha}  tau={fp_tau}"
            f"  [calibrated for D=64]",
            lambda: SGNNET_FastPhase(
                make_resonant(), fast_rule="attention",
                alpha_fast=fp_alpha, tau=fp_tau, beam_att=32),
            {"mechanism": "fast_W_phase", "alpha_fast": fp_alpha, "tau": fp_tau},
        ),
        (
            "D",
            f"D    + phase_exc alpha={exc_alpha} + interneurons {int_frac:.0%}"
            f"  [compound]",
            lambda: SGNNET_PhaseExcInterneuron(
                make_resonant(), n_input=int_n_input, alpha_exc=exc_alpha),
            {"mechanism": "phase_exc+interneurons",
             "alpha_exc": exc_alpha, "n_input": int_n_input},
        ),
        (
            "E",
            f"E    + phase_exc alpha={exc_alpha} + interneurons {int_frac:.0%}"
            f" + fast_W_phase alpha={fp_alpha} tau={fp_tau}  [triple compound]",
            lambda: SGNNET_TripleCompound(
                make_resonant(), n_input=int_n_input,
                alpha_exc=exc_alpha, alpha_fast=fp_alpha,
                tau_fast=fp_tau, beam_att=32),
            {"mechanism": "phase_exc+interneurons+fast_W_phase",
             "alpha_exc": exc_alpha, "n_input": int_n_input,
             "alpha_fast": fp_alpha, "tau": fp_tau},
        ),
    ]

    for key, label, factory, extra_meta in configs:
        model = factory().to(DEVICE)
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "phase": "full", **extra_meta}
        r     = run(label, model, meta, n_epochs=FULL_EPOCHS)
        r.update(meta)
        full_results[key] = r

    return full_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}  D={D}  N={N}  K_iter={K_ITER}  encoding=fourier")
    print(f"CALIB_EPOCHS={CALIB_EPOCHS}  FULL_EPOCHS={FULL_EPOCHS}")
    print("Goal: validate Gen1 mechanism winners at D=64 with recalibrated hyperparams")
    print("Rationale: cosine sims concentrate near zero on S^63 vs S^15 —")
    print("           D=16 alpha values NOT valid at D=64 without recalibration.")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    # ── Phase 1: calibration ──────────────────────────────────────────────────
    calib_results, best_params = run_calibration()

    print("\n" + "=" * 70)
    print("  Calibration complete. Best params for D=64:")
    for k, v in best_params.items():
        print(f"    {k} = {v}")
    print("  Proceeding to full runs...")
    print("=" * 70)

    # ── Phase 2: full runs with calibrated params ─────────────────────────────
    full_results = run_full(best_params)

    # ── Save ──────────────────────────────────────────────────────────────────
    all_results = {
        "calibration": calib_results,
        "full":        full_results,
        "best_params": best_params,
    }
    out = ROOT / "results" / "train_step29b_mechanisms_d64.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved -> {out}")

    # ── Final summary table ───────────────────────────────────────────────────
    ref_score = full_results.get("Ref", {}).get("top1_best", 0.5628)
    print(f"\n-- Phase 2 full results  (ref={ref_score:.4f}  expected~56.28%) ------")
    print("  %-72s  %9s  %+8s  %8s  %6s" % (
        "Config", "top1_best", "vs_ref", "ep_frac", "t(s)"))
    print("  " + "-" * 110)
    for k, r in full_results.items():
        d = r["top1_best"] - ref_score
        print("  %-72s  %9.4f  %+8.4f  %7.1f%%  %6.0f" % (
            r["label"][:72], r["top1_best"], d,
            r.get("best_epoch_frac", 0) * 100, r["elapsed_s"]))

    print(f"\n-- Calibration winners at D=64 ------------------------------------")
    print(f"  phase_exc    best alpha  = {best_params['exc_alpha']}"
          f"  (D=16 winner was 0.3)")
    print(f"  fast_W_phase best alpha  = {best_params['fp_alpha']}"
          f"  tau = {best_params['fp_tau']}"
          f"  (D=16 winner was alpha=0.1, tau=0.25)")
    print(f"  interneurons best frac   = {best_params['int_frac']:.0%}"
          f"  (n_input={best_params['int_n_input']})"
          f"  (D=16 winner was 50%)")

    print("\n  Interpretation:")
    print("  vs_ref > +1pp  -> mechanism helps at D=64 with calibrated params")
    print("  vs_ref ~ 0     -> mechanism is neutral at D=64")
    print("  vs_ref < -1pp  -> mechanism hurts at D=64 (do NOT include in Gen4)")
