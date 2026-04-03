"""Step 29c: Re-validate ALL Gen1 mechanism winners at D=64 using calibrated base params.

MOTIVATION
==========
Steps 29 and 29b ran mechanism sweeps against a D=16-calibrated base
(alpha_turing=0.3, K_phase=8, beam_size=32, alpha_reflect=0.3, geo_gamma=1.0).
Step 22b finds the TRUE D=64-optimal routing params via a full grid search.

Routing params and mechanism params interact: a mechanism gain measured against
a suboptimal base cannot be trusted. Both must be co-calibrated at D=64.

This script:
  1. Loads step22b's best routing params (CALIB_PARAMS).
  2. Phase 1 (40ep each): calibrates each mechanism's own alpha/tau/fraction
     against the calibrated base.
  3. Phase 2 (150ep each): runs Ref + each mechanism + 2-way / 3-way / full
     compound using calibrated base + calibrated mechanism params.

CONFIGS
=======
Phase 1 calibration:
  AntiHebb alpha sweep     : 0.1, 0.3, 0.5 (D=16 winner), 0.7, 1.0
  Phase excitatory alpha   : 0.1, 0.3 (D=16 winner), 0.5, 1.0
  Fast W_phase alpha/tau   : (0.1/0.25 D=16 win), (0.3/0.25), (0.1/0.5), (0.1/1.0)
  Interneuron fraction     : 25%, 50% (D=16 winner), 75%

Phase 2 full runs:
  Ref   : calibrated base, no extra mechanisms
  A     : + AntiHebb
  B     : + phase_exc
  C     : + interneurons (readout=all)
  D     : + fast_W_phase
  E     : + AntiHebb + phase_exc
  F     : + AntiHebb + phase_exc + interneurons
  G     : + AntiHebb + phase_exc + interneurons + fast_W_phase  [full compound]

All: D=64 N=1024 K_iter=8 Fourier dynamic_z_geo 150ep plateau SEED=42 store.h5.

To reproduce:
    python -u scripts/train_step29c_mechanisms_calibrated.py --device mps
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.mechanisms_excitatory import SGNNET_FastPhase


# ── Step 22b calibrated base params ───────────────────────────────────────────

STEP22B_RESULTS = Path("results/train_step22b_routing_calib_d64.json")


def load_calibrated_params() -> dict:
    """Load best routing params from step22b calibration."""
    if not STEP22B_RESULTS.exists():
        raise FileNotFoundError(
            f"step22b results not found at {STEP22B_RESULTS}. "
            "Run train_step22b_routing_calib_d64.py first."
        )
    data = json.loads(STEP22B_RESULTS.read_text())
    best = data["best_params"]
    print(f"Loaded calibrated base from step22b:")
    for k, v in best.items():
        print(f"  {k} = {v}")
    return best


CALIB_PARAMS = load_calibrated_params()


# ── CLI & constants ────────────────────────────────────────────────────────────

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


# ── Model factory ──────────────────────────────────────────────────────────────

def make_resonant(N=1024, D=64, K_iter=8) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"], K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"], norm_mode="l2", D=D, encoding_mode="fourier")
    return SGNNET_Resonant(base=base,
        K_phase=CALIB_PARAMS["K_phase"],
        beam_size=CALIB_PARAMS["beam_size"],
        theta_init=0.1,
        alpha_reflect=CALIB_PARAMS["alpha_reflect"],
        alpha_turing=CALIB_PARAMS["alpha_turing"],
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
        geo_gamma=CALIB_PARAMS["geo_gamma"])


# ── Mechanism classes (inlined from step29b) ───────────────────────────────────

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
# SGNNET_AntiHebbian imported from src.sgnnet.mechanisms_inhibitory above.


# ── Compound classes (single routing loops — no nested wrappers) ───────────────

class SGNNET_AntiHebb_PhaseExc(nn.Module):
    """AntiHebb suppression + phase teleportation in one routing loop.

    Pre-computes W_pos similarity for AntiHebb suppression weights (static,
    outside the loop). Phase excitatory adds K_phase K-NN excitation.
    Dynamic Z geo inhibition (alpha_turing) retained from base.

    Parameters
    ----------
    base_model   : SGNNET_Resonant
    alpha_ahebb  : AntiHebbian suppression strength
    alpha_exc    : phase excitatory weight
    """

    def __init__(
        self,
        base_model:  SGNNET_Resonant,
        alpha_ahebb: float = 0.3,
        alpha_exc:   float = 0.3,
    ):
        super().__init__()
        self.m           = base_model
        self.alpha_ahebb = alpha_ahebb
        self.alpha_exc   = alpha_exc

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]
        conn_hh   = self.m.base.conn_hh

        # Pre-compute W_pos similarity for AntiHebb (static, outside loop)
        N_h   = self.m.base.N_hidden
        W_n   = F.normalize(self.m.W_pos[:N_h], dim=-1)              # [N, D]
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)          # [N, K_hh]
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)                       # [1, N, K_hh, 1]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]                          # [B, N, K_hh, D]
            Z_struct = (Z_nb * supp_w).sum(2)                        # AntiHebb aggregation

            # Phase teleportation: excite via K_phase K-NN phase graph
            Z_exc = Z_fwd[:, self.m.conn_phase, :].sum(2)            # [B, N, D]

            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            Z = F.normalize(
                (Z_struct
                 + self.alpha_exc    * Z_exc
                 + self.m.alpha_turing * Z_inh
                ).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


class SGNNET_AntiHebb_PhaseExc_Intern(nn.Module):
    """AntiHebb suppression + phase teleportation + interneuron blanking.

    Adds interneuron masking (neurons [n_input:] start at Z=0) on top of
    the AntiHebb+PhaseExc compound, in a single routing loop.

    Parameters
    ----------
    base_model   : SGNNET_Resonant
    n_input      : first n_input neurons receive direct input; rest are interneurons
    alpha_ahebb  : AntiHebbian suppression strength
    alpha_exc    : phase excitatory weight
    """

    def __init__(
        self,
        base_model:  SGNNET_Resonant,
        n_input:     int,
        alpha_ahebb: float = 0.3,
        alpha_exc:   float = 0.3,
    ):
        super().__init__()
        self.m           = base_model
        self.n_input     = n_input
        self.alpha_ahebb = alpha_ahebb
        self.alpha_exc   = alpha_exc

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        Z[:, self.n_input:] = 0.0                                     # interneurons blank

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]
        conn_hh   = self.m.base.conn_hh

        # Pre-compute W_pos similarity for AntiHebb (static, outside loop)
        N_h     = self.m.base.N_hidden
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)            # [N, D]
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)          # [N, K_hh]
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)                       # [1, N, K_hh, 1]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]                          # [B, N, K_hh, D]
            Z_struct = (Z_nb * supp_w).sum(2)                        # AntiHebb aggregation

            # Phase teleportation: excite via K_phase K-NN phase graph
            Z_exc = Z_fwd[:, self.m.conn_phase, :].sum(2)            # [B, N, D]

            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            Z = F.normalize(
                (Z_struct
                 + self.alpha_exc      * Z_exc
                 + self.m.alpha_turing * Z_inh
                ).clamp(-10, 10),
                dim=-1,
            )

        return self.m.base._readout(Z)


class SGNNET_FullCompound(nn.Module):
    """Full compound: AntiHebb + phase teleportation + interneurons + fast_W_phase.

    All four Gen1 mechanisms combined in a single routing loop:
      1. Z[:, n_input:] = 0.0               (interneurons)
      2. Z_struct = (Z_nb * supp_w).sum(2)  (AntiHebb suppressed aggregation)
      3. Z_exc = Z_fwd[:, conn_phase, :].sum(2) * alpha_exc  (phase teleportation)
      4. Fast attention update on A          (attention rule, tau, beam=32)
      5. Z_inh from _phase_inhibit
      6. Z = normalize(Z_struct + Z_exc + Z_retrieved + Z_inh)
      7. Update A to track Z cluster centroids

    Parameters
    ----------
    base_model   : SGNNET_Resonant
    n_input      : interneuron split
    alpha_ahebb  : AntiHebbian suppression strength
    alpha_exc    : phase excitatory weight
    alpha_fast   : fast W_phase attention contribution weight
    tau_fast     : attention softmax temperature
    beam_att     : sparse beam for attention (0=full O(N^2), >0=sparse)
    """

    def __init__(
        self,
        base_model:  SGNNET_Resonant,
        n_input:     int,
        alpha_ahebb: float = 0.3,
        alpha_exc:   float = 0.3,
        alpha_fast:  float = 0.1,
        tau_fast:    float = 0.25,
        beam_att:    int   = 32,
    ):
        super().__init__()
        self.m           = base_model
        self.n_input     = n_input
        self.alpha_ahebb = alpha_ahebb
        self.alpha_exc   = alpha_exc
        self.alpha_fast  = alpha_fast
        self.tau_fast    = tau_fast
        self.beam_att    = beam_att

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        B, N_, D_ = Z.shape

        # 1. Interneurons: neurons [n_input:] receive no direct input
        Z[:, self.n_input:] = 0.0

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]
        conn_hh   = self.m.base.conn_hh

        # Pre-compute W_pos similarity for AntiHebb (static, outside loop)
        N_h     = self.m.base.N_hidden
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)            # [N, D]
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)          # [N, K_hh]
        supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                   ).unsqueeze(0).unsqueeze(-1)                       # [1, N, K_hh, 1]

        # Fast W_phase: per-forward-pass copy of slow prior
        A = W_ph_norm.clone()                                         # [N, D]

        for _ in range(self.m.base.K_iter):
            # 2. Z_struct: AntiHebb suppressed aggregation from structural graph
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]                          # [B, N, K_hh, D]
            Z_struct = (Z_nb * supp_w).sum(2)                        # [B, N, D]

            # 3. Phase teleportation: excite via K_phase K-NN phase graph
            Z_exc = Z_fwd[:, self.m.conn_phase, :].sum(2)            # [B, N, D]

            # 4. Fast W_phase attention: sparse self-attention with A as key matrix
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

            # 5. Dynamic Z geo inhibition
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # 6. Combine all contributions and normalise
            Z = F.normalize(
                (Z_struct
                 + self.alpha_exc      * Z_exc
                 + self.alpha_fast     * Z_retrieved
                 + self.m.alpha_turing * Z_inh
                ).clamp(-10, 10),
                dim=-1,
            )

            # 7. Update A to track Z cluster centroids (batch-averaged attention output)
            A = F.normalize(
                torch.einsum('bnm,bmd->bnd', weights,
                             F.normalize(Z, dim=-1)).mean(0),
                dim=-1,
            )

        return self.m.base._readout(Z)


# ── Training helper ────────────────────────────────────────────────────────────

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


# ── Phase 1: Calibration ───────────────────────────────────────────────────────

def run_calibration() -> tuple[dict, dict]:
    """Run calibration sweeps (CALIB_EPOCHS each) against calibrated base.

    Returns
    -------
    calib_results : dict of key -> result dict
    best_params   : {
        "ah_alpha":    best alpha for AntiHebbian,
        "exc_alpha":   best alpha for phase_exc,
        "fp_alpha":    best alpha for fast_W_phase,
        "fp_tau":      best tau   for fast_W_phase,
        "fp_key":      key string for best fast_W_phase config,
        "int_n_input": best n_input for interneurons,
        "int_frac":    float fraction label,
    }
    """
    print("\n" + "#" * 70)
    print("  PHASE 1: CALIBRATION  (calibrated base from step22b)")
    print(f"  Each config runs {CALIB_EPOCHS} epochs — ranking only, not final numbers.")
    print(f"  Base routing params from step22b — true D=64-optimal base.")
    print(f"  Mechanisms also need recalibration: cosine sims concentrate on S^63.")
    print("#" * 70)

    calib_results: dict[str, dict] = {}

    # ── AntiHebb alpha sweep ───────────────────────────────────────────────────
    print("\n--- AntiHebb alpha sweep ---")
    ah_configs = [
        ("calib_ah_01", 0.1),
        ("calib_ah_03", 0.3),
        ("calib_ah_05", 0.5),   # D=16 winner
        ("calib_ah_07", 0.7),
        ("calib_ah_10", 1.0),
    ]
    calib_ah_results: dict[str, dict] = {}

    for key, alpha in ah_configs:
        label = (f"calib AntiHebb  alpha={alpha:.1f}  D=64  K_iter={K_ITER}"
                 f"  [{CALIB_EPOCHS}ep]")
        base  = make_resonant().to(DEVICE)
        model = SGNNET_AntiHebbian(base, alpha_ahebb=alpha, variant="wpos")
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "mechanism": "anti_hebb",
                 "alpha_ahebb": alpha, "phase": "calibration"}
        r = run(label, model, meta, n_epochs=CALIB_EPOCHS)
        r.update(meta)
        calib_results[key]    = r
        calib_ah_results[key] = r

    best_ah_alpha_key = max(calib_ah_results, key=lambda k: calib_ah_results[k]["top1_best"])
    best_ah_alpha     = dict(ah_configs)[best_ah_alpha_key]

    # ── Phase excitatory alpha sweep ───────────────────────────────────────────
    print("\n--- Phase excitatory alpha sweep ---")
    exc_configs = [
        ("calib_exc_01", 0.1),
        ("calib_exc_03", 0.3),   # D=16 winner
        ("calib_exc_05", 0.5),
        ("calib_exc_10", 1.0),
    ]
    calib_exc_results: dict[str, dict] = {}

    for key, alpha in exc_configs:
        label = (f"calib phase_exc  alpha={alpha:.1f}  D=64  K_iter={K_ITER}"
                 f"  [{CALIB_EPOCHS}ep]")
        base  = make_resonant().to(DEVICE)
        model = SGNNET_PhaseExcitatory(base, alpha_exc=alpha, with_inhibition=True)
        meta  = {"N": N, "D": D, "K_iter": K_ITER, "mechanism": "phase_exc",
                 "alpha_exc": alpha, "phase": "calibration"}
        r = run(label, model, meta, n_epochs=CALIB_EPOCHS)
        r.update(meta)
        calib_results[key]     = r
        calib_exc_results[key] = r

    best_exc_alpha_key = max(calib_exc_results, key=lambda k: calib_exc_results[k]["top1_best"])
    best_exc_alpha     = dict(exc_configs)[best_exc_alpha_key]

    # ── Fast W_phase alpha/tau sweep ───────────────────────────────────────────
    print("\n--- Fast W_phase (attention) alpha/tau sweep ---")
    fp_configs = [
        ("calib_fp_a01_t025", 0.1, 0.25),   # D=16 winner
        ("calib_fp_a03_t025", 0.3, 0.25),
        ("calib_fp_a01_t050", 0.1, 0.50),
        ("calib_fp_a01_t100", 0.1, 1.0),
    ]
    fp_lookup = {k: (a, t) for k, a, t in fp_configs}
    calib_fp_results: dict[str, dict] = {}

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
        calib_results[key]    = r
        calib_fp_results[key] = r

    best_fp_key   = max(calib_fp_results, key=lambda k: calib_fp_results[k]["top1_best"])
    best_fp_alpha, best_fp_tau = fp_lookup[best_fp_key]

    # ── Interneuron fraction sweep ─────────────────────────────────────────────
    print("\n--- Interneuron fraction sweep (25% / 50% / 75%) ---")
    int_configs = [
        ("calib_int_25", N * 3 // 4, 0.25),   # 25% interneurons
        ("calib_int_50", N // 2,     0.50),   # 50% interneurons — D=16 winner
        ("calib_int_75", N // 4,     0.75),   # 75% interneurons
    ]
    int_lookup = {k: (ni, f) for k, ni, f in int_configs}
    calib_int_results: dict[str, dict] = {}

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
        calib_results[key]     = r
        calib_int_results[key] = r

    best_int_frac_key                = max(calib_int_results,
                                           key=lambda k: calib_int_results[k]["top1_best"])
    best_int_n_input, best_int_frac  = int_lookup[best_int_frac_key]

    # ── Calibration summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CALIBRATION SUMMARY — best hyperparameters at D=64 (calibrated base)")
    print("=" * 70)

    print(f"\n  AntiHebb alpha sweep  (D=64  K_iter={K_ITER}  {CALIB_EPOCHS}ep):")
    print(f"  {'Key':<20}  {'alpha':>6}  {'top1_best':>10}")
    print("  " + "-" * 42)
    for key, alpha in ah_configs:
        marker = "  <-- BEST" if key == best_ah_alpha_key else ""
        print(f"  {key:<20}  {alpha:6.1f}  "
              f"{calib_ah_results[key]['top1_best']:10.4f}{marker}")
    print(f"\n  => best_ah_alpha = {best_ah_alpha}"
          f"  (D=16 winner was 0.5)")

    print(f"\n  Phase excitatory alpha sweep  (D=64  K_iter={K_ITER}  {CALIB_EPOCHS}ep):")
    print(f"  {'Key':<20}  {'alpha':>6}  {'top1_best':>10}")
    print("  " + "-" * 42)
    for key, alpha in exc_configs:
        marker = "  <-- BEST" if key == best_exc_alpha_key else ""
        print(f"  {key:<20}  {alpha:6.1f}  "
              f"{calib_exc_results[key]['top1_best']:10.4f}{marker}")
    print(f"\n  => best_exc_alpha = {best_exc_alpha}"
          f"  (D=16 winner was 0.3)")

    print(f"\n  Fast W_phase alpha/tau sweep  (D=64  K_iter={K_ITER}  {CALIB_EPOCHS}ep):")
    print(f"  {'Key':<22}  {'alpha':>6}  {'tau':>6}  {'top1_best':>10}")
    print("  " + "-" * 52)
    for key, alpha, tau in fp_configs:
        marker = "  <-- BEST" if key == best_fp_key else ""
        print(f"  {key:<22}  {alpha:6.1f}  {tau:6.2f}  "
              f"{calib_fp_results[key]['top1_best']:10.4f}{marker}")
    print(f"\n  => best_fp_alpha = {best_fp_alpha}  best_fp_tau = {best_fp_tau}"
          f"  (D=16 winner was alpha=0.1, tau=0.25)")

    print(f"\n  Interneurons fraction sweep  (D=64  K_iter={K_ITER}  {CALIB_EPOCHS}ep"
          f"  readout=all):")
    print(f"  {'Key':<20}  {'frac':>6}  {'n_input':>8}  {'top1_best':>10}")
    print("  " + "-" * 52)
    for key, n_input, frac in int_configs:
        marker = "  <-- BEST" if key == best_int_frac_key else ""
        print(f"  {key:<20}  {frac:6.0%}  {n_input:8d}  "
              f"{calib_int_results[key]['top1_best']:10.4f}{marker}")
    print(f"\n  => best_int_n_input = {best_int_n_input}"
          f"  ({best_int_frac:.0%} interneurons)"
          f"  (D=16 winner was 50%)")

    best_mechanism_params = {
        "ah_alpha":    best_ah_alpha,
        "exc_alpha":   best_exc_alpha,
        "fp_alpha":    best_fp_alpha,
        "fp_tau":      best_fp_tau,
        "fp_key":      best_fp_key,
        "int_n_input": best_int_n_input,
        "int_frac":    best_int_frac,
    }
    return calib_results, best_mechanism_params


# ── Phase 2: Full runs ─────────────────────────────────────────────────────────

def run_full(best_mech: dict) -> dict:
    """Run full FULL_EPOCHS validation using calibrated base + calibrated mechanism params."""
    ah_alpha    = best_mech["ah_alpha"]
    exc_alpha   = best_mech["exc_alpha"]
    fp_alpha    = best_mech["fp_alpha"]
    fp_tau      = best_mech["fp_tau"]
    int_n_input = best_mech["int_n_input"]
    int_frac    = best_mech["int_frac"]

    print("\n" + "#" * 70)
    print("  PHASE 2: FULL RUNS  (D=64  K_iter=8  calibrated base + mechanisms)")
    print(f"  Base routing from step22b: {CALIB_PARAMS}")
    print(f"  Mechanism params:  ah_alpha={ah_alpha}  exc_alpha={exc_alpha}")
    print(f"                     fp_alpha={fp_alpha}  fp_tau={fp_tau}")
    print(f"                     int_n_input={int_n_input}  ({int_frac:.0%} interneurons)")
    print(f"  Each config runs {FULL_EPOCHS} epochs.")
    print("#" * 70)

    full_results: dict[str, dict] = {}

    configs = [
        (
            "Ref",
            "Ref   calibrated base, no mechanisms                 [post-step22b ceiling]",
            lambda: make_resonant(),
            {"mechanism": "vanilla"},
        ),
        (
            "A",
            f"A     + AntiHebb alpha={ah_alpha}",
            lambda: SGNNET_AntiHebbian(make_resonant(), alpha_ahebb=ah_alpha, variant="wpos"),
            {"mechanism": "anti_hebb", "alpha_ahebb": ah_alpha},
        ),
        (
            "B",
            f"B     + phase_exc alpha={exc_alpha}",
            lambda: SGNNET_PhaseExcitatory(
                make_resonant(), alpha_exc=exc_alpha, with_inhibition=True),
            {"mechanism": "phase_exc", "alpha_exc": exc_alpha},
        ),
        (
            "C",
            f"C     + interneurons {int_frac:.0%}  n_input={int_n_input}  readout=all",
            lambda: SGNNET_Interneuron(
                make_resonant(), n_input=int_n_input, readout_from="all"),
            {"mechanism": "interneurons", "n_input": int_n_input, "int_frac": int_frac},
        ),
        (
            "D",
            f"D     + fast_W_phase alpha={fp_alpha}  tau={fp_tau}",
            lambda: SGNNET_FastPhase(
                make_resonant(), fast_rule="attention",
                alpha_fast=fp_alpha, tau=fp_tau, beam_att=32),
            {"mechanism": "fast_W_phase", "alpha_fast": fp_alpha, "tau": fp_tau},
        ),
        (
            "E",
            f"E     + AntiHebb alpha={ah_alpha} + phase_exc alpha={exc_alpha}",
            lambda: SGNNET_AntiHebb_PhaseExc(
                make_resonant(), alpha_ahebb=ah_alpha, alpha_exc=exc_alpha),
            {"mechanism": "anti_hebb+phase_exc",
             "alpha_ahebb": ah_alpha, "alpha_exc": exc_alpha},
        ),
        (
            "F",
            f"F     + AntiHebb alpha={ah_alpha} + phase_exc alpha={exc_alpha}"
            f" + interneurons {int_frac:.0%}",
            lambda: SGNNET_AntiHebb_PhaseExc_Intern(
                make_resonant(), n_input=int_n_input,
                alpha_ahebb=ah_alpha, alpha_exc=exc_alpha),
            {"mechanism": "anti_hebb+phase_exc+interneurons",
             "alpha_ahebb": ah_alpha, "alpha_exc": exc_alpha,
             "n_input": int_n_input, "int_frac": int_frac},
        ),
        (
            "G",
            f"G     + AntiHebb alpha={ah_alpha} + phase_exc alpha={exc_alpha}"
            f" + interneurons {int_frac:.0%} + fast_W_phase alpha={fp_alpha}"
            f" tau={fp_tau}  [FULL GEN4]",
            lambda: SGNNET_FullCompound(
                make_resonant(), n_input=int_n_input,
                alpha_ahebb=ah_alpha, alpha_exc=exc_alpha,
                alpha_fast=fp_alpha, tau_fast=fp_tau, beam_att=32),
            {"mechanism": "anti_hebb+phase_exc+interneurons+fast_W_phase",
             "alpha_ahebb": ah_alpha, "alpha_exc": exc_alpha,
             "n_input": int_n_input, "int_frac": int_frac,
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}  D={D}  N={N}  K_iter={K_ITER}  encoding=fourier")
    print(f"CALIB_EPOCHS={CALIB_EPOCHS}  FULL_EPOCHS={FULL_EPOCHS}")
    print("Goal: re-validate ALL Gen1 mechanism winners against calibrated D=64 base")
    print("Rationale: step29/29b used D=16-calibrated routing params; step22b finds")
    print("           the true D=64-optimal base. Mechanism gains vs. suboptimal base")
    print("           are unreliable — routing params and mechanism params interact.")
    print(f"\nCalibrated base params (from step22b):")
    for k, v in CALIB_PARAMS.items():
        print(f"  {k} = {v}")
    get_loaders()
    print(f"\nDataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    # ── Phase 1: calibration ───────────────────────────────────────────────────
    calib_results, best_mech_params = run_calibration()

    print("\n" + "=" * 70)
    print("  Calibration complete. Best mechanism params for D=64 calibrated base:")
    for k, v in best_mech_params.items():
        print(f"    {k} = {v}")
    print("  Proceeding to full runs...")
    print("=" * 70)

    # ── Phase 2: full runs with calibrated base + calibrated mechanism params ──
    full_results = run_full(best_mech_params)

    # ── Save results ───────────────────────────────────────────────────────────
    all_results = {
        "calibrated_base_params": CALIB_PARAMS,
        "calibration":            calib_results,
        "best_mechanism_params":  best_mech_params,
        "full":                   full_results,
    }
    out = ROOT / "results" / "train_step29c_mechanisms_calibrated.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved -> {out}")

    # ── Final summary table ────────────────────────────────────────────────────
    ref_score     = full_results.get("Ref",  {}).get("top1_best", 0.5628)
    step22e_score = 0.5628   # step22 config E — D=64 ceiling reference
    step22b_f_score = CALIB_PARAMS.get("best_top1", ref_score)  # step22b full run F

    print(f"\n-- Phase 2 full results  (ref={ref_score:.4f}  "
          f"step22E={step22e_score:.4f}  "
          f"step22b_F~{step22b_f_score:.4f}) ---")
    print("  %-72s  %9s  %+8s  %+10s  %+9s  %8s  %6s" % (
        "Config", "top1", "vs_Ref",
        "vs_step22E", "vs_22b_F", "ep_frac", "t(s)"))
    print("  " + "-" * 130)
    for k, r in full_results.items():
        d_ref    = r["top1_best"] - ref_score
        d_22e    = r["top1_best"] - step22e_score
        d_22bf   = r["top1_best"] - step22b_f_score
        print("  %-72s  %9.4f  %+8.4f  %+10.4f  %+9.4f  %7.1f%%  %6.0f" % (
            r["label"][:72], r["top1_best"], d_ref, d_22e, d_22bf,
            r.get("best_epoch_frac", 0) * 100, r["elapsed_s"]))

    g_score = full_results.get("G", {}).get("top1_best", 0.0)
    print(f"\n  Gen4 compound ceiling: G = {g_score:.4f} ({g_score*100:.2f}%)")

    print(f"\n-- Calibrated mechanism params (vs D=16 defaults) ---")
    print(f"  anti_hebb     best alpha = {best_mech_params['ah_alpha']}"
          f"  (D=16 winner was 0.5)")
    print(f"  phase_exc     best alpha = {best_mech_params['exc_alpha']}"
          f"  (D=16 winner was 0.3)")
    print(f"  fast_W_phase  best alpha = {best_mech_params['fp_alpha']}"
          f"  tau = {best_mech_params['fp_tau']}"
          f"  (D=16 winner was alpha=0.1, tau=0.25)")
    print(f"  interneurons  best frac  = {best_mech_params['int_frac']:.0%}"
          f"  (n_input={best_mech_params['int_n_input']})"
          f"  (D=16 winner was 50%)")

    print("\n  Interpretation:")
    print("  vs_Ref > +1pp per mechanism -> additive at D=64 calibrated base")
    print("  vs_Ref ~ 0                  -> neutral — do not include in Gen4 combo")
    print("  vs_Ref < -1pp               -> interference — exclude from Gen4")
    print("  G > step22E (56.28%)        -> true ceiling improvement over vanilla D=64")
