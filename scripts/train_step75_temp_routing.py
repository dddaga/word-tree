"""Step 75: Input-modulated temperature routing.

MOTIVATION
==========
Step 73 tests softmax redistribution with a FIXED temperature τ applied uniformly
to all neurons. This experiment asks: does a PER-NEURON temperature that adapts
to the current activation state add further benefit?

Key insight: in a mixture-of-experts view, some neurons should route sharply
(route to the single best neighbor) while others route diffusely (spread signal).
A fixed τ forces the same routing style on all neurons. Per-neuron temperature
allows specialization: high-activity neurons route decisively, low-activity neurons
diffuse.

Implementation:
    temp_h = τ_0 × (1 + σ(dot(W_temp_h, Z_h)))   — W_temp: [N, D], per-neuron learned
    w[h,j] = softmax(score(Z_h, Z_nb_j) / temp_h + logit_AH[h,j], dim=j)

W_temp initialised to zero → σ(0)=0.5 → temp_h = 1.5×τ_0 at init (uniform baseline).
W_temp is a learnable parameter added to the optimizer via param_group injection
(same pattern as step77 theta fix).

CONFIGS (N=1024, D=64, K_iter=8, Gen4+ params, 50%/75ep)
==========================================================
  Ref : AH sum routing (control = step69 Ref 83.36%)
  A   : fixed τ=1.0  softmax + AH logit  (= step73 Config B — cross-validation)
  B   : fixed τ=0.3  softmax + AH logit  (= step73 Config D — sharper)
  C   : input-mod τ_0=1.0  (W_temp learned at lr_wpos)
  D   : input-mod τ_0=0.3  (W_temp learned, sharper baseline)

ABLATION AXIS
=============
  A vs Ref : does fixed-τ softmax help? (cross-check with step73)
  C vs A   : does per-neuron temperature add to fixed-τ benefit?
  D vs B   : same question at sharper τ
  C vs D   : τ_0 sensitivity for learned temperature

To reproduce:
    python -u scripts/train_step75_temp_routing.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import (
    trainer_kwargs, topology_kwargs, run_metadata, GA_BEST,
)
from src.training.dataset              import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = 75
BATCH     = 128
SEED      = 42
DATA      = "data/store.h5"
N         = 1024
D         = 64

K_PHASE       = 8
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

LR_WPOS    = GA_BEST["lr_Wpos"]   # 2.364e-3
STEP69_REF = 0.8336

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(
            subset, batch_size=BATCH, shuffle=True, num_workers=0
        )
        _loaders = (tr, va)
    return _loaders


# ── Temperature routing wrapper ────────────────────────────────────────────────

class SGNNET_TempRouting(nn.Module):
    """Softmax routing with fixed or input-modulated per-neuron temperature.

    Fixed mode (learnable_temp=False):
        w = softmax((Z_dot / τ_0) + AH_logit, dim=K_hh)

    Input-modulated mode (learnable_temp=True):
        temp_h = τ_0 × (1 + σ(dot(W_temp_h, Z_h)))   — Z-dependent temperature
        w = softmax((Z_dot / temp_h) + AH_logit, dim=K_hh)

    W_temp is initialised to 0: σ(0)=0.5 → temp_h = 1.5×τ_0 at init.
    W_temp must be added to optimizer externally (see run() below).
    """

    def __init__(
        self,
        resonant: SGNNET_Resonant,
        alpha_ahebb: float = 1.0,
        tau_0: float = 1.0,
        learnable_temp: bool = False,
    ):
        super().__init__()
        self.m              = resonant
        self.alpha_ahebb    = alpha_ahebb
        self.tau_0          = tau_0
        self.learnable_temp = learnable_temp

        if learnable_temp:
            N_h = resonant.base.N_hidden
            D_  = resonant.base.W_pos.shape[1]
            # Init to 0: uniform temperature at start, learns per-neuron preference
            self.W_temp = nn.Parameter(torch.zeros(N_h, D_))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        N_h       = self.m.base.N_hidden
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)   # [N_h, K_hh]
        ah_logit = -self.alpha_ahebb * pos_sim.clamp(min=0)    # [N_h, K_hh]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)           # [B, N, D]
            Z_nb  = Z_fwd[:, conn_hh, :]            # [B, N, K_hh, D]

            # Z-state score: dot(Z_h, Z_nb_j)
            z_score = (Z_fwd.unsqueeze(2) * Z_nb).sum(-1)   # [B, N, K_hh]

            if self.learnable_temp:
                # Per-neuron temperature from current activation
                # dot(W_temp_h, Z_fwd_h) → sigmoid → scale
                temp_score = (self.W_temp * Z_fwd).sum(-1)   # [B, N]
                temp = self.tau_0 * (1.0 + torch.sigmoid(temp_score))  # [B, N], ≥ τ_0
                logit = z_score / temp.unsqueeze(-1) + ah_logit.unsqueeze(0)
            else:
                logit = z_score / self.tau_0 + ah_logit.unsqueeze(0)

            w        = F.softmax(logit, dim=2)                # [B, N, K_hh]
            Z_struct = (w.unsqueeze(-1) * Z_nb).sum(dim=2)   # [B, N, D]

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            if self.m.alpha_turing != 0.0:
                W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
                Z_inh     = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
                Z_new     = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh
            else:
                Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Model factory ──────────────────────────────────────────────────────────────

def make_resonant(seed_offset: int = 0) -> SGNNET_Resonant:
    torch.manual_seed(SEED + seed_offset)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=8, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model_ref(seed_offset: int = 0) -> SGNNET_AntiHebbian:
    return SGNNET_AntiHebbian(make_resonant(seed_offset), alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def make_model_temp(tau_0: float, learnable: bool, seed_offset: int = 0) -> SGNNET_TempRouting:
    return SGNNET_TempRouting(make_resonant(seed_offset), alpha_ahebb=ALPHA_AHEBB,
                              tau_0=tau_0, learnable_temp=learnable)


# ── Training loop ──────────────────────────────────────────────────────────────

def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)

    # Inject W_temp into optimizer for learnable-temperature configs
    if hasattr(model, "W_temp"):
        trainer.optimizer.add_param_group({
            "params": [model.W_temp],
            "lr": LR_WPOS,
            "weight_decay": 0.0,  # same treatment as W_pos
        })
        print(f"  W_temp: LEARNABLE at lr={LR_WPOS:.3e}  shape={tuple(model.W_temp.shape)}")
    else:
        print(f"  W_temp: none (fixed temperature)")

    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best      = max(top1_hist)
    best_ep   = int(np.argmax(top1_hist)) + 1
    frac      = best_ep / len(history)
    delta     = best - STEP69_REF

    result = {
        "label":            label,
        "top1_best":        best,
        "top1_last":        history[-1].get("val_top1", 0.0),
        "final_task_loss":  float(np.mean([h.get("task_loss", 0.0) for h in history[-5:]])),
        "best_epoch":       best_ep,
        "epochs_run":       len(history),
        "elapsed_s":        round(elapsed, 1),
        "best_epoch_frac":  round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history":     top1_hist,
        "step69_ref":       STEP69_REF,
        "delta_vs_ref":     round(delta, 4),
        "_meta":            run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(
        f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
        f"  vs_ref={delta:+.4f}  t={elapsed:.0f}s"
    )
    return result


# ── Configs ────────────────────────────────────────────────────────────────────

CONFIGS = [
    # (key, label, tau_0, learnable, seed_offset)
    ("Ref", "Ref  AH sum routing (control)",
     1.0, False, 0),
    ("A",   "A    fixed τ=1.0 softmax + AH logit  (step73 B cross-check)",
     1.0, False, 1),
    ("B",   "B    fixed τ=0.3 softmax + AH logit  (step73 D cross-check)",
     0.3, False, 2),
    ("C",   "C    input-mod τ_0=1.0  (W_temp learned — main hypothesis)",
     1.0, True,  3),
    ("D",   "D    input-mod τ_0=0.3  (W_temp learned, sharper baseline)",
     0.3, True,  4),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 75: Input-modulated temperature routing")
    print(f"Ref: step69 = {STEP69_REF:.4f}")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results: dict = {}
    out_path = ROOT / "results" / "train_step75_temp_routing.json"

    for key, label, tau_0, learnable, seed_off in CONFIGS:
        if key == "Ref":
            model = make_model_ref(seed_offset=seed_off).to(DEVICE)
        else:
            model = make_model_temp(tau_0, learnable, seed_offset=seed_off).to(DEVICE)
        meta = {
            "N": N, "D": D, "K_iter": 8,
            "tau_0": tau_0, "learnable_temp": learnable,
            "alpha_ahebb": ALPHA_AHEBB, "alpha_reflect": ALPHA_REFLECT,
            "data_frac": 0.5,
        }
        results[key] = run(label, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")

    print(f"\n{'='*70}")
    print(f"STEP 75 COMPLETE — Input-modulated temperature routing")
    print(f"Ref = {STEP69_REF:.4f}")
    print()
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs_ref':>8s}  {'tau_0':>6s}  {'learned':>8s}")
    for key, label, tau_0, learnable, _ in CONFIGS:
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:4s}  {r['top1_best']:.4f}    {r['delta_vs_ref']:+.4f}"
              f"  {tau_0:>6.1f}  {'yes' if learnable else 'no':>8s}")
    print()
    print("  C > A: per-neuron temperature adds to fixed-τ benefit")
    print("  C < A: uniform softmax is optimal (all neurons same routing style)")
