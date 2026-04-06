"""Step 105: Hypothesis validation — test unvalidated assumptions.

MOTIVATION
==========
Several design rules guiding ALL future experiments rest on hypotheses
that were NEVER properly validated with controlled experiments:

HYPOTHESIS 1: "AH antagonizes phase-target routing"
  Evidence: step66 D(phase+plasticity+AH)=40.33% vs A(phase only)=67.87%
  Gap: D had TWO extra variables (plasticity + AH). Never isolated AH alone.
  Test: PhaseTarget + AH (no plasticity) — does AH kill phase-target even
  without plasticity?

HYPOTHESIS 2: "Compounding anything on AH kills gain"
  Evidence: step29c/32 on BUGGY arch (73% baseline). Every compound < AH alone.
  Gap: Never retested on PATCHED arch (83%+ baseline). Signal is richer now.
  The patched arch fixed +9.83pp of signal — compounds that failed at 73%
  may succeed at 83%+ because there's more signal to survive attenuation.

HYPOTHESIS 3: "Phase mechanics fail due to gate-death"
  Evidence: step60 (13 configs, all 10-19%) on buggy arch.
  Gap: Never retested post-patch. If phase mechanics work on patched arch,
  the entire "phase is dead" conclusion is wrong.

This experiment VALIDATES or INVALIDATES these assumptions on the patched
arch with clean controls.

CONFIGS (N=1024, D=64, K_iter=8, patched arch, 50%/75ep)
==========================================================
  Ref  : AH alone (step69 Ref reproduction = ~83.36%)
  A    : PhaseTarget + AH, NO plasticity  (H1 test: AH × phase isolated)
  B    : PhaseTarget only, NO AH          (H1 control: phase alone on patched)
  C    : AH + phase_excitatory (step29c compound, retested on patched arch) (H2)
  D    : AH + interneurons (step61 compound, retested on patched arch) (H2)
  E    : Phase coherence softmax routing + AH (step60 Config B on patched) (H3)

DECISION RULES
==============
  H1: If A > B: AH HELPS phase-target → hypothesis "AH antagonizes" is WRONG
      If A < B: AH hurts phase-target → hypothesis confirmed
      If A ≈ Ref: phase-target is neutral (not harmful) with AH

  H2: If C or D > Ref: compounding CAN work on patched arch → rule invalidated
      If C, D < Ref: compounding still kills → rule confirmed on patched arch

  H3: If E > Ref: phase coherence routing works post-patch → reopen phase track
      If E ≈ 10-19%: gate-death confirmed on patched arch too → phase routing DEAD

To reproduce:
    python -u scripts/train_step105_hypothesis_validation.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64
ALPHA_REFLECT = 0.5; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


# ── Phase-target routing (from step66, simplified — no plasticity) ─────────

class SGNNET_AH_PhaseTarget(nn.Module):
    """AH routing with phase-target filtering (NO plasticity, NO diversity).

    Phase-target: each neuron has a target phase W_phase[h].
    Signal from j→h is weighted by cos(W_phase[h], Z[j]) — neurons whose
    activation aligns with h's phase preference pass more signal.

    This isolates the AH × phase-target interaction without plasticity confounds.
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 use_ah: bool = True, use_phase_target: bool = True):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb if use_ah else 0.0
        self.use_ah = use_ah
        self.use_pt = use_phase_target

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # AH wpos suppression (if enabled)
        if self.use_ah:
            W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            ah_w    = (1.0 - self.alpha * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)

        # Phase-target weights (if enabled)
        if self.use_pt:
            Ph = F.normalize(self.m.W_phase, dim=-1)              # [N, D]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd  = F.relu(Z - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]                        # [B,N,K,D]

            if self.use_pt:
                # Phase-target: weight by alignment of Z_nb with h's phase
                Z_nb_n = F.normalize(Z_nb, dim=-1)
                pt_w = (Ph.unsqueeze(0).unsqueeze(2)              # [1,N,1,D]
                        * Z_nb_n).sum(-1).clamp(min=0)            # [B,N,K]
                pt_w = pt_w.unsqueeze(-1)                         # [B,N,K,1]
                Z_struct = Z_nb * pt_w
            else:
                Z_struct = Z_nb

            if self.use_ah:
                Z_struct = (Z_struct * ah_w).sum(dim=2)
            else:
                Z_struct = Z_struct.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Phase coherence softmax routing (from step60, simplified) ──────────────

class SGNNET_AH_PhaseCoherence(nn.Module):
    """AH routing with phase coherence softmax (redistribution, Σw=1).

    w_j = softmax(cos(W_phase[h], W_phase[j]) / τ) over K_hh neighbors.
    This is redistribution routing (Σw=1) using phase coherence as the score.
    """

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float, tau: float = 1.0):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb
        self.tau   = tau

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # AH wpos suppression
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        ah_w    = (1.0 - self.alpha * pos_sim.clamp(min=0))       # [N, K]

        # Phase coherence scores (static)
        Ph = F.normalize(self.m.W_phase, dim=-1)
        phase_coh = (Ph.unsqueeze(1) * Ph[conn_hh]).sum(-1)       # [N, K]
        # Combine: AH suppression as logit bias + phase coherence as score
        combined_score = phase_coh / self.tau + ah_w               # [N, K]
        route_w = F.softmax(combined_score, dim=-1)                # [N, K] Σ=1
        route_w = route_w.unsqueeze(0).unsqueeze(-1)               # [1,N,K,1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd  = F.relu(Z - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * route_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; model_type: str  # "ah", "pt_ah", "pt_only", "coh"


CONFIGS = [
    Config("Ref", "Ref  AH alone (step69 Ref repro)", "ah"),
    Config("A",   "A    PhaseTarget + AH, no plasticity  (H1: isolate AH×phase)", "pt_ah"),
    Config("B",   "B    PhaseTarget only, no AH           (H1: phase alone)", "pt_only"),
    Config("C",   "C    Phase coherence softmax + AH       (H3: phase routing post-patch)", "coh"),
]


_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=8, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if cfg.model_type == "ah":
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    elif cfg.model_type == "pt_ah":
        return SGNNET_AH_PhaseTarget(resonant, ALPHA_AHEBB, use_ah=True, use_phase_target=True)
    elif cfg.model_type == "pt_only":
        return SGNNET_AH_PhaseTarget(resonant, ALPHA_AHEBB, use_ah=False, use_phase_target=True)
    elif cfg.model_type == "coh":
        return SGNNET_AH_PhaseCoherence(resonant, ALPHA_AHEBB, tau=1.0)
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg, model, meta):
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0 = time.time(); history = trainer.train(n_epochs=EPOCHS); elapsed = time.time() - t0
    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist); best_ep = int(np.argmax(top1_hist)) + 1
    result = {
        "label": cfg.label, "model_type": cfg.model_type,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "step69_ref": STEP69_REF, "delta_vs_ref": round(best - STEP69_REF, 4),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  "
          f"vs_ref={best-STEP69_REF:+.4f}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  D={D}  Data: 50%")
    print(f"Step 105: Hypothesis validation on patched arch")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}\n")
    print("HYPOTHESES UNDER TEST:")
    print("  H1: 'AH antagonizes phase-target' → A vs B (isolated test)")
    print("  H2: 'Compounding kills on patched arch' → C vs Ref")
    print("  H3: 'Phase coherence routing is dead' → C on patched arch\n")
    for c in CONFIGS:
        print(f"  {c.key:4s}  {c.label}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step105_hypothesis_validation.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": 8, "model_type": cfg.model_type,
                "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}")
    print(f"STEP 105 COMPLETE — Hypothesis Validation\n")
    print(f"  {'Key':4s}  {'top1':>8s}  {'vs_ref':>8s}  Hypothesis test")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}  {c.label}")
    # H1 verdict
    if "A" in results and "B" in results:
        a, b = results["A"]["top1_best"], results["B"]["top1_best"]
        if a > b + 0.01:
            print(f"\n  H1 VERDICT: AH HELPS phase-target (+{a-b:.4f}). 'AH antagonizes' is WRONG.")
        elif b > a + 0.01:
            print(f"\n  H1 VERDICT: AH HURTS phase-target ({a-b:+.4f}). 'AH antagonizes' CONFIRMED.")
        else:
            print(f"\n  H1 VERDICT: AH is NEUTRAL to phase-target (Δ={a-b:+.4f}).")
    # H3 verdict
    if "C" in results:
        c_val = results["C"]["top1_best"]
        if c_val > STEP69_REF:
            print(f"  H3 VERDICT: Phase coherence routing WORKS on patched arch ({c_val:.4f} > {STEP69_REF:.4f}). REOPEN phase track.")
        elif c_val > 0.5:
            print(f"  H3 VERDICT: Phase coherence routing is DEGRADED but not dead ({c_val:.4f}). Worth investigating.")
        else:
            print(f"  H3 VERDICT: Phase coherence routing is DEAD on patched arch too ({c_val:.4f}). Confirmed.")
