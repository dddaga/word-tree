"""Step 25: Beam-gated routing — activation propagation only for top-K neurons.

MOTIVATION & THEORY
====================
Current routing step costs (B=128, N=512, K_hh=6, D=16, rad_beam=32):
  Threshold (Z_fwd=relu(Z-θ)):  B×N×D        = 1.0M mults  [unavoidable]
  Structural (Z_struct):         B×N×K_hh×D   = 6.3M mults  [15%]
  Radiation (_phase_inhibit):    B×rad×N×D    = 33.6M mults  [80%] ← BOTTLENECK
  Normalize:                     B×N×D        = 1.0M mults  [2%]
  TOTAL per step ≈ 42M, × K_iter=3 → 126M FLOPs

KEY INSIGHT: Radiation is O(rad_beam × N). If we restrict routing to top-K_route
neurons and radiation to ONLY within that beam:
  Radiation: B × K_route × K_route × D = O(K_route²)
  At K_route=32: 32×32×16 = 16K vs 32×512×16 = 262K → 16× cheaper per step

Total with K_route=32, K_iter=8:
  8 × [1.0M + 32×6×16/1M + 32×32×16/1M] ≈ 8 × [1.04M] ≈ 8.3M
  vs K_iter=3 full: 126M FLOPs → ~15× reduction WITH more iterations.

This IS measurably faster on MPS because the dominant bmm shrinks from
[B, N, rad_beam] → [B, K_route, rad_beam] (smaller matmul → better GPU utilisation).

DYNAMIC BEAM (start wide, narrow each step)
============================================
Step 1: K_route=128 — broad sweep, many neurons compete
Step 2: K_route=64  — surviving signals concentrate
Step 3: K_route=32  — final refinement on most class-relevant subset
This mirrors biological "winner-take-all" dynamics in cortical columns.

NON-BEAM NEURONS: carry forward their previous Z (don't reset to 0).
Their gradients still flow to the input seed via the Z carry-forward path.
This is "hard attention" — selection is non-differentiable but Z updates are.

CONFIGS (D=16 N=512 Fourier, 90ep plateau, measure time+FLOPs):
  Ref : K_iter=3 route=512 (full N, baseline)
  A   : K_iter=3 route=128
  B   : K_iter=3 route=64
  C   : K_iter=3 route=32
  D   : K_iter=8 route=32   ← primary hypothesis: sparse-deep cheaper+better
  E   : K_iter=5 route=32
  F   : dynamic [128→64→32] K_iter=3
  G   : dynamic [256→64→16] K_iter=3
  H   : K_iter=8 route=16   ← very narrow beam, many iters

All: 90ep (efficiency sweep, not accuracy maximisation).
Reference: step13 K_iter=3 full = 29.04%  K_iter=8 full = 36.69%

To reproduce:
    python -u scripts/train_step25_beam_routing.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

EPOCHS = 90     # efficiency sweep — shorter to test more configs
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D, N   = 16, 512

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def theory_flops(K_iter: int, K_route: int,
                 N: int = 512, K_hh: int = 6, D: int = 16,
                 rad_beam: int = 32, B: int = 128,
                 beam_model: bool = True) -> int:
    """Theoretical mults per batch for K_iter routing steps.

    beam_model=True  (SGNNET_BeamGated):
      Threshold:   B × N × D                  [scan all N for beam selection, unavoidable]
      Structural:  B × K_route × K_hh × D     [only beam neurons receive]
      Radiation:   B × K_route² × D           [within-beam: O(K_route²)]

    beam_model=False (current SGNNET_Resonant):
      Threshold:   B × N × D
      Structural:  B × N × K_hh × D           [all N receive]
      Radiation:   B × rad_beam × N × D       [top-rad_beam radiate to all N]

    Crossover: beam radiation O(K_route²) < current O(rad_beam×N) when K_route < sqrt(rad_beam×N)
    sqrt(32×512) ≈ 128 → beam cheaper below route=128, more expensive above.
    """
    if beam_model:
        per_step = (B * N * D                      # threshold
                    + B * K_route * K_hh * D       # structural: beam only
                    + B * K_route * K_route * D)    # radiation: O(K_route²)
    else:
        per_step = (B * N * D                      # threshold
                    + B * N * K_hh * D             # structural: all N
                    + B * rad_beam * N * D)         # radiation: top-32 to all N
    return per_step * K_iter


def make_resonant(K_iter: int = 3) -> SGNNET_Resonant:
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


class SGNNET_BeamGated(nn.Module):
    """Beam-restricted routing: only top-K_route neurons update per routing step.

    Two efficiency gains vs full N routing:
    1. Structural gather: O(K_route × K_hh × D)  not  O(N × K_hh × D)
    2. Radiation (phase inhibit): O(K_route × rad_beam × D)  not  O(N × rad_beam × D)

    Non-beam neurons carry forward their current Z (no reset).
    Beam selection: top-K_route by Z_fwd L2 norm after threshold.

    beam_schedule: list of K_route values, one per routing step.
    Static:  [K] * K_iter
    Dynamic: [128, 64, 32] for K_iter=3 (coarse-to-fine)
    """

    def __init__(self, base: SGNNET_Resonant, beam_schedule: list[int]):
        super().__init__()
        self.m             = base
        self.beam_schedule = beam_schedule

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)               # [B, N, D]
        B, N_, D_ = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)   # [1,1,1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh                             # [N, K_hh]
        K_hh      = conn_hh.shape[1]
        rad_beam  = min(self.m.beam_size, N_)

        for K_route in self.beam_schedule:
            K_route = min(K_route, N_)

            # ── Threshold ──────────────────────────────────────────────────
            Z_fwd = F.relu(Z - theta_pos)              # [B, N, D]

            # ── Beam selection ─────────────────────────────────────────────
            norms     = Z_fwd.norm(dim=-1)             # [B, N]
            _, b_idx  = norms.topk(K_route, dim=-1)    # [B, K_route]

            # ── Structural routing (beam neurons only) ─────────────────────
            # Gather connection indices for beam neurons: [B, K_route, K_hh]
            b_conn    = conn_hh[b_idx.reshape(-1)].view(B, K_route, K_hh)
            # Gather Z_fwd values at those neighbor indices
            flat_nb   = b_conn.reshape(B, -1)          # [B, K_route*K_hh]
            Z_nb      = Z_fwd.gather(1, flat_nb.unsqueeze(-1).expand(-1,-1,D_))
            Z_struct_b = Z_nb.view(B, K_route, K_hh, D_).sum(2)    # [B, K_route, D]

            # ── Radiation within beam (beam→beam only, O(K_route²)) ─────────
            # Source and target are BOTH restricted to the beam.
            # W_phase rows for beam neurons serve as the phase query vectors.
            # Beam neuron h is inhibited by beam neuron j if cos(W_phase[h], Z[j]) > 0.
            # Cost: O(K_route² × D) — vs O(N × rad_beam × D) for full radiation.
            W_ph_b    = W_ph_norm[b_idx.reshape(-1)].view(B, K_route, D_)  # [B, K_route, D]
            Z_b       = Z.gather(1, b_idx.unsqueeze(-1).expand(-1,-1,D_))  # [B, K_route, D]
            cos_bm    = torch.bmm(W_ph_b, Z_b.transpose(1, 2))             # [B, K_route, K_route]
            alpha_t   = self.m.alpha_turing
            Z_inh_b   = -alpha_t * torch.bmm(cos_bm.clamp(min=0), Z_b)    # [B, K_route, D]

            # ── Compose and normalize beam neurons ────────────────────────
            Z_beam_new = F.normalize(
                (Z_struct_b + Z_inh_b).clamp(-10, 10), dim=-1)              # [B, K_route, D]

            # ── Scatter back: beam neurons get new Z, others carry forward ─
            beam_mask = torch.zeros(B, N_, 1, device=Z.device)
            beam_mask.scatter_(1, b_idx.unsqueeze(-1), 1.0)                 # [B, N, 1]
            Z_candidate = torch.zeros_like(Z)
            Z_candidate.scatter_(1, b_idx.unsqueeze(-1).expand(-1,-1,D_), Z_beam_new)
            Z = Z_candidate * beam_mask + Z * (1.0 - beam_mask)

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va   = get_loaders()
    tk       = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer  = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0       = time.time()
    history  = trainer.train(n_epochs=EPOCHS)
    elapsed  = time.time() - t0

    best    = max(h.get("val_top1", 0.0) for h in history)
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    flops   = meta.get("flops_M", 0)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss",0) for h in history[-5:]])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1), "best_epoch_frac": round(frac, 3),
        "top1_history": [round(h.get("val_top1",0.0),4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  "
          f"FLOPs={flops:.0f}M  t={elapsed:.0f}s ({elapsed/len(history):.1f}s/ep)")
    return result


# (label, K_iter, beam_schedule, meta_extra)
_f = lambda ki, kr: round(theory_flops(ki, kr) / 1e6, 1)
CONFIGS = [
    ("Ref. K_iter=3  route=512 (full N)",
     3, [512]*3,           {"route_k": 512, "flops_M": _f(3, 512)}),
    ("A.  K_iter=3  route=128",
     3, [128]*3,           {"route_k": 128, "flops_M": _f(3, 128)}),
    ("B.  K_iter=3  route=64",
     3, [64]*3,            {"route_k":  64, "flops_M": _f(3,  64)}),
    ("C.  K_iter=3  route=32",
     3, [32]*3,            {"route_k":  32, "flops_M": _f(3,  32)}),
    ("D.  K_iter=8  route=32  [sparse-deep]",
     8, [32]*8,            {"route_k":  32, "flops_M": _f(8,  32)}),
    ("E.  K_iter=5  route=32",
     5, [32]*5,            {"route_k":  32, "flops_M": _f(5,  32)}),
    ("F.  K_iter=3  dynamic [128→64→32]",
     3, [128, 64, 32],     {"route_k": "128→64→32", "flops_M": round(
         sum(theory_flops(1,k)/1e6 for k in [128,64,32]), 1)}),
    ("G.  K_iter=3  dynamic [256→64→16]",
     3, [256, 64, 16],     {"route_k": "256→64→16", "flops_M": round(
         sum(theory_flops(1,k)/1e6 for k in [256,64,16]), 1)}),
    ("H.  K_iter=8  route=16  [very narrow-deep]",
     8, [16]*8,            {"route_k":  16, "flops_M": _f(8,  16)}),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}")
    print("Goal: beam-restricted routing — accuracy vs compute tradeoff")
    print("Reference FLOPs: K_iter=3 full = {:.0f}M".format(_f(3, 512)))
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys    = ["Ref", "A", "B", "C", "D", "E", "F", "G", "H"]
    for key, (label, K_iter, schedule, meta) in zip(keys, CONFIGS):
        resonant = make_resonant(K_iter=K_iter).to(DEVICE)
        model    = (resonant if schedule == [512]*K_iter
                    else SGNNET_BeamGated(resonant, beam_schedule=schedule))
        meta["K_iter"] = K_iter
        meta["beam_schedule"] = str(schedule)
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step25_beam_routing.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref     = results.get("Ref", {}).get("top1_best", 0.2904)
    ref_f   = _f(3, 512)
    print(f"\n-- Beam routing sweep  (ref={ref:.4f}  ref_flops={ref_f:.0f}M) ------")
    print("  %-46s  %9s  %+8s  %7s  %7s  %6s" % (
        "Config", "top1", "vs_ref", "FLOPs_M", "speedup", "t(s)"))
    print("  " + "-"*90)
    for k, r in results.items():
        d  = r["top1_best"] - ref
        fm = r.get("flops_M", ref_f)
        sp = ref_f / fm if fm > 0 else 0
        print("  %-46s  %9.4f  %+8.4f  %7.1f  %7.1fx  %6.0f" % (
            r["label"][:46], r["top1_best"], d, fm, sp, r["elapsed_s"]))
