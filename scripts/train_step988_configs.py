"""step988 config builders — imported by train_step988_perneighbor_gate_t0.py.

Configs:
  Ref                — canonical ΔW-proj (step199 baseline, control)
  A_softmax_gate     — pure per-neighbor softmax (replaces ΔW-proj, no compound)
  B_softmax_gate_dw  — compound: softmax(s) * |ΔW-proj|, renorm
                       (same signal path — expect noise per compounding rule)
  C_temperature      — per-neighbor softmax with learned temperature tau (init=1.0)

Compounding note: B is on the same signal path as ΔW-proj.
Isolation ablation (A) must be understood before interpreting B.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.model_perneighbor_gate import SGNNET_PerNeighborGate

# ── Constants ─────────────────────────────────────────────────────────────────
N       = 2048; N_IN = 25088; N_OUT = 10
D       = 16;   K_HH = 2;     K_IN  = 25; K_ITER = 5
ALPHA_REFLECT = 0.5
N_GROUPS = max(8, N // 8)


# ── ΔW-proj helpers (Ref only) ────────────────────────────────────────────────
def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


# ── Base factory ──────────────────────────────────────────────────────────────
def _make_base(seed: int = 42) -> SGNNET_SmallWorld:
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    return SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=N_GROUPS, norm_mode="l2", encoding_mode="fourier",
    )


# ── Ref: canonical ΔW-proj (step199) ─────────────────────────────────────────
class RefModel(nn.Module):
    def __init__(self, seed: int = 42):
        super().__init__()
        base = _make_base(seed)
        self.m = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=0.0, beam_size=16, mode="dynamic_z_geo",
        )
        self.W_phase = None

    @property
    def W_pos(self): return self.m.W_pos

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        dw        = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref     = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


# ── Config table ──────────────────────────────────────────────────────────────
def make_configs(seed: int = 42) -> dict:
    return {
        "Ref": lambda: RefModel(seed),
        "A_softmax_gate": lambda: SGNNET_PerNeighborGate(
            _make_base(seed),
            alpha_reflect=ALPHA_REFLECT,
            dw_compound=False,
            temperature=None,
        ),
        "B_softmax_gate_dw": lambda: SGNNET_PerNeighborGate(
            _make_base(seed),
            alpha_reflect=ALPHA_REFLECT,
            dw_compound=True,
            temperature=None,
        ),
        "C_temperature": lambda: SGNNET_PerNeighborGate(
            _make_base(seed),
            alpha_reflect=ALPHA_REFLECT,
            dw_compound=False,
            temperature=1.0,
        ),
    }
