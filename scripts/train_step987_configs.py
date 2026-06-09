"""step987 config builders — imported by train_step987_dw_phasegate_t0.py.

Configs:
  Ref                  — canonical ΔW-proj (imported from step985, bit-identical)
  A_compound_mul       — ΔW-proj × PhaseGate multiplicative
  B_compound_add       — ΔW-proj + lambda * PhaseGate additive
  C_gate_only_perneighbor — PhaseGate per-neighbor only (ablation)

Advance threshold: ≥+0.5pp vs Ref on T0.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_dw_phasegate   import SGNNET_DWPhaseGate
from scripts.train_step985_configs   import RefModel   # bit-identical Ref

# ── Constants (must match main script) ────────────────────────────────────────
N       = 2048; N_IN = 25088; N_OUT = 10
D       = 16;   K_HH = 2;     K_IN  = 25; K_ITER = 5
ALPHA_REFLECT = 0.5
N_GROUPS = max(8, N // 8)
SEED    = 42


# ── Base factory ──────────────────────────────────────────────────────────────
def _make_base(seed: int = SEED) -> SGNNET_SmallWorld:
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    return SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=N_GROUPS, norm_mode="l2", encoding_mode="fourier",
    )


# ── Config table ──────────────────────────────────────────────────────────────
def make_configs(seed: int = SEED) -> dict:
    return {
        "Ref": lambda: RefModel(seed),
        "A_compound_mul": lambda: SGNNET_DWPhaseGate(
            _make_base(seed), alpha_reflect=ALPHA_REFLECT,
            compound_mode="mul", alpha_mode="random"),
        "B_compound_add": lambda: SGNNET_DWPhaseGate(
            _make_base(seed), alpha_reflect=ALPHA_REFLECT,
            compound_mode="add", alpha_mode="random"),
        "C_gate_only_perneighbor": lambda: SGNNET_DWPhaseGate(
            _make_base(seed), alpha_reflect=ALPHA_REFLECT,
            compound_mode="gate_only", alpha_mode="random"),
    }
