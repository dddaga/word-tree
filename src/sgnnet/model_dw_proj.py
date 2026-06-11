"""DeltaW ΔW-proj model: wraps Resonant with ΔW-proj routing forward pass.
Used by step982 (CIFAR-10 aug T2) and step980 (CIFAR-10 multi-seed T2).
Canonical ΔW-proj implementation matching step887/step980 results.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:conn_hh.shape[0]]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class DeltaWModel(nn.Module):
    """ΔW-proj routing wrapper around SGNNET_Resonant."""
    def __init__(self, resonant: SGNNET_Resonant, K_iter: int, alpha_reflect: float):
        super().__init__()
        self.m = resonant
        self.K_iter = K_iter
        self.alpha_reflect = alpha_reflect

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
        for _ in range(self.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = self.alpha_reflect * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def build_dw_model(N, N_IN, N_OUT, D, K_HH, K_IN, K_ITER, ALPHA_REFLECT, SEED, DEVICE):
    """Build and return DeltaWModel for given config."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    model = DeltaWModel(resonant, K_ITER, ALPHA_REFLECT).to(DEVICE)
    return torch.compile(model) if DEVICE.type == "cuda" else model
