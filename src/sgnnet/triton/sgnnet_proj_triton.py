"""SGNNET_DeltaProjTriton — drop-in replacement for SGNNET_DeltaProj using the
Triton fused routing kernel.

Same forward signature as SGNNET_DeltaProj from train_step268.
Falls back to PyTorch reference if Triton unavailable.

Design notes:
- dw_norm is precomputed per epoch via tick_epoch() matching SGNNET_Resonant pattern
- conn_hh cast to int32 at init — avoids step803 CUDA graph int64 issue
- The Triton kernel receives (Z, theta, conn_hh_i32, dw_norm_3d, Z_refl, alpha)
  per routing step; tick_epoch re-derives dw_norm from current W_pos
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_smallworld import SGNNET_SmallWorld
from ..model_resonant   import SGNNET_Resonant
from .routing_kernel    import routing_step_proj, TRITON_AVAILABLE, _routing_step_proj_pytorch
from .routing_kernel_fused import (
    routing_fused_parallel,
    routing_fused_sequential,
    _routing_step_proj_pytorch_k_iter,
)


class SGNNET_DeltaProjTriton(nn.Module):
    """ΔW projection routing with Triton fused kernel.

    Equivalent to SGNNET_DeltaProj from train_step268_dwproj_aug_k4.py.
    Differences:
      1. Routing loop uses fused Triton kernel (or falls back to PyTorch).
      2. conn_hh is pre-cast to int32 to enable CUDA Graph capture.
      3. dw_norm is cached and recomputed on tick_epoch() (not every forward).
    """

    def __init__(
        self,
        N_hidden:     int   = 2048,
        N_out:        int   = 10,
        D_:           int   = 16,
        N_in:         int   = 25088,
        K_in:         int   = 25,
        K_iter:       int   = 5,
        K_local:      int   = 1,
        K_random:     int   = 1,
        n_groups:     int   = 256,
        alpha_reflect: float = 0.5,
        seed:         int   = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D_, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier",
        )
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0,
        )
        self.alpha_reflect = alpha_reflect
        self._use_triton = TRITON_AVAILABLE
        # Fused K_iter mode: "off" | "parallel" | "sequential"
        # "off"        — per-step Triton kernel (original step530, K_ITER launches)
        # "parallel"   — single launch, fixed-Z neighbor reads (approx semantics)
        # "sequential" — single launch, ping-pong buffers (approx, race-condition if concurrent blocks)
        self._fused_mode: str = "off"

        # Cast conn_hh to int32 once — fixes step803 CUDA graph int64 blocker
        # Register as non-parameter buffer so it moves with .to(device)
        conn_i32 = self.base.conn_hh.to(torch.int32)
        self.register_buffer("conn_hh_i32", conn_i32)

        # dw_norm cache: recomputed by tick_epoch(); None until first call
        self._dw_norm_cache: "torch.Tensor | None" = None

    # ------------------------------------------------------------------
    # Properties for trainer compatibility
    # ------------------------------------------------------------------

    @property
    def W_pos(self):
        return self.base.W_pos

    @property
    def W_phase(self):
        return self.resonant.W_phase

    # ------------------------------------------------------------------
    # Epoch hook — recompute dw_norm from current W_pos
    # ------------------------------------------------------------------

    def tick_epoch(self):
        """Recompute dw_norm from current W_pos (call once per epoch from Trainer)."""
        with torch.no_grad():
            self._recompute_dw_norm()
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def _recompute_dw_norm(self):
        """Derive unit delta-W vectors from W_pos: [N, K_hh, D]."""
        N_h      = self.base.N_hidden
        W_h      = self.base.W_pos[:N_h]                        # [N, D]
        conn     = self.base.conn_hh                            # [N, K_hh] int64
        delta_w  = W_h.unsqueeze(1) - W_h[conn]                # [N, K_hh, D]
        dw_norm  = F.normalize(delta_w, dim=-1)                 # [N, K_hh, D]
        self._dw_norm_cache = dw_norm.detach()

    def _get_dw_norm(self) -> "torch.Tensor":
        """Return cached dw_norm, computing on first call or when device has changed."""
        target_device = self.base.W_pos.device
        if (self._dw_norm_cache is None
                or self._dw_norm_cache.device != target_device):
            self._recompute_dw_norm()
        return self._dw_norm_cache  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        Z = self.base._seed(x)                                  # [B, N, D]

        theta_pos = self.resonant.theta.abs()                   # [N] positive threshold
        dw_norm   = self._get_dw_norm()                         # [N, K_hh, D]
        conn_i32  = self.conn_hh_i32                            # [N, K_hh] int32

        Z_reflected = torch.zeros_like(Z)

        if self._use_triton and self._fused_mode == "parallel":
            # Single kernel launch — K_iter parallel rounds, fixed-Z neighbor reads
            Z, Z_reflected = routing_fused_parallel(
                Z, theta_pos, conn_i32, dw_norm,
                Z_reflected, self.alpha_reflect,
                K_iter=self.base.K_iter,
            )
        elif self._use_triton and self._fused_mode == "sequential":
            # Single kernel launch — ping-pong buffers, approximate sequential
            Z, Z_reflected = routing_fused_sequential(
                Z, theta_pos, conn_i32, dw_norm,
                Z_reflected, self.alpha_reflect,
                K_iter=self.base.K_iter,
            )
        elif self._use_triton:
            # Default: per-step Triton kernel (original step530)
            for _ in range(self.base.K_iter):
                Z, Z_reflected = routing_step_proj(
                    Z, theta_pos, conn_i32, dw_norm,
                    Z_reflected, self.alpha_reflect,
                )
        else:
            # PyTorch fallback (also used for gradient computation if needed)
            theta_unsq = theta_pos.unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
            dw_unsq    = dw_norm.unsqueeze(0)                   # [1, N, K_hh, D]
            conn_i64   = self.base.conn_hh                      # [N, K_hh] int64

            for _ in range(self.base.K_iter):
                Z_fwd    = F.relu(Z - theta_unsq)
                Z_nb     = Z_fwd[:, conn_i64, :]
                proj     = (Z_nb * dw_unsq).sum(-1, keepdim=True)
                Z_nb     = Z_nb * proj.abs()
                Z_struct = Z_nb.sum(dim=2)

                Z_remainder  = Z_fwd - Z
                Z_reflected  = self.alpha_reflect * Z_reflected + Z_remainder
                Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)

        return self.base._readout(Z)
