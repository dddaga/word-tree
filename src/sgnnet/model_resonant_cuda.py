"""SGNNET_Resonant_CUDA: CUDA-optimized drop-in replacement for SGNNET_Resonant.

Targets the dynamic_z_geo mode with alpha_turing=0 (efficiency config path).

Optimizations applied:
  1. @torch.compile(mode="reduce-overhead")  — CUDA graph capture + kernel fusion
  2. Contiguous layout enforcement on conn_hh and Z before gather
  3. Inner loop body extracted to _routing_step() so compile sees the full loop
  4. alpha_turing=0 branch hardcoded — phase inhibition code removed entirely
  5. AH suppression baked into the same compiled graph (SGNNET_AntiHebbian_CUDA)

Drop-in compatibility:
  - Same __init__ signature as SGNNET_Resonant
  - Same forward(x) API
  - uncompile() method returns to eager mode for debugging
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld
from .model_resonant import SGNNET_Resonant


# ---------------------------------------------------------------------------
# Compiled inner loop helper (fused by torch.compile)
# ---------------------------------------------------------------------------

def _routing_step_no_turing(
    Z: torch.Tensor,           # [B, N, D]
    Z_reflected: torch.Tensor, # [B, N, D]
    conn_hh: torch.Tensor,     # [N, K_hh]
    theta_pos: torch.Tensor,   # [1, N, 1]
    alpha_reflect: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single routing iteration with alpha_turing=0 hardcoded.

    Returns (Z_new, Z_reflected_new) — both contiguous, l2-normalised Z.
    Extracted so torch.compile can fuse across the loop body.
    """
    Z_fwd = F.relu(Z - theta_pos)                           # [B, N, D]
    Z_struct = Z_fwd[:, conn_hh, :].sum(dim=2)              # [B, N, D]
    Z_remainder = Z_fwd - Z                                  # below-threshold residual
    Z_reflected_new = alpha_reflect * Z_reflected + Z_remainder
    Z_new = Z_struct + Z_reflected_new
    return F.normalize(Z_new, dim=-1), Z_reflected_new


def _routing_step_ah_wpos(
    Z: torch.Tensor,           # [B, N, D]
    Z_reflected: torch.Tensor, # [B, N, D]
    conn_hh: torch.Tensor,     # [N, K_hh]
    theta_pos: torch.Tensor,   # [1, N, 1]
    supp_w: torch.Tensor,      # [1, N, K_hh, 1]  pre-computed AH weights
    alpha_reflect: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single routing iteration with AH (wpos variant) baked in, alpha_turing=0."""
    Z_fwd = F.relu(Z - theta_pos)                           # [B, N, D]
    Z_nb  = Z_fwd[:, conn_hh, :]                            # [B, N, K_hh, D]
    Z_struct = (Z_nb * supp_w).sum(dim=2)                   # [B, N, D]
    Z_remainder = Z_fwd - Z
    Z_reflected_new = alpha_reflect * Z_reflected + Z_remainder
    Z_new = Z_struct + Z_reflected_new
    # clamp matches SGNNET_AntiHebbian original
    return F.normalize(Z_new.clamp(-10, 10), dim=-1), Z_reflected_new


# ---------------------------------------------------------------------------
# SGNNET_Resonant_CUDA
# ---------------------------------------------------------------------------

class SGNNET_Resonant_CUDA(nn.Module):
    """CUDA-optimized SGNNET_Resonant.

    Drop-in replacement for SGNNET_Resonant.  Optimised for the
    alpha_turing=0 efficiency config path (dynamic_z_geo or any mode).

    Wrapped with torch.compile(mode="reduce-overhead") by default so the
    JIT compiler can fuse the K_iter routing loop into CUDA kernels.

    Parameters mirror SGNNET_Resonant.__init__ exactly for drop-in use.
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        K_phase: int = 8,
        beam_size: int = 32,
        theta_init: float = 0.1,
        alpha_reflect: float = 0.3,
        alpha_turing: float = 0.0,   # efficiency config: always 0
        mode: str = "dynamic_z_geo",
        resonance_threshold: float = 0.0,
        geo_gamma: float = 1.0,
        routing_dropout_p: float = 0.0,
        rebuild_interval: int = 0,
        compile: bool = True,         # False → eager mode (debugging)
    ):
        super().__init__()

        if alpha_turing != 0.0:
            raise ValueError(
                "SGNNET_Resonant_CUDA only supports alpha_turing=0. "
                "Use SGNNET_Resonant for phase-inhibition paths."
            )

        self.base              = base
        self.K_phase           = K_phase
        self.beam_size         = beam_size
        self.alpha_reflect     = alpha_reflect
        self.alpha_turing      = 0.0   # hardcoded — dead code eliminated
        self.mode              = mode
        self.resonance_threshold = resonance_threshold
        self.geo_gamma         = geo_gamma
        self.routing_dropout_p = routing_dropout_p
        self._rebuild_interval = rebuild_interval
        self._rebuild_step     = 0
        self.W_phase           = None  # not used with alpha_turing=0

        N = base.N_hidden
        self.theta = nn.Parameter(torch.full((N,), theta_init))

        # Ensure conn_hh is stored contiguous — critical for gather performance
        base.conn_hh = base.conn_hh.contiguous()

        # Compile the inner routing step
        self._compiled = compile
        if compile and torch.cuda.is_available():
            self._routing_fn = torch.compile(
                _routing_step_no_turing, mode="reduce-overhead"
            )
        else:
            self._routing_fn = _routing_step_no_turing

    # ------------------------------------------------------------------
    # Graph management (tick_epoch / tick_step — keep API compatible)
    # ------------------------------------------------------------------

    def tick_epoch(self):
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    def tick_step(self):
        pass  # no phase graph to rebuild

    # ------------------------------------------------------------------
    # Compatibility shim
    # ------------------------------------------------------------------

    @property
    def W_pos(self):
        return self.base.W_pos

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.base._seed(x)   # [B, N, D]

        # Contiguous before first gather (subsequent steps stay contiguous)
        if not Z.is_contiguous():
            Z = Z.contiguous()

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        conn_hh   = self.base.conn_hh                              # [N, K_hh], contiguous

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z, Z_reflected = self._routing_fn(
                Z, Z_reflected, conn_hh, theta_pos, self.alpha_reflect
            )

        return self.base._readout(Z)

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------

    def uncompile(self) -> "SGNNET_Resonant_CUDA":
        """Return to eager mode (useful for debugging / numerical comparison)."""
        self._routing_fn = _routing_step_no_turing
        self._compiled   = False
        return self

    @classmethod
    def from_resonant(
        cls,
        resonant: SGNNET_Resonant,
        compile: bool = True,
    ) -> "SGNNET_Resonant_CUDA":
        """Construct from an existing SGNNET_Resonant, copying learned weights."""
        if resonant.alpha_turing != 0.0:
            raise ValueError("alpha_turing must be 0 for CUDA model")

        obj = cls(
            base              = resonant.base,
            K_phase           = resonant.K_phase,
            beam_size         = resonant.beam_size,
            theta_init        = 0.1,  # overwritten below
            alpha_reflect     = resonant.alpha_reflect,
            alpha_turing      = 0.0,
            mode              = resonant.mode,
            resonance_threshold = resonant.resonance_threshold,
            geo_gamma         = resonant.geo_gamma,
            routing_dropout_p = resonant.routing_dropout_p,
            rebuild_interval  = resonant._rebuild_interval,
            compile           = compile,
        )
        with torch.no_grad():
            obj.theta.copy_(resonant.theta)
        return obj


# ---------------------------------------------------------------------------
# SGNNET_AntiHebbian_CUDA
# ---------------------------------------------------------------------------

class SGNNET_AntiHebbian_CUDA(nn.Module):
    """Anti-Hebbian lateral inhibition with alpha_turing=0, compiled.

    AH suppression weights are baked into the same compiled routing step,
    so the gather + suppress + sum fuses into a single kernel invocation.

    Only supports variant='wpos' (static suppression weights, pre-computed
    once at init — no dynamic cosine sim re-computation per forward).

    Parameters mirror SGNNET_AntiHebbian.__init__ exactly.
    """

    def __init__(
        self,
        base: SGNNET_Resonant,      # SGNNET_Resonant (or SGNNET_Resonant_CUDA)
        alpha_ahebb: float = 0.3,
        variant: str = "wpos",
        compile: bool = True,
    ):
        super().__init__()

        if variant != "wpos":
            raise ValueError(
                "SGNNET_AntiHebbian_CUDA only supports variant='wpos'. "
                "Use SGNNET_AntiHebbian for zact dynamic suppression."
            )

        # Accept either SGNNET_Resonant or SGNNET_Resonant_CUDA as base
        if isinstance(base, SGNNET_Resonant_CUDA):
            self.m = base
            self._resonant_cuda = True
        elif isinstance(base, SGNNET_Resonant):
            if base.alpha_turing != 0.0:
                raise ValueError("alpha_turing must be 0 for CUDA model")
            # Wrap it
            self.m = SGNNET_Resonant_CUDA.from_resonant(base, compile=False)
            self._resonant_cuda = True
        else:
            raise TypeError(f"base must be SGNNET_Resonant or SGNNET_Resonant_CUDA, got {type(base)}")

        self.alpha_ahebb = alpha_ahebb
        self.variant     = variant
        self._compiled   = compile

        # Pre-compute static AH suppression weights [1, N, K_hh, 1]
        self._supp_w: torch.Tensor | None = None  # lazy — computed on first forward

        # Ensure conn_hh is contiguous
        self.m.base.conn_hh = self.m.base.conn_hh.contiguous()

        # Compile routing step
        if compile and torch.cuda.is_available():
            self._routing_fn = torch.compile(
                _routing_step_ah_wpos, mode="reduce-overhead"
            )
        else:
            self._routing_fn = _routing_step_ah_wpos

    # ------------------------------------------------------------------

    def _get_supp_w(self, device: torch.device) -> torch.Tensor:
        """Compute (or return cached) AH suppression weights."""
        if self._supp_w is None or self._supp_w.device != device:
            conn_hh = self.m.base.conn_hh
            N_h     = self.m.base.N_hidden
            W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)    # [N_h, D]
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)  # [N_h, K_hh]
            supp_w  = (
                1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
            ).unsqueeze(0).unsqueeze(-1)                          # [1, N, K_hh, 1]
            self._supp_w = supp_w.to(device)
        return self._supp_w

    def _invalidate_supp_w(self):
        """Call this if W_pos changes (e.g., after an optimizer step)."""
        self._supp_w = None

    # ------------------------------------------------------------------
    # Compatibility shims
    # ------------------------------------------------------------------

    @property
    def W_pos(self):
        return self.m.W_pos

    @property
    def W_phase(self):
        return None

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()
        # W_pos may have changed — invalidate cached suppression weights
        self._invalidate_supp_w()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Retrieve pre-computed suppression weights (cached, recomputed if W_pos device changed)
        supp_w = self._get_supp_w(x.device)

        Z = self.m.base._seed(x)   # [B, N, D]
        if not Z.is_contiguous():
            Z = Z.contiguous()

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        conn_hh   = self.m.base.conn_hh

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z, Z_reflected = self._routing_fn(
                Z, Z_reflected, conn_hh, theta_pos, supp_w, self.m.alpha_reflect
            )

        return self.m.base._readout(Z)

    # ------------------------------------------------------------------

    def uncompile(self) -> "SGNNET_AntiHebbian_CUDA":
        """Return to eager mode."""
        self._routing_fn = _routing_step_ah_wpos
        self._compiled   = False
        return self
