"""SGNNET_PhaseGate: per-node phase-gated routing (step985 v2 — corrected).

Mechanism (K_hh=2 required):
  1. Phase scalar:  s_i = dot(Z[i], w_n_norm[i])          [B, N]
  2. Asymmetric gate:
       gate_pos = relu(s_i)                                [B, N]
       gate_neg = alpha * relu(-s_i)                       [B, N]
     alpha source depends on alpha_mode (see below).
  3. Normalise: gate = [gate_pos, gate_neg] / (sum + 1e-8) [B, N, 2]
     NOT softmax — zero side stays zero, giving true directional selectivity.
  4. Aggregation: Z_agg[i] = gate[:,i,0]*Z[nb0] + gate[:,i,1]*Z[nb1]
  5. L2-normalise output.

alpha_mode options:
  "random"  — alpha ~ Uniform(0,1) per forward call during train; 0.5 at eval
  "fixed"   — alpha = 1.0 (symmetric magnitudes, no stochasticity)
  "learned" — alpha is a single global nn.Parameter, init=0.5 (scalar)

v1 bug: used gate=[s,-s]+softmax → always 50/50 regardless of s (symmetric).
v2 fix: relu split + sum-divide normalization → true directional selectivity.

Compounding note: PhaseGate REPLACES ΔW-proj on the same signal path.
Compare vs Ref (ΔW-proj), not stacked on top.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld


class SGNNET_PhaseGate(nn.Module):
    """Per-node phase-gated routing replacing ΔW-proj aggregation.

    Parameters
    ----------
    base          : SGNNET_SmallWorld backbone (K_hh=2 required)
    alpha_reflect : self-inhibition factor (default 0.5, matches step199)
    alpha_mode    : "random" | "fixed" | "learned"
    theta_init    : initial per-neuron threshold
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        alpha_reflect: float = 0.5,
        alpha_mode: str = "random",
        theta_init: float = 0.1,
    ):
        super().__init__()
        assert alpha_mode in ("random", "fixed", "learned"), (
            f"alpha_mode must be random|fixed|learned, got {alpha_mode!r}"
        )

        K_hh = base.conn_hh.shape[1]
        assert K_hh == 2, f"PhaseGate requires K_hh=2, got {K_hh}"

        self.base          = base
        self.alpha_reflect = alpha_reflect
        self.alpha_mode    = alpha_mode
        self.W_phase       = None  # Trainer compatibility shim

        N = base.N_hidden
        D = base.W_pos.shape[1]

        # Per-neuron threshold
        self.theta = nn.Parameter(torch.full((N,), theta_init))

        # Per-node phase direction [N, D], uniform init
        self.w_n = nn.Parameter(torch.rand(N, D))

        # Learned alpha: single global scalar, init=0.5
        if alpha_mode == "learned":
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.alpha = None

    # ------------------------------------------------------------------
    # Compatibility shims (Trainer calls these)
    # ------------------------------------------------------------------

    @property
    def W_pos(self):
        return self.base.W_pos

    def tick_epoch(self):
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    # ------------------------------------------------------------------
    # PhaseGate routing step
    # ------------------------------------------------------------------

    def _phase_gate_agg(
        self, Z: torch.Tensor, debug: bool = False
    ) -> torch.Tensor:
        """Asymmetric phase-gated weighted aggregation.

        Args:
            Z     : [B, N, D]  L2-normalised activations
            debug : if True, print gate stats for verification

        Returns:
            Z_agg : [B, N, D]  weighted aggregate (un-normalised)
        """
        conn_hh = self.base.conn_hh                              # [N, 2]

        # L2-normalise w_n at forward time
        w_n_norm = F.normalize(self.w_n, dim=-1)                 # [N, D]

        # Phase scalar: dot(Z[b,i], w_n_norm[i]) → [B, N]
        s = (Z * w_n_norm.unsqueeze(0)).sum(dim=-1)              # [B, N]

        # Asymmetric gate split
        gate_pos = F.relu(s)                                     # [B, N]
        gate_neg_raw = F.relu(-s)                                # [B, N]

        # Resolve alpha
        if self.alpha_mode == "random":
            if self.training:
                alpha = torch.rand(1, device=Z.device).item()
            else:
                alpha = 0.5
        elif self.alpha_mode == "fixed":
            alpha = 1.0
        else:  # learned
            alpha = self.alpha

        gate_neg = alpha * gate_neg_raw                          # [B, N]

        # Stack and normalise (sum-divide, NOT softmax)
        gate = torch.stack([gate_pos, gate_neg], dim=-1)         # [B, N, 2]
        gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-8)   # [B, N, 2]

        if debug:
            g0 = gate[..., 0].mean().item()
            g1 = gate[..., 1].mean().item()
            tied = (gate[..., 0] - gate[..., 1]).abs().lt(1e-4).float().mean().item()
            frac_pos = (s > 0).float().mean().item()
            print(
                f"  [PhaseGate gate debug]  "
                f"gate[0].mean={g0:.4f}  gate[1].mean={g1:.4f}  "
                f"tied_frac={tied:.4f}  s>0_frac={frac_pos:.4f}  "
                f"alpha={alpha if isinstance(alpha, float) else alpha.item():.4f}"
            )

        Z_nb  = Z[:, conn_hh, :]                                 # [B, N, 2, D]
        Z_agg = (gate.unsqueeze(-1) * Z_nb).sum(dim=2)          # [B, N, D]
        return Z_agg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        Z = self.base._seed(x)                                   # [B, N, D]

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        Z_ref     = torch.zeros_like(Z)
        _debug_printed = False

        for _ in range(self.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)

            # Phase-gated aggregation (replaces ΔW-proj)
            _print_debug = debug and not _debug_printed
            Z_agg = self._phase_gate_agg(Z_fwd, debug=_print_debug)
            _debug_printed = _debug_printed or _print_debug

            # Self-inhibition reflection
            Z_ref = self.alpha_reflect * Z_ref + (Z_fwd - Z)

            # Combine and L2-normalise
            Z = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.base._readout(Z)
