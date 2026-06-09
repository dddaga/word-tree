"""SGNNET_DWPhaseGate: ΔW-proj × per-neighbor PhaseGate compound (step987).

Modes: mul (A), add (B), gate_only (C).
Scaffolding matches Ref: leaky_relu Z-fwd, reflection, normalize+clamp.
alpha resolved once per forward (not per K_iter).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld


class SGNNET_DWPhaseGate(nn.Module):
    """ΔW-proj × per-neighbor PhaseGate compound (step987).

    compound_mode: "mul" | "add" | "gate_only"
    alpha_mode:    "random" (Uniform(0,1) once per forward) | "fixed" (0.5)
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        alpha_reflect: float = 0.5,
        compound_mode: str = "mul",
        alpha_mode: str = "random",
        theta_init: float = 0.1,
    ):
        assert compound_mode in ("mul", "add", "gate_only"), (
            f"compound_mode must be mul|add|gate_only, got {compound_mode!r}"
        )
        assert alpha_mode in ("random", "fixed"), (
            f"alpha_mode must be random|fixed, got {alpha_mode!r}"
        )
        super().__init__()
        self.base          = base
        self.alpha_reflect = alpha_reflect
        self.compound_mode = compound_mode
        self.alpha_mode    = alpha_mode
        self.W_phase       = None   # Trainer compatibility shim

        N = base.N_hidden
        D = base.W_pos.shape[1]

        # Per-neuron threshold (matches Ref)
        self.theta = nn.Parameter(torch.full((N,), theta_init))

        # Per-node phase direction [N, D], L2-normalised at forward time
        self.w_n = nn.Parameter(torch.rand(N, D))

        # Config B: additive blend scale (avoids shadowing Python 'lambda')
        if compound_mode == "add":
            self.lambda_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.lambda_gate = None

        # Pre-compute ΔW-proj (static, matches step199 / Ref)
        with torch.no_grad():
            dw = self._compute_dw(base.W_pos, base.conn_hh, N)
        self.register_buffer("dw", dw)  # [1, N, K_hh, D]

    @staticmethod
    def _compute_dw(W_pos, conn_hh, N):
        """Normalised ΔW direction vectors → [1, N, K_hh, D]."""
        W_h   = W_pos[:N]
        delta = W_h.unsqueeze(1) - W_h[conn_hh]       # [N, K_hh, D]
        return F.normalize(delta, dim=-1).unsqueeze(0)  # [1, N, K_hh, D]

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
    # Routing weight helpers
    # ------------------------------------------------------------------

    def _dw_proj_weights(self, Z_fwd):
        """ΔW-proj raw weights → [B, N, K_hh] (before abs/norm)."""
        Z_nb = Z_fwd[:, self.base.conn_hh, :]    # [B, N, K_hh, D]
        return (Z_nb * self.dw).sum(-1)           # [B, N, K_hh]

    def _phase_gate_weights(self, Z_fwd, alpha):
        """Per-neighbor PhaseGate → [B, N, K_hh] ≥0, sum-normalised.

        gate always ≥0, so in mul mode: (r*gate).abs() == |r|*gate.
        """
        w_n_nb = F.normalize(self.w_n, dim=-1)[self.base.conn_hh]  # [N, K_hh, D]
        s      = (Z_fwd.unsqueeze(2) * w_n_nb.unsqueeze(0)).sum(-1) # [B, N, K_hh]
        gate   = F.relu(s) + alpha * F.relu(-s)
        return gate / (gate.sum(-1, keepdim=True) + 1e-8)

    def _routing_weights(self, Z_fwd, alpha, debug=False):
        """Final routing weights → [B, N, K_hh]."""
        gate = self._phase_gate_weights(Z_fwd, alpha)              # [B, N, K_hh] ≥0
        if self.compound_mode == "gate_only":
            weight = gate
        else:
            r = self._dw_proj_weights(Z_fwd)
            if self.compound_mode == "mul":
                weight = (r * gate).abs()  # gate≥0 → == |r|*gate
            else:  # add
                weight = r.abs() + self.lambda_gate.clamp(min=0.0) * gate
        weight = weight / (weight.sum(-1, keepdim=True) + 1e-8)
        if debug:
            tied = (gate - gate.mean(-1, keepdim=True)).abs().lt(1e-4).float().mean().item()
            print(f"  [DWPhaseGate debug] mode={self.compound_mode}  "
                  f"gate.mean={gate.mean():.4f}  tied_frac={tied:.4f}  alpha={alpha:.4f}")
        return weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        Z         = self.base._seed(x)                            # [B, N, D]
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        conn_hh   = self.base.conn_hh

        # Resolve alpha ONCE per forward (not per K_iter)
        if self.alpha_mode == "random" and self.training:
            alpha = torch.rand(1).item()
        else:
            alpha = 0.5

        Z_ref   = torch.zeros_like(Z)
        _dbg    = debug

        for _ in range(self.base.K_iter):
            # Leaky-relu Z-forward with per-neuron threshold (matches Ref)
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)

            # Routing weights [B, N, K_hh]
            weight = self._routing_weights(Z_fwd, alpha, debug=_dbg)
            _dbg   = False   # print only on first iter

            # Weighted aggregation
            Z_nb  = Z_fwd[:, conn_hh, :]                         # [B, N, K_hh, D]
            Z_agg = (weight.unsqueeze(-1) * Z_nb).sum(-2)        # [B, N, D]

            # Self-inhibition reflection (matches Ref)
            Z_ref = self.alpha_reflect * Z_ref + (Z_fwd - Z)

            # Combine and L2-normalise (matches Ref)
            Z = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.base._readout(Z)
