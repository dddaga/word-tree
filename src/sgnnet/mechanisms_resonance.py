"""Resonance excitatory mechanism wrapper for SGNNET_Resonant.

SGNNET_ResonanceExcitatory: phase-excitatory signal with optional activation gate.
Wraps SGNNET_Resonant and replaces the routing loop with one that adds an excitatory
phase channel on top of the structural routing.

Design
------
- Phase excitatory: neurons in the W_phase K-NN graph (conn_phase) send excitatory
  (positive) signals, not inhibitory ones.
- Activation gate: if use_activation_gate=True, a source neuron's excitatory
  contribution is scaled by how well its current activation aligns with its own
  phase anchor (dot(Z[h], W_phase[h]).clamp(0)). This is a resonance gate —
  only neurons "in tune" with their phase anchor broadcast.
- rebuild_per_batch: if True, rebuild conn_phase from W_phase on every forward
  pass so the topology stays fresh as W_phase evolves. If False, only rebuild at
  epoch boundaries (tick_epoch).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_resonant import SGNNET_Resonant


class SGNNET_ResonanceExcitatory(nn.Module):
    """Phase-excitatory routing with optional activation gate.

    Parameters
    ----------
    base_model        : SGNNET_Resonant (provides theta, W_phase, conn_phase,
                        base.conn_hh, _phase_inhibit, alpha_turing)
    alpha_exc         : weight on the excitatory phase signal
    rebuild_per_batch : if True, rebuild conn_phase from W_phase each forward;
                        if False, rely on tick_epoch to keep it fresh
    use_activation_gate : if True, scale each source's excitatory contribution
                        by dot(Z[h], W_phase[h]).clamp(0) — resonance gate
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        alpha_exc: float = 0.3,
        rebuild_per_batch: bool = True,
        use_activation_gate: bool = True,
    ):
        super().__init__()
        self.m                  = base_model
        self.alpha_exc          = alpha_exc
        self.rebuild_per_batch  = rebuild_per_batch
        self.use_activation_gate = use_activation_gate

    # ------------------------------------------------------------------
    # Property delegates
    # ------------------------------------------------------------------

    @property
    def W_pos(self):
        return self.m.W_pos

    @property
    def W_phase(self):
        return self.m.W_phase

    # ------------------------------------------------------------------
    # Epoch / step hooks
    # ------------------------------------------------------------------

    def tick_epoch(self):
        if not self.rebuild_per_batch:
            # Only rebuild at epoch boundary when not doing per-batch rebuilds
            self.m.tick_epoch()
        else:
            # Always delegate the base tick (W_pos clamping etc.) but skip
            # graph rebuild — we already rebuild per batch
            if hasattr(self.m.base, "tick_epoch"):
                self.m.base.tick_epoch()

    def _rebuild_conn_phase(self):
        """Rebuild W_phase K-NN graph (same logic as SGNNET_Resonant._build_phase_graph)."""
        self.m._build_phase_graph()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)                                     # [B, N, D]

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)    # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)              # [N, D]

        if self.rebuild_per_batch:
            self._rebuild_conn_phase()

        for _ in range(self.m.base.K_iter):
            # ── 1. Excitatory gate ─────────────────────────────────────
            Z_fwd = F.relu(Z - theta_pos)                            # [B, N, D]

            # ── 2. Structural routing (conn_hh) ───────────────────────
            Z_struct = Z_fwd[:, self.m.base.conn_hh, :].sum(2)      # [B, N, D]

            # ── 3. Phase excitatory signal (conn_phase) ───────────────
            # conn_phase is [N, K_phase]; Z_fwd[:, conn_phase, :] is [B, N, K_phase, D]
            Z_phase_nb = Z_fwd[:, self.m.conn_phase, :]              # [B, N, K_phase, D]

            if self.use_activation_gate:
                # Gate: how well does each source's current activation align
                # with its own phase anchor?
                # dot(Z[b, h], W_phase[h]) for each source h = conn_phase[n, k]
                res_gate = (F.normalize(Z, dim=-1) * W_ph_norm.unsqueeze(0)).sum(-1)
                # [B, N] — resonance gate per neuron

                res_gate = res_gate.clamp(min=0)                     # [B, N]

                # Gather gate values for each source in conn_phase
                # conn_phase [N, K_phase] → src_gate [B, N, K_phase]
                src_gate = res_gate[:, self.m.conn_phase]            # [B, N, K_phase]

                # Scale each source vector by its gate value
                Z_exc = (Z_phase_nb * src_gate.unsqueeze(-1)).sum(2) # [B, N, D]
            else:
                Z_exc = Z_phase_nb.sum(2)                            # [B, N, D]

            # ── 4. Long-range phase inhibition ────────────────────────
            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)  # [B, N, D]

            # ── 5. Combine & normalise ─────────────────────────────────
            Z_new = Z_struct + self.alpha_exc * Z_exc + self.m.alpha_turing * Z_inh
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)
