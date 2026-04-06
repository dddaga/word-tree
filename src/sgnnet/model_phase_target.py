"""SGNNET_PhaseTarget: Attention-like routing with local phase plasticity.

Two weights per neuron (analogous to Q and K in attention):
  W_pos         [N, D]  — signal operator (Key/Value analog). Learned via gradient.
  phase_target  [N, D]  — query vector for routing selection. Updated ONLY by
                          local plasticity rule (no gradient, no optimizer).

Routing per iteration:
  score[i, j] = phase_target[i] · W_pos[j]          (for j in conn_hh neighbors)
              - β · phase_target[i] · phase_target[j]  (optional diversity penalty)
  weights = softmax(score / τ)
  Z_new   = Σ_j weights[i, j] · Z[j]
  Z       = normalize(Z_new.clamp(-10, 10))

Per-batch plasticity (tick_step, no gradients required):
  For each neuron i, rank its K neighbors by contribution:
    contribution[j] = mean_over_batch(normalize(Z[i]) · normalize(Z[j]))
    (High = j's activation direction aligns with i's → helpful)
  Top-K/2  → attract: phase_target[i] += lr_phase · mean(W_pos[j] for helpful j)
  Bottom-K/2 → repel:  phase_target[i] -= lr_phase · mean(W_pos[j] for suppressive j)
  Renormalize phase_target to unit sphere after update.

Optional AH suppression (ah_alpha > 0):
  Adds static anti-Hebbian suppression on W_pos similarity (same as SGNNET_AntiHebbian):
    ah_supp[i, j] = 1 - ah_alpha · cos(W_pos[i], W_pos[j]).clamp(0)
  Applied as a multiplicative weight on attention scores before softmax.

Design notes
------------
- phase_target is NOT in the optimizer (requires_grad=False). Updated by tick_step().
- W_pos IS in the optimizer (standard gradient-based learning).
- No theta gate — avoids the gate-death failure mode seen in wave-1 mechanisms.
- Diversity penalty (β > 0) discourages multiple neurons from querying the same direction,
  analogous to how AH discourages neurons from occupying the same position in W_pos space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld


class SGNNET_PhaseTarget(nn.Module):
    """Phase-target attention routing with local plasticity.

    Parameters
    ----------
    base        : SGNNET_SmallWorld — provides W_pos, conn_hh, _seed, _readout
    tau         : float — softmax temperature for attention scores (default 1.0)
    beta        : float — diversity penalty weight (0 = disabled)
    lr_phase    : float — learning rate for phase_target plasticity update
    plasticity  : bool — whether to apply per-batch phase plasticity
    ah_alpha    : float — anti-Hebbian suppression strength (0 = disabled)
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        tau: float = 1.0,
        beta: float = 0.0,
        lr_phase: float = 0.01,
        plasticity: bool = True,
        ah_alpha: float = 0.0,
    ):
        super().__init__()
        self.base       = base
        self.tau        = tau
        self.beta       = beta
        self.lr_phase   = lr_phase
        self.plasticity = plasticity
        self.ah_alpha   = ah_alpha

        N = base.N_hidden
        D = base.D

        # Query vectors — unit sphere, NOT in optimizer
        pt = F.normalize(torch.randn(N, D), dim=-1)
        self.register_buffer("phase_target", pt)  # [N, D]

        # AH suppression weights (static per forward, recomputed in tick_epoch)
        # None until first forward call if ah_alpha > 0
        self._ah_supp: torch.Tensor | None = None

        # Buffer for last-batch Z activations (set in forward, used in tick_step)
        self._last_Z: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Trainer-compatibility properties
    # ------------------------------------------------------------------

    @property
    def W_pos(self) -> nn.Parameter:
        return self.base.W_pos

    @property
    def W_phase(self):
        return None

    # ------------------------------------------------------------------
    # AH suppression (precomputed from W_pos)
    # ------------------------------------------------------------------

    def _build_ah_supp(self) -> torch.Tensor | None:
        """Precompute static AH suppression weights [N, K] from current W_pos."""
        if self.ah_alpha <= 0.0:
            return None
        N = self.base.N_hidden
        conn = self.base.conn_hh              # [N, K]
        W_n  = F.normalize(self.base.W_pos[:N].detach(), dim=-1)  # [N, D]
        cos_sim = (W_n.unsqueeze(1) * W_n[conn]).sum(-1)          # [N, K]
        return (1.0 - self.ah_alpha * cos_sim.clamp(min=0))       # [N, K]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.base._seed(x)               # [B, N, D]
        Z = self._route(Z)                   # [B, N, D]
        self._last_Z = Z.detach()            # store for tick_step (no gradient)
        return self.base._readout(Z)         # [B, N_out]

    def _route(self, Z: torch.Tensor) -> torch.Tensor:
        N    = self.base.N_hidden
        conn = self.base.conn_hh             # [N, K]
        W_h  = self.base.W_pos[:N]           # [N, D]

        # Precompute static parts (not inside loop — W_pos doesn't change per iter)
        W_nb = W_h[conn]                     # [N, K, D]

        # Attention base scores: phase_target[i] · W_pos[neighbor_j]
        score_base = torch.einsum("nd,nkd->nk", self.phase_target, W_nb)  # [N, K]

        # Diversity penalty: - β · phase_target[i] · phase_target[neighbor_j]
        if self.beta > 0.0:
            pt_nb     = self.phase_target[conn]  # [N, K, D]
            diversity = torch.einsum("nd,nkd->nk", self.phase_target, pt_nb)  # [N, K]
            score_base = score_base - self.beta * diversity

        # AH suppression (multiplicative on scores before softmax)
        if self.ah_alpha > 0.0:
            if self._ah_supp is None:
                self._ah_supp = self._build_ah_supp()
            score_base = score_base * self._ah_supp  # [N, K]

        # Attention weights [N, K]
        weights = F.softmax(score_base / self.tau, dim=-1)          # [N, K]
        weights = weights.unsqueeze(0).unsqueeze(-1)                 # [1, N, K, 1]

        for _ in range(self.base.K_iter):
            Z_nb = Z[:, conn, :]                    # [B, N, K, D]
            Z    = (weights * Z_nb).sum(dim=2)      # [B, N, D]
            Z    = F.normalize(Z.clamp(-10, 10), dim=-1)

        return Z

    # ------------------------------------------------------------------
    # Per-batch plasticity (called by Trainer via tick_step hook)
    # ------------------------------------------------------------------

    def tick_step(self) -> None:
        """Update phase_target using last batch's activations. No gradients."""
        if not self.plasticity or self._last_Z is None:
            return

        Z    = self._last_Z                  # [B, N, D], detached
        conn = self.base.conn_hh             # [N, K]
        N    = self.base.N_hidden
        K    = conn.shape[1]
        K2   = K // 2

        # Normalize Z for direction-only comparison
        Z_unit = F.normalize(Z, dim=-1)      # [B, N, D]

        # Contribution: mean_batch(Z[i] · Z[neighbor_j])
        Z_nb = Z_unit[:, conn, :]            # [B, N, K, D]
        Z_i  = Z_unit.unsqueeze(2)           # [B, N, 1, D]
        contrib = (Z_i * Z_nb).sum(-1).mean(0)  # [N, K]

        # Rank neighbors ascending (low contrib = suppressive, high = helpful)
        sorted_idx       = contrib.argsort(dim=-1)        # [N, K]
        suppressive_idx  = sorted_idx[:, :K2]             # [N, K//2]
        helpful_idx      = sorted_idx[:, K2:]             # [N, K//2]

        # Gather W_pos for each group
        W_h = self.base.W_pos[:N].detach()                # [N, D]

        helpful_nb      = conn.gather(1, helpful_idx)     # [N, K//2]
        suppressive_nb  = conn.gather(1, suppressive_idx) # [N, K//2]

        attract = W_h[helpful_nb].mean(1)                 # [N, D]
        repel   = W_h[suppressive_nb].mean(1)             # [N, D]

        # Apply update and renormalize
        self.phase_target.add_(self.lr_phase * (attract - repel))
        self.phase_target.copy_(F.normalize(self.phase_target, dim=-1))

        self._last_Z = None  # clear buffer

    # ------------------------------------------------------------------
    # Epoch hook (Trainer calls if hasattr)
    # ------------------------------------------------------------------

    def tick_epoch(self) -> None:
        """Invalidate AH suppression cache when W_pos updates between epochs."""
        self._ah_supp = None
