"""SGNNET_DistPhase: Distance-based phase routing via exp(-γ·d) edge weights.

Motivation
----------
Every failed wave-1 mechanism (steps 58-63) introduced complex learned phase dynamics
(W_phase, resonance gating, attention over candidates) and collapsed due to gate death
or additive excitation saturation. This model strips phase back to first principles:

    Distance determines signal attenuation.

Neurons closer in W_pos space send stronger signals. No W_phase, no learned phase
coefficients, no K-NN phase graph — pure geometry. This is step 1 in a progressive
phase-mechanism build-up: geometry → dynamic → stabilize.

Architecture
------------
For each (i, j) edge in conn_hh (the small-world hidden→hidden graph):

    dist[i,j]      = ||W_pos[i] - W_pos[j]||₂
    dist_norm[i,j] = dist[i,j] / mean(dist)        ← scale-independent: γ is meaningful
    raw_w[i,j]     = exp(-gamma * dist_norm[i,j])   ← Gaussian falloff

    if weighting == 'softmax':
        w[i,j] = raw_w[i,j] / Σ_k raw_w[i,k]      ← sum-to-1 per neuron (redistributes)
    else:  # 'raw'
        w[i,j] = raw_w[i,j]                         ← unnormalized (boosts near, dims far)

For each routing iteration:
    Z_fwd   = relu(Z - theta_pos)                   ← threshold gate (per-neuron)
    Z_nb    = Z_fwd[:, conn_hh, :]                  ← [B, N, K_hh, D] neighbours
    Z_struct = Σ_j w[i,j] * Z_nb[:, i, j, :]       ← [B, N, D] weighted sum
    Z       = normalize(Z_struct.clamp(-10, 10), dim=-1)

Weighting modes
---------------
- 'softmax': weights are row-normalized (sum-to-1 per neuron). Total excitation magnitude
  is conserved — same as uniform routing but biased toward nearby neurons. More numerically
  stable; prevents unbounded amplification.
- 'raw': weights are raw exp(-γ·d_norm) values. Near neighbours contribute >1 weight;
  far neighbours contribute <1. Total excitation magnitude scales with the number of
  neighbours and their distances. May amplify or attenuate depending on graph structure.

γ controls falloff relative to mean inter-neuron distance (scale-independent):
  γ=0.5 → soft falloff; all edges within ~2σ contribute meaningfully
  γ=1.0 → moderate falloff; edges at mean distance get weight exp(-1) ≈ 0.37
  γ=2.0 → sharp falloff; only the nearest edges matter; effectively a hard-KNN

Distance weights are precomputed in __init__ and refreshed in tick_epoch() as W_pos
evolves during training (weights change as neurons reposition).

Expose W_pos property delegating to self.base.W_pos so Trainer safety valve works.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld


class SGNNET_DistPhase(nn.Module):
    """Distance-based phase routing wrapper around SGNNET_SmallWorld.

    Parameters
    ----------
    base       : SGNNET_SmallWorld instance (provides W_pos, conn_hh, _seed, _readout)
    gamma      : float — decay rate for exp(-γ·d_norm). Fixed hyperparameter, not learned.
    weighting  : 'softmax' | 'raw' — how to normalize edge weights per neuron
    theta_init : float — initial per-neuron threshold value (default 0.1)
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        gamma: float = 1.0,
        weighting: str = "softmax",
        theta_init: float = 0.1,
    ):
        super().__init__()
        self.base = base
        self.gamma = gamma
        self.weighting = weighting

        N = base.N_hidden

        # Per-neuron threshold gate — shape [N_hidden]
        self.theta = nn.Parameter(torch.full((N,), theta_init))

        # Precompute distance weights from initial W_pos
        # Registered as buffer so it lives on the correct device;
        # recomputed by tick_epoch() as W_pos evolves.
        w = self._compute_dist_weights()
        self.register_buffer("dist_weights", w)  # [N_hidden, K_hh]

    # ------------------------------------------------------------------
    # W_pos property — Trainer safety valve reads model.W_pos directly
    # ------------------------------------------------------------------

    @property
    def W_pos(self) -> nn.Parameter:
        return self.base.W_pos

    @property
    def W_phase(self):
        return None

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_dist_weights(self) -> torch.Tensor:
        """Compute exp(-γ·d_norm) edge weights from current W_pos.

        Returns
        -------
        w : [N_hidden, K_hh] float32 tensor
        """
        W_pos = self.base.W_pos.detach()          # [N+N_out, D]
        conn  = self.base.conn_hh                 # [N, K_hh]
        N     = self.base.N_hidden

        W_hidden = W_pos[:N]                      # [N, D]
        W_nb     = W_hidden[conn]                 # [N, K_hh, D]

        # Euclidean distance per edge: [N, K_hh]
        diff = W_hidden.unsqueeze(1) - W_nb       # [N, K_hh, D]
        dist = diff.norm(dim=-1)                  # [N, K_hh]

        # Scale-independent normalization
        mean_dist = dist.mean().clamp(min=1e-8)
        dist_norm = dist / mean_dist              # [N, K_hh]

        raw_w = torch.exp(-self.gamma * dist_norm)  # [N, K_hh]

        if self.weighting == "softmax":
            w = raw_w / raw_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            w = raw_w

        return w.float()

    def tick_epoch(self):
        """Refresh dist_weights after each epoch as W_pos evolves.

        Call from the training loop once per epoch (after optimizer step).
        """
        with torch.no_grad():
            new_w = self._compute_dist_weights()
            self.dist_weights.copy_(new_w)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: seed → distance-weighted routing → readout."""
        Z = self.base._seed(x)           # [B, N, D]
        Z = self._route(Z)
        return self.base._readout(Z)

    def _route(self, Z: torch.Tensor) -> torch.Tensor:
        """K_iter rounds of distance-weighted hidden→hidden routing.

        Each round:
          1. Threshold gate: Z_fwd = relu(Z - θ)   [B, N, D]
          2. Gather neighbours: Z_nb [B, N, K_hh, D]
          3. Weighted sum: Z_struct = Σ_j w[i,j] · Z_nb[b,i,j,:]
          4. Clamp + L2 normalize
        """
        conn = self.base.conn_hh          # [N, K_hh]
        w    = self.dist_weights          # [N, K_hh]
        K_iter = self.base.K_iter

        # theta broadcast: [1, N, 1]
        theta = self.theta.unsqueeze(0).unsqueeze(-1)

        for _ in range(K_iter):
            # Threshold gate
            Z_fwd = F.relu(Z - theta)                       # [B, N, D]

            # Gather neighbours: [B, N, K_hh, D]
            Z_nb = Z_fwd[:, conn, :]

            # Weighted sum: w [N, K_hh] → [1, N, K_hh, 1] broadcast
            w_bcast = w.unsqueeze(0).unsqueeze(-1)          # [1, N, K_hh, 1]
            Z_struct = (w_bcast * Z_nb).sum(dim=2)          # [B, N, D]

            # Clamp + normalize
            Z = F.normalize(Z_struct.clamp(-10.0, 10.0), dim=-1)

        return Z
