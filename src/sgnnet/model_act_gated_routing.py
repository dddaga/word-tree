"""SGNNET_ActGatedRouting: input-dependent excitatory routing via activation soft-attention.

Architecture
------------
Replaces the static sum(Z[conn_hh]) with input-specific attention:

  candidates: conn_expanded [N, K'] built from conn_hh and/or conn_phase
  score[b,i,j] = dot(Z[b,i], Z[b,candidates[i,j]]) / sqrt(D)
  weights      = softmax(score, dim=-1)               [B, N, K']   (soft mode)
               OR top-K-of-K' hard selection                        (hard mode)
  Z_struct[i]  = sum_j( weights[i,j] * Z_fwd[candidates[i,j]] )

The input decides which candidates to weight each forward pass — same weights
serve all inputs, different subgraphs are activated per input.

candidate_mode controls the candidate pool:
  "mixed"   : conn_hh (spatial small-world) + conn_phase (W_phase K-NN) — K'=K_hh+K_phase
  "spatial" : conn_hh only                                                — K'=K_hh
  "phase"   : conn_phase only (pure phase-resonant routing)               — K'=K_phase

routing_mode:
  "soft" : softmax over K' candidates (differentiable, bounded excitation — weights sum to 1)
  "hard" : hard top-K_select of K' (sparse, non-differentiable selection)

ah_alpha > 0: W_pos-based Anti-Hebbian suppression — biases attention AWAY from
  spatially similar neighbors, same geometric suppression as SGNNET_AntiHebbian
  but applied as a pre-softmax score penalty rather than a post-sum subtract.

hop_decay < 1.0: signal attenuates per routing step.
  score is multiplied by hop_decay^k at iteration k (k=0,1,...,K_iter-1).
  Models wave attenuation: signals that have propagated more hops carry less certainty.

Long-range phase inhibition from SGNNET_Resonant._phase_inhibit is retained unchanged
— only the excitatory routing is replaced.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_resonant import SGNNET_Resonant


class SGNNET_ActGatedRouting(nn.Module):
    """Input-dependent routing: activation soft-attention over expanded candidate set.

    Parameters
    ----------
    base           : SGNNET_Resonant (provides _seed, _readout, conn_hh, conn_phase,
                     theta, W_phase, _phase_inhibit)
    routing_mode   : "soft" (softmax attention) | "hard" (top-K selection)
    candidate_mode : "mixed" | "spatial" | "phase"  — defines the candidate pool
    K_select       : for hard mode: number of candidates to select per neuron per step
    ah_alpha       : Anti-Hebbian W_pos suppression strength (0 = off)
    hop_decay      : per-step score multiplier (1.0 = no decay, 0.9 = 10% decay/hop)
    temp           : softmax temperature (1.0 = standard; lower = sharper selection)
    """

    def __init__(
        self,
        base: SGNNET_Resonant,
        routing_mode: str = "soft",
        candidate_mode: str = "mixed",
        K_select: int = 6,
        ah_alpha: float = 0.0,
        hop_decay: float = 1.0,
        temp: float = 1.0,
    ):
        super().__init__()
        self.base           = base
        self.routing_mode   = routing_mode
        self.candidate_mode = candidate_mode
        self.K_select       = K_select
        self.ah_alpha       = ah_alpha
        self.hop_decay      = hop_decay
        self.temp           = temp

        self._build_conn_expanded()

    # ------------------------------------------------------------------
    # Candidate graph
    # ------------------------------------------------------------------

    def _build_conn_expanded(self):
        """Build conn_expanded from conn_hh and/or conn_phase."""
        conn_hh    = self.base.base.conn_hh    # [N, K_hh]
        conn_phase = self.base.conn_phase       # [N, K_phase]

        if self.candidate_mode == "spatial":
            conn_exp = conn_hh
        elif self.candidate_mode == "phase":
            conn_exp = conn_phase
        else:  # "mixed"
            conn_exp = torch.cat([conn_hh, conn_phase], dim=1)  # [N, K_hh+K_phase]

        self.register_buffer("conn_expanded", conn_exp)

    def tick_epoch(self):
        """Rebuild conn_phase (W_phase K-NN) and refresh conn_expanded."""
        self.base.tick_epoch()
        self._build_conn_expanded()

    # ------------------------------------------------------------------
    # Compatibility shims
    # ------------------------------------------------------------------

    @property
    def W_pos(self):
        return self.base.W_pos

    @property
    def W_phase(self):
        return self.base.W_phase

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.base.base._seed(x)             # [B, N, D]
        B, N, D = Z.shape

        theta_pos  = self.base.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        W_ph_norm  = F.normalize(self.base.W_phase, dim=-1)            # [N, D]
        conn_exp   = self.conn_expanded                                  # [N, K']
        K_prime    = conn_exp.shape[1]
        scale      = math.sqrt(D)

        # Precompute static AH suppression weights over conn_expanded (if ah_alpha > 0)
        if self.ah_alpha > 0:
            N_h    = self.base.base.N_hidden
            W_n    = F.normalize(self.base.W_pos[:N_h], dim=-1)        # [N, D]
            W_n_nb = W_n[conn_exp]                                       # [N, K', D]
            pos_sim = (W_n.unsqueeze(1) * W_n_nb).sum(-1)              # [N, K']
            ah_penalty = self.ah_alpha * pos_sim.clamp(min=0)          # [N, K']
            # pre-softmax score penalty: subtract to discourage similar-W_pos neighbors
            ah_penalty = ah_penalty.unsqueeze(0)                        # [1, N, K']

        for step in range(self.base.base.K_iter):

            # ── Threshold gate ─────────────────────────────────────────
            Z_fwd = F.relu(Z - theta_pos)                               # [B, N, D]

            # ── Gather K' candidate activations ───────────────────────
            Z_cand = Z_fwd[:, conn_exp, :]                              # [B, N, K', D]

            # ── Activation resonance score ─────────────────────────────
            # dot(Z[i], Z_fwd[candidate]) / sqrt(D)
            score = (Z.unsqueeze(2) * Z_cand).sum(-1) / scale          # [B, N, K']

            # ── Hop-count attenuation ──────────────────────────────────
            if self.hop_decay < 1.0:
                score = score * (self.hop_decay ** step)

            # ── AH suppression: penalise spatially similar neighbors ───
            if self.ah_alpha > 0:
                score = score - ah_penalty                               # [B, N, K']

            # ── Selection ─────────────────────────────────────────────
            if self.routing_mode == "soft":
                weights = F.softmax(score / self.temp, dim=-1)          # [B, N, K']
                Z_struct = (weights.unsqueeze(-1) * Z_cand).sum(2)      # [B, N, D]

            else:  # "hard"
                K_sel  = min(self.K_select, K_prime)
                topk   = score.topk(K_sel, dim=-1).indices              # [B, N, K_sel]
                # Uniform average over selected K_sel candidates
                Z_cand_sel = torch.gather(
                    Z_cand, 2,
                    topk.unsqueeze(-1).expand(-1, -1, -1, D)
                )                                                        # [B, N, K_sel, D]
                Z_struct = Z_cand_sel.sum(2) / K_sel                    # [B, N, D]

            # ── Long-range phase inhibition (from Resonant, unchanged) ─
            Z_inh = self.base._phase_inhibit(Z, W_ph_norm, theta_pos)  # [B, N, D]

            # ── Combine and normalise ──────────────────────────────────
            Z_new = Z_struct + self.base.alpha_turing * Z_inh
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base.base._readout(Z)
