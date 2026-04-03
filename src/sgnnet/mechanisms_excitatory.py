"""Excitatory routing mechanism wrappers for SGNNET_Resonant.

Step 17 — Fast W_phase adaptation (W_phase as input-specific fast weight):
  SGNNET_FastPhase  — W_phase cloned per-forward as local 'A', never accumulated.
                      fast_rule controls how A is adapted and how it drives routing:
                        'oja_shared'     : Oja rule (batch-shared A, α=0.1 or 0.3)
                        'oja_per_sample' : Oja rule (per-sample A, α=0.1)
                        'hopfield'       : Z attracted toward stored A directions
                        'attention'      : A updated to Z-cluster centroids via attention

Step 18 — Signed coupling + STDP:
  SGNNET_SignedCoupling — unified excitatory/inhibitory via cosine(Z_h, Z_j) sign
                          sparse_k>0: top-k positive + top-k negative pairs only
  SGNNET_STDP_S2        — causal excitation: Z from step k excites similar neurons at k+1
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_resonant import SGNNET_Resonant


# ── Shared property mixin ─────────────────────────────────────────────────────

def _proxy_props(cls):
    """Add W_pos / W_phase property delegates and tick_epoch to a wrapper class."""
    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase
    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()
    cls.W_pos      = W_pos
    cls.W_phase    = W_phase
    cls.tick_epoch = tick_epoch
    return cls


# ── SGNNET_FastPhase ──────────────────────────────────────────────────────────

@_proxy_props
class SGNNET_FastPhase(nn.Module):
    """Fast W_phase: per-forward adaptation within routing loop.

    A = self.m.W_phase.clone() at start of each forward — discarded after.
    A is input-specific; gradient descent on W_phase continues as a 'slow prior'.

    Parameters
    ----------
    base_model  : SGNNET_Resonant (dynamic_z_geo mode)
    fast_rule   : 'oja_shared' | 'oja_per_sample' | 'hopfield' | 'attention'
    alpha_fast  : step size for Oja/Hopfield update and excitation weight
    tau         : softmax temperature for attention rule (default 1/sqrt(D)=0.25)
    beam_att    : sparse beam for attention rule (0=full O(N²), >0=sparse)
    """

    def __init__(
        self,
        base_model: SGNNET_Resonant,
        fast_rule: str = "oja_shared",
        alpha_fast: float = 0.1,
        tau: float = 0.25,
        beam_att: int = 32,
    ):
        super().__init__()
        self.m          = base_model
        self.fast_rule  = fast_rule
        self.alpha_fast = alpha_fast
        self.tau        = tau
        self.beam_att   = beam_att

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]
        conn_hh   = self.m.base.conn_hh                               # [N, K_hh]

        # Initialise fast weight A from slow prior
        per_sample = (self.fast_rule == "oja_per_sample")
        A = W_ph_norm.unsqueeze(0).expand(B, -1, -1).clone() if per_sample \
            else W_ph_norm.clone()  # [B,N,D] or [N,D]

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)                          # [B, N, D]
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)                   # [B, N, D]
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            # Fast W_phase excitation
            A_exp = A if per_sample else A.unsqueeze(0)               # [B/1, N, D]

            if self.fast_rule in ("oja_shared", "oja_per_sample"):
                # Excite Z toward current A where aligned, then update A via Oja
                align  = (A_exp * F.normalize(Z, dim=-1)).sum(-1, keepdim=True).clamp(min=0)
                Z_fast = A_exp * align                                # [B, N, D]
                Z      = F.normalize(
                    (Z_struct + self.m.alpha_turing * Z_inh + self.alpha_fast * Z_fast)
                    .clamp(-10, 10), dim=-1)
                sim = (A_exp * Z).sum(-1, keepdim=True)               # [B, N, 1]
                if per_sample:
                    A = F.normalize(A + self.alpha_fast * Z * sim, dim=-1)
                else:
                    A = F.normalize(A + self.alpha_fast * (Z * sim).mean(0), dim=-1)

            elif self.fast_rule == "hopfield":
                # Hopfield attractor: Z pulled toward stored A directions
                gate   = torch.sigmoid((A.unsqueeze(0) * Z).sum(-1, keepdim=True))
                Z_fast = A.unsqueeze(0) * gate
                Z      = F.normalize(
                    (Z_struct + self.m.alpha_turing * Z_inh + self.alpha_fast * Z_fast)
                    .clamp(-10, 10), dim=-1)
                # A is NOT updated — it is the stored attractor

            elif self.fast_rule == "attention":
                # Sparse attention: Z retrieves from itself with A as key matrix
                Z_n    = F.normalize(Z, dim=-1)
                scores = torch.einsum('bnd,md->bnm', Z_n, A) / self.tau  # [B,N,N]
                if 0 < self.beam_att < N:
                    topk_v, topk_i = scores.topk(self.beam_att, dim=-1)
                    mask = torch.full_like(scores, float('-inf'))
                    mask.scatter_(-1, topk_i, topk_v)
                    weights = F.softmax(mask, dim=-1)
                else:
                    weights = F.softmax(scores, dim=-1)
                Z_retrieved = torch.bmm(weights, Z)                   # [B, N, D]
                Z      = F.normalize(
                    (Z_struct + self.m.alpha_turing * Z_inh + self.alpha_fast * Z_retrieved)
                    .clamp(-10, 10), dim=-1)
                # Update A to track Z cluster centroids (batch-averaged)
                A = F.normalize(
                    torch.einsum('bnm,bmd->bnd', weights, F.normalize(Z, dim=-1)).mean(0),
                    dim=-1)

        return self.m.base._readout(Z)


# ── SGNNET_SignedCoupling ─────────────────────────────────────────────────────

@_proxy_props
class SGNNET_SignedCoupling(nn.Module):
    """Unified excitatory/inhibitory via signed cosine coupling.

    Z_h += alpha * Σ_j cos(Z_h, Z_j) * Z_j
    Positive cosine → same phase group → excite.
    Negative cosine → opposite phase group → inhibit.
    Replaces two separate exc/inh pathways with one.

    Parameters
    ----------
    sparse_k : if >0, use only top-sparse_k positive + top-sparse_k negative pairs.
               if 0, full O(N²) coupling.
    alpha_signed : coupling strength
    """

    def __init__(self, base_model: SGNNET_Resonant, alpha_signed: float = 0.3, sparse_k: int = 0):
        super().__init__()
        self.m            = base_model
        self.alpha_signed = alpha_signed
        self.sparse_k     = sparse_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            Z_n   = F.normalize(Z, dim=-1)
            sim   = torch.bmm(Z_n, Z_n.transpose(1, 2))               # [B, N, N]
            eye   = torch.eye(Z.shape[1], device=Z.device).unsqueeze(0)
            sim   = sim - 1e9 * eye                                    # mask self

            if self.sparse_k > 0:
                K = min(self.sparse_k, Z.shape[1] - 1)
                exc_v, exc_i = sim.topk(K, dim=-1)
                inh_v, inh_i = (-sim).topk(K, dim=-1)
                sparse = torch.zeros_like(sim)
                sparse.scatter_(-1, exc_i, exc_v.clamp(min=0))
                sparse.scatter_(-1, inh_i, (-inh_v).clamp(max=0))
                Z_coupled = torch.bmm(sparse, Z)
            else:
                Z_coupled = torch.bmm(sim.clamp(-1, 1), Z)

            Z = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh + self.alpha_signed * Z_coupled)
                .clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── SGNNET_STDP_S2 ────────────────────────────────────────────────────────────

@_proxy_props
class SGNNET_STDP_S2(nn.Module):
    """STDP S2: causal cross-step excitation.

    Neurons active at routing step k excite neurons that become similar at step k+1.
    Creates a 'routing agenda': early activations guide later activations within
    the same forward pass. No new learnable parameters.

    Z_prev_beam excites Z_new proportional to cosine(Z_prev_beam, Z_new).
    """

    def __init__(self, base_model: SGNNET_Resonant, alpha_stdp: float = 0.2):
        super().__init__()
        self.m          = base_model
        self.alpha_stdp = alpha_stdp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        M         = min(self.m.beam_size, N)
        Z_prev    = None

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z_new    = F.normalize(
                (Z_struct + self.m.alpha_turing * Z_inh).clamp(-10, 10), dim=-1)

            if Z_prev is not None:
                prev_idx    = Z_prev.norm(dim=-1).topk(M, dim=-1).indices  # [B, M]
                idx_exp     = prev_idx.unsqueeze(-1).expand(-1, -1, D)
                Z_pb        = torch.gather(Z_prev, 1, idx_exp)             # [B, M, D]
                cos_sim     = torch.bmm(
                    F.normalize(Z_new, dim=-1),
                    F.normalize(Z_pb, dim=-1).transpose(1, 2))             # [B, N, M]
                Z_stdp      = torch.bmm(cos_sim.clamp(min=0), Z_pb)       # [B, N, D]
                Z_new = F.normalize(
                    (Z_new + self.alpha_stdp * Z_stdp).clamp(-10, 10), dim=-1)

            Z_prev = Z.clone()
            Z      = Z_new

        return self.m.base._readout(Z)
