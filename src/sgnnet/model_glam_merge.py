"""GLAM-MERGE — locality + low-rank cross-chunk INFORMATION MERGING.

The /goal lever (2026-08-13): the LOC champion compresses each of P chunks
INDEPENDENTLY (per-channel 49->d_out) and the ONLY cross-chunk merge is the final
readout. On CIFAR-100 that leaves a STRUCTURAL -3.14pp gap vs dense (a missing-
merging signature; capacity alone recovers only ~1pp then saturates). This module
adds one cheap structured merge BETWEEN the per-chunk locality projection and the
readout: a low-rank token-mix over the P (=channel) axis.

Design (gate-death-compliant, mirrors the ΔW-proj survivor):
  - RESIDUAL: Y <- Y + mix(Y); merge grows only if it earns accuracy.
  - UNBOUNDED, applied ONCE (not a per-step squashed gate that compounds g^K).
  - W2 zero-init => net starts as pure LOC champion (identity merge), so any gain
    is attributable to the merge, and it can never start below the LOC baseline.
  - LOW-RANK r<<P is the structural prior: mixes all P channels through an r-dim
    bottleneck (2*P*r params) instead of a dense P*P mix (=the readout it precedes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_glam import GLAMLayer


class LowRankMerge(nn.Module):
    """Low-rank token-mix over the P axis, shared across the d_out slices.

    Y:[B,P,d] -> transpose [B,d,P] -> P->r->P bottleneck -> residual add.
    Params = 2*P*rank (independent of d_out and of B).
    """

    def __init__(self, P: int, rank: int, act: str = "gelu"):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(P, rank) * (1.0 / P) ** 0.5)
        self.w2 = nn.Parameter(torch.zeros(rank, P))  # zero-init => identity at start
        self.act_name = act

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        if self.act_name == "gelu":
            return F.gelu(z)
        if self.act_name == "relu":
            return F.relu(z)
        return z  # linear merge

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        t = y.transpose(1, 2)               # [B, d, P]
        h = self._act(t @ self.w1)          # [B, d, r]
        d = h @ self.w2                     # [B, d, P]
        return y + d.transpose(1, 2)        # residual, [B, P, d]


class BilinearMerge(nn.Module):
    """2nd-order low-rank cross-channel merge: Y <- Y + q(Y) ⊙ k(Y).

    Both q,k are P->r->P low-rank channel-mixes (shared across d slices). Their
    ELEMENTWISE PRODUCT is 2nd order in Y, so it encodes cross-channel interactions
    that a LINEAR readout provably cannot reproduce (unlike additive LowRankMerge,
    which the dense readout absorbs -> neutral, step008).

    Gate-death-compliant (mirrors the ΔW-proj survivor): unbounded, applied ONCE,
    residual. Geometric priming: q's output layer is zero-init (product = 0 at start
    => exact LOC identity) while k's is random, so ∂(q⊙k)/∂w_q2 ∝ k ≠ 0 gives a
    non-zero gradient from step 1 — no dead g^K compounding, no zero-gradient stall.
    Params = 4*P*rank.
    """

    def __init__(self, P: int, rank: int, act: str = "gelu"):
        super().__init__()
        s = (1.0 / P) ** 0.5
        self.wq1 = nn.Parameter(torch.randn(P, rank) * s)
        self.wq2 = nn.Parameter(torch.zeros(rank, P))   # zero => product 0 => identity start
        self.wk1 = nn.Parameter(torch.randn(P, rank) * s)
        self.wk2 = nn.Parameter(torch.randn(rank, P) * s)  # random => primes q's gradient
        self.act_name = act

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        if self.act_name == "gelu":
            return F.gelu(z)
        if self.act_name == "relu":
            return F.relu(z)
        return z

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        t = y.transpose(1, 2)                         # [B, d, P]
        q = self._act(t @ self.wq1) @ self.wq2        # [B, d, P]
        k = self._act(t @ self.wk1) @ self.wk2        # [B, d, P]
        return y + (q * k).transpose(1, 2)            # 2nd-order residual, [B, P, d]


def _make_merge(kind: str, P: int, rank: int, act: str) -> nn.Module:
    return BilinearMerge(P, rank, act) if kind == "bilinear" else LowRankMerge(P, rank, act)


class GLAMMergeNet(nn.Module):
    """LOC locality projection + cross-channel merge + linear readout.

    Everything except the merge block matches the LOC champion exactly
    (gsz=1 additive locality, shared M-expert pool, concat->readout), so the
    only new mechanism is the merge. rank=0 reproduces LOC bit-compatibly.
    merge_kind: "add" (LowRankMerge, linear) | "bilinear" (2nd-order product).
    """

    def __init__(self, in_dim: int = 25088, n_out: int = 100, P: int = 512,
                 d_out: int = 8, M: int = 16, rank: int = 32, merge_act: str = "gelu",
                 n_merge: int = 1, merge_where: str = "post", merge_kind: str = "add",
                 readout_rank: int = 0, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.P, self.d_out = P, d_out
        self.chunk_dim = in_dim // P
        self.merge_where = merge_where               # "post" (codes) | "pre" (raw chunks)
        self.loc = GLAMLayer(in_dim, P=P, d_out=d_out, M=M, G=P, gsz=1,
                             within_op="add", across_op="concat",
                             selectivity=False, seed=seed)
        # readout built BEFORE merges so its random init is rank-independent:
        # a rank sweep then differs ONLY by the (zero-init, identity-at-start) merge.
        # readout_rank>0: LOW-RANK factored readout (P*d_out -> r -> n_out). The r-bottleneck
        # MERGES all P*d_out codes into a compact class-logit basis at a fraction of the dense
        # P*d_out*n_out params. This is where param-efficiency lives (dense readout = 97% of params).
        if readout_rank > 0:
            self.readout = nn.Sequential(
                nn.Linear(P * d_out, readout_rank, bias=False),
                nn.Linear(readout_rank, n_out))
        else:
            self.readout = nn.Linear(P * d_out, n_out)
        # "post": mix the d_out codes (add REDUNDANT with dense readout -> neutral, step008;
        #         bilinear is 2nd order so NOT redundant).
        # "pre":  mix the raw chunk_dim features BEFORE per-channel compression.
        self.merges = nn.ModuleList(
            [_make_merge(merge_kind, P, rank, merge_act) for _ in range(n_merge)]
            if rank > 0 else [])

    def set_slope(self, slope: float) -> None:
        self.loc.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.merge_where == "pre" and self.merges:
            xc = x.view(B, self.P, self.chunk_dim)     # raw per-channel spatial maps
            for merge in self.merges:
                xc = merge(xc)                         # cross-channel merge BEFORE compression
            x = xc.reshape(B, self.P * self.chunk_dim)
        y = self.loc(x).view(B, self.P, self.d_out)    # LOC per-chunk projection
        if self.merge_where == "post":
            for merge in self.merges:
                y = merge(y)                           # cross-channel merge AFTER compression
        return self.readout(y.reshape(B, self.P * self.d_out))
