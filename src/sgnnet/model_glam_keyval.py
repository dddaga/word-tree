"""GLAM key×value — the user's faithful mechanism (2026-08-13).

Distinct from model_glam.py's gsz (which paired two CHUNKS). Here each chunk = one
group, attended by TWO pooled weights: a KEY and a VALUE (distinct expert pair per
chunk). The key passes through an activation (sigmoid) first, then multiplies the
value's raw output — asymmetric, applied ONCE per layer:

    key = sigmoid(E_k · x_c + b_k)          # squashed selector, [B, P, d_out]
    val =         E_v · x_c + b_v           # raw content
    y_c = key * val                          # asymmetric multiplicative combine

Diversity is combinatorial: n weights give nC2 distinct pairs, so the pool floor is
MC2 >= P. For the VGG head (P=512) that forces M >= 33 (33C2=528) — the additive
champion's M=16 (120 pairs) is BELOW this floor and cannot serve distinct pairs.

GATE-DEATH NOTE (CONFIRMED prior): sigmoid(key) is a bounded [0,1] gate — the dead
shape. It is applied ONCE per layer (not compounded over K routing steps), the
survivable regime (cf. the ΔW-proj survivor). key_act='none' isolates the sigmoid's
effect (raw asymmetric product, still distinct-pair).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _distinct_pairs(P: int, M: int, seed: int) -> np.ndarray:
    """Return [P, 2] distinct unordered expert pairs from a pool of M. Frozen topology.

    Requires MC2 = M(M-1)/2 >= P so every chunk gets a unique (key, value) pair.
    """
    n_pairs = M * (M - 1) // 2
    assert n_pairs >= P, f"pool M={M} gives only {n_pairs} pairs < P={P}; need M>= ~{int((2*P)**0.5)+1}"
    all_pairs = np.array([(i, j) for i in range(M) for j in range(i + 1, M)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_pairs)[:P]
    pairs = all_pairs[idx]
    # randomise which side is key vs value per chunk (break i<j asymmetry bias)
    flip = rng.integers(0, 2, size=P).astype(bool)
    pairs[flip] = pairs[flip][:, ::-1]
    return pairs


class GLAMKeyValLayer(nn.Module):
    """[B, in_dim] -> [B, out_dim]. Each chunk: sigmoid(key)*value, distinct pair."""

    def __init__(self, in_dim: int, P: int, d_out: int, M: int,
                 key_act: str = "sigmoid", across_op: str = "concat", seed: int = 42):
        super().__init__()
        assert in_dim % P == 0, f"in_dim {in_dim} not divisible by P {P}"
        self.P, self.chunk_dim, self.d_out = P, in_dim // P, d_out
        self.M, self.key_act, self.across_op = M, key_act, across_op
        self.slope = 1.0  # value-branch leaky-ReLU slope (schedule sets each epoch)

        self.E = nn.Parameter(torch.randn(M, self.chunk_dim, d_out) * (2.0 / self.chunk_dim) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(M, d_out))

        pairs = torch.tensor(_distinct_pairs(P, M, seed), dtype=torch.long)
        self.register_buffer("key_idx", pairs[:, 0])                 # [P] chunk -> key expert
        self.register_buffer("val_idx", pairs[:, 1])                 # [P] chunk -> value expert

        self.out_dim = (P * d_out) if across_op == "concat" else d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        xc = x.view(B, self.P, self.chunk_dim)                       # [B, P, chunk_dim]
        Ek, Ev = self.E[self.key_idx], self.E[self.val_idx]          # [P, chunk_dim, d_out]
        key = torch.einsum("bpc,pcd->bpd", xc, Ek) + self.bias[self.key_idx]
        val = torch.einsum("bpc,pcd->bpd", xc, Ev) + self.bias[self.val_idx]
        key = torch.sigmoid(key) if self.key_act == "sigmoid" else key
        val = F.leaky_relu(val, negative_slope=self.slope)
        y = key * val                                                # asymmetric, once
        if self.across_op == "add":
            agg = y.sum(dim=1)                                       # [B, d_out] (collapses)
            return F.layer_norm(agg, (self.d_out,))
        return y.reshape(B, self.P * self.d_out)                     # concat -> wide, stackable


class GLAMKeyValNet(nn.Module):
    """Stack of key×value layers + linear readout. Layer 0 partitions raw input."""

    def __init__(self, in_dim: int = 25088, n_out: int = 10, L: int = 1,
                 P: int = 512, d_out: int = 8, M: int = 34,
                 key_act: str = "sigmoid", across_op: str = "concat",
                 seed: int = 42, collapse_last: bool = False):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        cur = in_dim
        for li in range(L):
            last = li == L - 1
            a_op = "add" if (last and L > 1 and collapse_last) else across_op
            p = P if li == 0 else max(1, next(d for d in range(min(P, cur), 0, -1) if cur % d == 0))
            lyr = GLAMKeyValLayer(cur, P=p, d_out=d_out, M=M, key_act=key_act,
                                  across_op=a_op, seed=seed + li)
            layers.append(lyr)
            cur = lyr.out_dim
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(cur, n_out)

    def set_slope(self, slope: float) -> None:
        for lyr in self.layers:
            lyr.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lyr in self.layers:
            x = lyr(x)
        return self.readout(x)
