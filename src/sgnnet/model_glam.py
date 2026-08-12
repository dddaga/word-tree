"""GLAM — Grouped Local Anti-hebbian Multiplicative routing, as a stackable LAYER.

Mental model (user, 2026-08-13): a new layer type that
  1. takes an input vector, breaks it into P chunks;
  2. operates on each chunk with weights drawn from a SHARED pool (param reduction);
  3. WITHIN a group of chunks, MULTIPLIES the chunk outputs (2nd-order interaction);
  4. ACROSS groups, ADDS the group outputs.
within-op (mul) and across-op (add) are the DEFAULT; both are flags because "there
could be multiple permutations/combinations to figure out what is more effective."

Design rules from prior evidence (see concepts/glam_grouped_multiplicative_routing.md):
  - the product is UNBOUNDED + RESIDUAL, applied ONCE per layer, never a squashed
    per-step gate that compounds g^K (gate_death.md; the ΔW-proj survivor is unbounded).
  - assignment/grouping are FROZEN topology buffers, built once (FM6).
  - selectivity shares AH's signal path → ablated with AH off; AH is a separate arm.
All causal claims HYPOTHESIS until the T0 ladder lands.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _constrained_assign(n_slots: int, M: int, seed: int) -> np.ndarray:
    """Assign each of n_slots to a pool expert in [0,M), penalising reuse.

    p(m) ∝ max(eps, 1 - used_m / quota), quota = ceil(n_slots / M). Frozen topology.
    """
    rng = np.random.default_rng(seed)
    quota = math.ceil(n_slots / M)
    used = np.zeros(M)
    out = np.empty(n_slots, dtype=np.int64)
    for i in range(n_slots):
        w = np.maximum(1e-3, 1.0 - used / quota)
        p = w / w.sum()
        m = rng.choice(M, p=p)
        out[i] = m
        used[m] += 1
    return out


def _build_groups(P: int, G: int, gsz: int, seed: int) -> np.ndarray:
    """Return [G, gsz] chunk indices per group by constrained random pairing.

    Second-order = gsz=2. Chunks sampled without over-reuse across groups (FM6 frozen).
    """
    assign = _constrained_assign(G * gsz, P, seed + 1).reshape(G, gsz)
    return assign


class GLAMLayer(nn.Module):
    """One GLAM layer: [B, in_dim] -> [B, out_dim].

    P chunks of chunk_dim; each chunk -> d_out via a shared pool of M experts.
    Groups of gsz chunks; within-group reduce (mul/add), across handled by out shape.
    out_dim = G * d_out (across_op='concat', stackable) or d_out (across_op='add').
    """

    def __init__(self, in_dim: int, P: int, d_out: int, M: int, G: int,
                 gsz: int = 2, within_op: str = "mul", across_op: str = "concat",
                 selectivity: bool = False, seed: int = 42):
        super().__init__()
        assert in_dim % P == 0, f"in_dim {in_dim} not divisible by P {P}"
        self.P, self.chunk_dim, self.d_out = P, in_dim // P, d_out
        self.M, self.G, self.gsz = M, G, gsz
        self.within_op, self.across_op = within_op, across_op
        self.selectivity = selectivity
        self.slope = 1.0  # leaky-ReLU negative slope; schedule sets it each epoch
        self.aux_loss = torch.zeros(())  # decorrelation penalty, read by train loop

        # Shared expert pool: M matrices [chunk_dim, d_out]. He-scaled.
        self.E = nn.Parameter(torch.randn(M, self.chunk_dim, d_out) * (2.0 / self.chunk_dim) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(M, d_out))
        if selectivity:
            self.proto = nn.Parameter(F.normalize(torch.randn(M, self.chunk_dim), dim=-1))

        # Frozen topology buffers
        chunk_assign = torch.tensor(_constrained_assign(P, M, seed), dtype=torch.long)
        self.register_buffer("chunk_assign", chunk_assign)          # [P] chunk -> expert
        groups = torch.tensor(_build_groups(P, G, gsz, seed), dtype=torch.long)
        self.register_buffer("groups", groups)                      # [G, gsz] group -> chunks

        self.out_dim = (G * d_out) if across_op == "concat" else d_out

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(z, negative_slope=self.slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        xc = x.view(B, self.P, self.chunk_dim)                      # [B, P, chunk_dim]
        Ec = self.E[self.chunk_assign]                             # [P, chunk_dim, d_out]
        y = torch.einsum("bpc,pcd->bpd", xc, Ec) + self.bias[self.chunk_assign]

        if self.selectivity:
            # affinity of each chunk to its expert prototype; scales contribution
            # (unbounded |cos|, geometrically primed — mirrors ΔW-proj, not sigmoid).
            proto_c = self.proto[self.chunk_assign]               # [P, chunk_dim]
            a = F.cosine_similarity(xc, proto_c.unsqueeze(0), dim=-1)  # [B, P]
            y = y * a.abs().unsqueeze(-1)

        y = self._act(y)

        yg = y[:, self.groups, :]                                  # [B, G, gsz, d_out]
        if self.within_op == "mul":
            g = yg.prod(dim=2)                                     # WITHIN group: multiply
        else:
            g = yg.sum(dim=2)

        if self.selectivity and self.training:
            self.aux_loss = self._decor(yg)

        if self.across_op == "add":
            agg = g.sum(dim=1)                                     # ACROSS groups: add -> [B, d_out]
            agg = F.layer_norm(agg, (self.d_out,))                # re-frame the group-of-groups
            return self._act(agg)
        return g.reshape(B, self.G * self.d_out)                   # concat -> stackable (next layer reframes)

    def _decor(self, yg: torch.Tensor) -> torch.Tensor:
        """Anti-Hebbian: penalise the paired chunks in a group co-firing (gsz=2)."""
        if self.gsz != 2:
            return torch.zeros((), device=yg.device)
        a, b = yg[:, :, 0, :], yg[:, :, 1, :]                      # [B, G, d_out]
        a = a - a.mean(0, keepdim=True)
        b = b - b.mean(0, keepdim=True)
        num = (a * b).mean(0)
        den = a.pow(2).mean(0).sqrt() * b.pow(2).mean(0).sqrt() + 1e-6
        return (num / den).abs().mean()


class GLAMNet(nn.Module):
    """Stack of L GLAM layers + linear readout. Depth-over-breadth vehicle.

    Layer 0 partitions the raw input (25088 = 512*49 -> P=512, chunk_dim=49).
    Deeper layers partition the previous out_dim. Final layer collapses (across='add').
    """

    def __init__(self, in_dim: int = 25088, n_out: int = 10, L: int = 1,
                 P: int = 512, d_out: int = 8, M: int = 16, G: int = 512,
                 gsz: int = 2, within_op: str = "mul", across_op: str = "concat",
                 selectivity: bool = False, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        cur = in_dim
        for li in range(L):
            last = li == L - 1
            a_op = "add" if (last and L > 1) else across_op
            p = P if li == 0 else min(G, cur)
            p = max(1, next(d for d in range(p, 0, -1) if cur % d == 0))  # largest divisor <= p
            lyr = GLAMLayer(cur, P=p, d_out=d_out, M=M, G=G, gsz=gsz,
                            within_op=within_op, across_op=a_op,
                            selectivity=selectivity, seed=seed + li)
            layers.append(lyr)
            cur = lyr.out_dim
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(cur, n_out)

    def set_slope(self, slope: float) -> None:
        for lyr in self.layers:
            lyr.slope = slope

    def aux_loss(self) -> torch.Tensor:
        return sum(lyr.aux_loss for lyr in self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lyr in self.layers:
            x = lyr(x)
        return self.readout(x)
