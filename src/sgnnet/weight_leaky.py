"""Randomized-leaky-ReLU magnitude gate on a Linear's WEIGHTS (2026-08-14).

New research line (post GLAM information-merging TERMINUS). Prior art synthesised
from RReLU (Xu 2015), LTP, Reizinger 2019, BNN/S-STE STE lineage:
  - the soft-weight-transform / hard-infer SKELETON is proven (LTP);
  - infer-HARD (slope->0 at test) is the UNVALIDATED, genuinely-novel leg — RReLU
    never infers hard (stays at slope 0.18), so it gets its own arm here;
  - accuracy prior is NEGATIVE (ffn_step002/003, RReLU is anti-overfit-on-small-data)
    -> this is an ENERGY arm: scored on real weight-zeros vs accuracy delta, not acc.

Mechanism (this file, minimal 'pct' rung — no learned threshold yet):
  m = |w|;  τ = global magnitude quantile at sparsity target s (recomputed / step;
  no learned-τ collapse trap — that is the NEXT rung, guarded by the 4 LTP fixes).
  train:  w_eff = w * gate,  gate = 1 where m>=τ else α   (leaky slope, STE grad)
          α scheduled 1.0->~0 (α=1 dense identity; randomized: α~U-band around mean)
  infer:  gate = 1 where m>=τ else 0   -> REAL zeros (energy / effective-param cut)
The gate is computed under no_grad so d/dw (w*gate)=gate = exact leaky-slope surrogate
(1 in kept region, α in pruned region) — the BNN/leaky-STE pattern, applied to weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyPrunedLinear(nn.Module):
    """nn.Linear whose weight passes a randomized-leaky magnitude gate.

    sparsity : target fraction of weights in the leaky/pruned (sub-τ) region.
    randomized: if True, α sampled per forward ~ U(α/r, α*r) clipped [0,1] (RReLU
                transfer; per-STEP not per-example — weights are shared across batch).
    """

    def __init__(self, in_features: int, out_features: int, sparsity: float = 0.7,
                 randomized: bool = False, rand_ratio: float = 2.0, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.sparsity, self.randomized, self.rand_ratio = sparsity, randomized, rand_ratio
        self.slope = 1.0  # α; schedule drives 1.0 -> ~0 over training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def set_slope(self, slope: float) -> None:
        self.slope = float(slope)

    def _threshold(self) -> torch.Tensor:
        # global magnitude quantile at the sparsity target; no grad (STE boundary).
        m = self.weight.detach().abs()
        return torch.quantile(m.flatten(), self.sparsity)

    def _gate(self) -> torch.Tensor:
        with torch.no_grad():
            m = self.weight.abs()
            keep = m >= self._threshold()
            if self.training:
                a = self.slope
                if self.randomized and a < 1.0:
                    lo, hi = a / self.rand_ratio, min(1.0, a * self.rand_ratio)
                    a = torch.empty_like(m).uniform_(lo, hi)
                gate = torch.where(keep, torch.ones_like(m),
                                   torch.as_tensor(a, dtype=m.dtype, device=m.device)
                                   if not torch.is_tensor(a) else a)
            else:
                gate = keep.to(m.dtype)  # infer-HARD: real zeros
        return gate

    def zero_fraction(self) -> float:
        with torch.no_grad():
            m = self.weight.abs()
            return (m < self._threshold()).float().mean().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.weight * self._gate()  # gate const wrt w -> leaky-slope STE grad
        return F.linear(x, w_eff, self.bias)


def attach_leaky_readout(net: nn.Module, sparsity: float, randomized: bool,
                         rand_ratio: float = 2.0) -> LeakyPrunedLinear:
    """Swap net.readout (nn.Linear) for a LeakyPrunedLinear, copying its weights.

    Returns the new readout so the train loop can drive its slope schedule directly.
    """
    old = net.readout
    new = LeakyPrunedLinear(old.in_features, old.out_features, sparsity=sparsity,
                            randomized=randomized, rand_ratio=rand_ratio,
                            bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    net.readout = new
    return new
