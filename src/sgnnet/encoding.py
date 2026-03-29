"""Spatial encoding for VGG16 pool5 feature maps.

Two modes:
  'linear'  — [channel_norm, h_norm, w_norm] in [0,1]^3, always 3 dims (original)
  'fourier' — sinusoidal Fourier embedding to D-1 dims, one dim reserved for
              the raw feature value which is concatenated in _seed.

Fourier embedding places each (h, w, channel) position as a distinct near-orthogonal
direction on S^(D-2) before the feature value is appended. Two positions that were
linearly close in the original encoding can become well-separated in Fourier space,
giving hidden neurons richer initial conditions and better direction diversity on S^(D-1).

Allocation (all dims are sin/cos of normalised coordinates × π):
  D=4  → 3 spatial dims: sin(h), sin(w), sin(c·2π)           (single-freq)
  D=8  → 7 spatial dims: sin+cos for h, w, c (6 dims) + sin(2h) (second freq)
  D=16 → 15 spatial dims: 2 freqs each for h, w, c (12 dims) + 3 linear fallback dims
  other D → generalises: ⌊(D-1)/3⌋ sin+cos pairs per axis, remainder filled linearly

After concatenation with feature value, l2-normalisation in _seed maps the full
D-dim vector onto S^(D-1).
"""

from __future__ import annotations
import math
import torch


def compute_spatial_encoding(
    n_in: int = 25088,
    h: int = 7,
    w: int = 7,
    c: int = 512,
) -> torch.Tensor:
    """Linear encoding: [channel_norm, h_norm, w_norm] — always 3 dims.

    Original encoding. Returns [n_in, 3] float32.
    Only valid for D=4 (3 spatial + 1 feature value).
    """
    k   = torch.arange(n_in, dtype=torch.long)
    hw  = h * w
    ch  = k // hw
    row = (k % hw) // w
    col = (k % hw) % w

    return torch.stack([
        ch.float()  / max(c - 1, 1),
        row.float() / max(h - 1, 1),
        col.float() / max(w - 1, 1),
    ], dim=-1)   # [n_in, 3]


def compute_fourier_encoding(
    n_in: int = 25088,
    D: int = 4,
    h: int = 7,
    w: int = 7,
    c: int = 512,
) -> torch.Tensor:
    """Fourier sinusoidal encoding: returns [n_in, D-1] float32.

    Reserves 1 dim for the feature value (concatenated in _seed).
    The D-1 spatial dims use sin/cos pairs at increasing frequencies so that
    every (row, col, channel) triple maps to a distinct direction on S^(D-2).

    At D=4: three single-freq terms  [sin(h), sin(w), sin(c)]
    At D=8: sin+cos for h, w, c (6) + sin(2h) (1 extra)
    At D=16: 2 freqs for h, w, c (12) + 3 linear fallback dims

    After l2-normalisation with the feature value, the seed lives on S^(D-1).
    """
    k   = torch.arange(n_in, dtype=torch.long)
    hw  = h * w
    ch  = k // hw
    row = (k % hw) // w
    col = (k % hw) % w

    # Normalise axes to [0, 1]
    h_n = row.float() / max(h - 1, 1)   # [n_in]
    w_n = col.float() / max(w - 1, 1)
    c_n = ch.float()  / max(c - 1, 1)

    dims: list[torch.Tensor] = []
    spatial_D = D - 1   # dims available for spatial encoding

    # How many sin/cos pairs (= freqs) can we afford per axis?
    # We have 3 axes (h, w, c).  Each pair costs 2 dims.
    # Reserve any remainder for linear fallback dims.
    n_pairs = spatial_D // (3 * 2)          # full pairs per axis
    remainder = spatial_D - n_pairs * 3 * 2  # leftover dims

    # Build pairs: freq 1, 2, ... for each axis
    for freq in range(1, n_pairs + 1):
        scale = freq * math.pi
        dims += [torch.sin(h_n * scale), torch.cos(h_n * scale)]
        dims += [torch.sin(w_n * scale), torch.cos(w_n * scale)]
        dims += [torch.sin(c_n * scale * 2), torch.cos(c_n * scale * 2)]
        # c uses 2π base so it wraps around (channel is not ordered like position)

    # Fill remainder with single-freq terms we haven't covered yet, then linear
    extra_candidates = [
        torch.sin(h_n * (n_pairs + 1) * math.pi),
        torch.sin(w_n * (n_pairs + 1) * math.pi),
        torch.sin(c_n * (n_pairs + 1) * math.pi * 2),
        h_n, w_n, c_n,  # linear fallback
    ]
    for t in extra_candidates[:remainder]:
        dims.append(t)

    # Edge case: D=4 with n_pairs=0
    if not dims:
        dims = [
            torch.sin(h_n * math.pi),
            torch.sin(w_n * math.pi),
            torch.sin(c_n * math.pi * 2),
        ][:spatial_D]
        # pad with zeros if still short (shouldn't happen for D>=4)
        while len(dims) < spatial_D:
            dims.append(torch.zeros(n_in))

    enc = torch.stack(dims[:spatial_D], dim=-1).float()   # [n_in, D-1]
    return enc
