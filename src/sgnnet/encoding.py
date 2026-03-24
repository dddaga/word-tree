"""Spatial encoding for VGG16 pool5 feature maps.

Maps flat indices [0, n_in) of the [C, H, W] feature map to normalized
spatial coordinates [channel_norm, h_norm, w_norm] in [0, 1]^3.
"""

from __future__ import annotations

import torch


def compute_spatial_encoding(
    n_in: int = 25088,
    h: int = 7,
    w: int = 7,
    c: int = 512,
) -> torch.Tensor:
    """Precompute normalized spatial coordinates for each input neuron.

    For flat index k in [0, n_in):
      channel = k // (h * w)
      row     = (k % (h * w)) // w
      col     = (k % (h * w)) % w

    Returns [n_in, 3] float32 tensor of [channel_norm, h_norm, w_norm].
    """
    k = torch.arange(n_in, dtype=torch.long)
    hw = h * w

    channel = k // hw
    row = (k % hw) // w
    col = (k % hw) % w

    channel_norm = channel.float() / max(c - 1, 1)
    h_norm = row.float() / max(h - 1, 1)
    w_norm = col.float() / max(w - 1, 1)

    return torch.stack([channel_norm, h_norm, w_norm], dim=-1)
