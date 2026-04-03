"""SGNNET_SpatialGrouped: learned per-group input projection with bridge neurons.

Design
------
Neurons are split into two populations:

  Exclusive neurons (fraction = 1 - overlap_frac):
    Each group g has its own learned linear layer (G_in → ng_exclusive).
    Group g sees ONLY its assigned input slice — no cross-group leakage.
    Specialisation: group g neurons learn features specific to spatial region g.

  Bridge neurons (fraction = overlap_frac):
    Random K_in=50 sparse gather from the FULL input range (all N_in dims).
    Cross-group mixing: each bridge neuron randomly integrates inputs from
    ANY spatial region, providing inter-group communication at the input stage.
    Zero extra parameters — reuses the fixed sparse gather mechanism.

The two populations concatenate to form Z [B, N_hidden, D], then proceed
through the usual small-world routing (conn_hh unchanged).

Z assembly:
  Exclusive: vals[b,h] × normalize(W_pos[h]) → [B, n_exclusive, D]
    Scalar projection value modulates the neuron's learned geometric direction.
  Bridge:    gather K_in inputs → sum → [B, n_bridge, D] (same as current Ref seed)

Spatial modes (for exclusive neurons only):
  'flat'      — contiguous blocks of flattened 25088 (no permute)
  'rows'      — VGG16 spatial rows: view [B,512,7,7] → [B,7,3584]
  'positions' — VGG16 positions:    view [B,512,7,7] → [B,49,512]

Parameter counts (N=1024, D=64, N_in=25088, overlap_frac=0.2):
  n_exclusive=819, n_bridge=205
  n_groups=7  separate: ~3.0M   (row-level VGG16 spatial)
  n_groups=49 separate: ~430K   (position-level VGG16 spatial)
  n_groups=8  shared:   ~329K   (flat equal blocks, shared weights)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld


class SGNNET_SpatialGrouped(SGNNET_SmallWorld):
    """Learned per-group projection + bridge neurons for cross-group mixing.

    Parameters
    ----------
    n_groups_proj : number of spatial input groups (7=rows, 49=positions, 8=flat)
    shared_proj   : if True all groups share one weight matrix
    spatial_mode  : 'flat' | 'rows' | 'positions'
    overlap_frac  : fraction of N_hidden neurons allocated as bridge neurons
                    (receive from full N_in range via sparse K_in gather)
                    0.0 = fully exclusive, 0.2 = 20% bridge neurons
    vgg_channels  : VGG16 pool5 channels (default 512)
    vgg_spatial   : VGG16 pool5 spatial size (default 7)
    bridge_seed   : RNG seed for bridge neuron conn_in construction

    All remaining kwargs forwarded to SGNNET_SmallWorld.
    """

    def __init__(
        self,
        n_groups_proj: int = 7,
        shared_proj: bool = False,
        spatial_mode: str = "flat",
        overlap_frac: float = 0.2,
        vgg_channels: int = 512,
        vgg_spatial: int = 7,
        bridge_seed: int = 99,
        **kwargs,
    ):
        super().__init__(**kwargs)

        N_hidden = self.N_hidden
        N_in     = self.N_in
        D        = self.D
        K_in     = kwargs.get("K_in", 50)

        self.n_groups_proj = n_groups_proj
        self.shared_proj   = shared_proj
        self.spatial_mode  = spatial_mode
        self.overlap_frac  = overlap_frac
        self.vgg_channels  = vgg_channels
        self.vgg_spatial   = vgg_spatial

        # ── Neuron budget split ────────────────────────────────────────────
        n_bridge    = round(N_hidden * overlap_frac)
        n_exclusive = N_hidden - n_bridge
        self.n_bridge    = n_bridge
        self.n_exclusive = n_exclusive

        # ── Validate spatial mode & compute G_in ──────────────────────────
        if spatial_mode == "rows":
            assert n_groups_proj == vgg_spatial, (
                f"spatial_mode='rows' requires n_groups_proj={vgg_spatial}, "
                f"got {n_groups_proj}"
            )
            assert N_in == vgg_channels * vgg_spatial * vgg_spatial
            G_in = vgg_channels * vgg_spatial   # 512 × 7 = 3584
        elif spatial_mode == "positions":
            assert n_groups_proj == vgg_spatial * vgg_spatial, (
                f"spatial_mode='positions' requires n_groups_proj={vgg_spatial**2}, "
                f"got {n_groups_proj}"
            )
            assert N_in == vgg_channels * vgg_spatial * vgg_spatial
            G_in = vgg_channels                  # 512
        else:  # flat
            assert N_in % n_groups_proj == 0, (
                f"N_in={N_in} must be divisible by n_groups_proj={n_groups_proj}"
            )
            G_in = N_in // n_groups_proj

        self.G_in = G_in

        # ── Per-group output sizes for exclusive neurons ───────────────────
        base_ng  = n_exclusive // n_groups_proj
        extra    = n_exclusive  % n_groups_proj
        # First `extra` groups get base_ng+1
        ng_sizes = [base_ng + (1 if g < extra else 0) for g in range(n_groups_proj)]
        assert sum(ng_sizes) == n_exclusive
        self.ng_sizes   = ng_sizes
        self.ng_offsets = [sum(ng_sizes[:g]) for g in range(n_groups_proj + 1)]

        # ── Learned projections (exclusive neurons only) ───────────────────
        if shared_proj:
            ng_shared = ng_sizes[0]
            assert all(s == ng_shared for s in ng_sizes), (
                f"shared_proj requires n_exclusive divisible by n_groups_proj; "
                f"n_exclusive={n_exclusive}, n_groups_proj={n_groups_proj}"
            )
            self.W_proj = nn.Linear(G_in, ng_shared, bias=False)
            nn.init.kaiming_uniform_(self.W_proj.weight, a=math.sqrt(5))
        else:
            self.W_proj = nn.ModuleList([
                nn.Linear(G_in, ng_sizes[g], bias=False)
                for g in range(n_groups_proj)
            ])
            for layer in self.W_proj:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        # ── Bridge neuron connectivity (global random sparse) ──────────────
        # Neurons [n_exclusive : N_hidden] receive from random K_in positions
        # spanning the FULL N_in range — independent of group boundaries.
        if n_bridge > 0:
            rng = np.random.default_rng(bridge_seed)
            conn_bridge = rng.integers(0, N_in, size=(n_bridge, K_in)).astype(np.int64)
            self.register_buffer(
                "conn_bridge", torch.tensor(conn_bridge, dtype=torch.long)
            )   # [n_bridge, K_in]
        else:
            self.conn_bridge = None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return x as [B, n_groups_proj, G_in] for the chosen spatial mode."""
        B = x.shape[0]
        if self.spatial_mode == "rows":
            # channel-major [512,7,7] → row-major [7,3584]
            x3d = x.view(B, self.vgg_channels, self.vgg_spatial, self.vgg_spatial)
            return x3d.permute(0, 2, 1, 3).reshape(B, self.n_groups_proj, self.G_in)
        elif self.spatial_mode == "positions":
            # [512,7,7] → [49,512]: group by position, not channel
            x3d = x.view(B, self.vgg_channels, self.vgg_spatial, self.vgg_spatial)
            return x3d.permute(0, 2, 3, 1).reshape(B, self.n_groups_proj, self.G_in)
        else:  # flat
            return x.view(B, self.n_groups_proj, self.G_in)

    # ── Override _seed ─────────────────────────────────────────────────────────

    def _seed(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid seed: per-group learned projection + global bridge gather.

        Exclusive neurons [0 : n_exclusive]:
          1. Reshape x → [B, n_groups, G_in]
          2. Per-group linear: G_in → ng_g scalars
          3. Z_excl = val × normalize(W_pos[h])  → [B, n_exclusive, D]

        Bridge neurons [n_exclusive : N_hidden]:
          4. Gather K_in inputs from full N_in range (conn_bridge)
          5. Z_bridge = sum of [val, spatial_coords] over K_in  → [B, n_bridge, D]

        Both populations normalised; concat → [B, N_hidden, D].
        """
        B = x.shape[0]

        # ── Exclusive: per-group projection ───────────────────────────────
        x_groups = self._reshape_input(x)   # [B, n_groups, G_in]
        parts = []
        for g in range(self.n_groups_proj):
            x_g = x_groups[:, g, :]  # [B, G_in]
            if self.shared_proj:
                v_g = self.W_proj(x_g)        # [B, ng]
            else:
                v_g = self.W_proj[g](x_g)     # [B, ng_g]
            parts.append(v_g)

        excl_vals = torch.cat(parts, dim=1)   # [B, n_exclusive]

        # Assemble D-dim: scalar × learned geometric direction
        W_pos_excl = F.normalize(self.W_pos[:self.n_exclusive], dim=-1)  # [n_exclusive, D]
        Z_excl = excl_vals.unsqueeze(-1) * W_pos_excl.unsqueeze(0)       # [B, n_exclusive, D]
        Z_excl = self._normalise(Z_excl)

        # ── Bridge: global sparse gather (cross-group mixing) ─────────────
        if self.n_bridge > 0:
            spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)  # [B, N_in, D-1]
            A_input = torch.cat([x.unsqueeze(-1), spatial], dim=-1)        # [B, N_in, D]
            Z_bridge = A_input[:, self.conn_bridge, :].sum(dim=2)          # [B, n_bridge, D]
            Z_bridge = self._normalise(Z_bridge)
            return torch.cat([Z_excl, Z_bridge], dim=1)                    # [B, N_hidden, D]
        else:
            return Z_excl
