"""Stacked and parallel SGNNET compositions for step64 depth experiments.

MOTIVATION
==========
All wave-1 mechanisms (steps 58-63) failed by adding excitation additively —
the routing diversity collapsed via safety-valve death. A fundamentally different
hypothesis: instead of adding mechanisms WITHIN a routing layer, stack multiple
independent routing layers IN SERIES or run them IN PARALLEL.

Key question: does depth (more routing layers) help, or does the single-layer
Gen4 AntiHebb already extract all learnable structure in 8 iterations?

ARCHITECTURE
============

SGNNET_Stacked  — N layers of SGNNET_AntiHebbian in series.
    Layer 1:  Z = seed(x) → route_layer1(Z) → Z1
    Layer 2+: Z = route_layerK(Z_prev) → ZK
    Final:    logits = readout(ZK)  [from layer-1's _readout]

    series_mode='passthrough': Z flows between layers unchanged (no new params).
    series_mode='bridge': learned nn.Linear(D,D,bias=False) between layers;
                          Z is re-normalized after each bridge.
    skip=True: Z_out = normalize(Z_layer_output + Z_layer_input) for layers 2+.

SGNNET_Parallel — two independent streams fused before readout.
    stream_a: seed_a(x) → route_a(Z) → Z_a
    stream_b: seed_b(x) → route_b(Z) → Z_b
    fusion_mode='sum':    Z_fused = normalize(Z_a + Z_b)
    fusion_mode='concat': Z_cat = [Z_a, Z_b] [B,N,2D], proj nn.Linear(2D,D,bias=False),
                          Z_fused = normalize(proj(Z_cat))
    Final: logits = stream_a._readout(Z_fused)

CONFIGS (see train_step64_stacked_sgnnet.py)
============================================
  Ref  — 1-layer baseline
  A    — 2-layer series passthrough, no skip
  B    — 2-layer series passthrough, skip
  C    — 2-layer series bridge, no skip
  D    — 3-layer series passthrough, no skip
  E    — 2-parallel sum
  F    — 2-parallel concat-project
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mechanisms_inhibitory import SGNNET_AntiHebbian


# ---------------------------------------------------------------------------
# Helper: run one AntiHebbian routing iteration without seed/readout
# ---------------------------------------------------------------------------

def _route_layer(layer: SGNNET_AntiHebbian, Z: torch.Tensor) -> torch.Tensor:
    """Run the AH routing loop on Z, skipping _seed and _readout.

    Identical logic to SGNNET_AntiHebbian.forward() but operates on an
    already-seeded Z [B, N, D] — used when chaining layers in series.
    """
    resonant  = layer.m                          # SGNNET_Resonant
    base      = resonant.base                    # SGNNET_SmallWorld
    theta_pos = resonant.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1,N,1]
    W_ph_norm = F.normalize(resonant.W_phase, dim=-1)
    conn_hh   = base.conn_hh

    # Pre-compute static AH suppression weights (wpos variant)
    N_h    = base.N_hidden
    W_n    = F.normalize(layer.m.W_pos[:N_h], dim=-1)       # [N, D]
    pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)     # [N, K_hh]
    supp_w  = (1.0 - layer.alpha_ahebb * pos_sim.clamp(min=0)
               ).unsqueeze(0).unsqueeze(-1)                  # [1,N,K_hh,1]

    for _ in range(base.K_iter):
        Z_fwd    = F.relu(Z - theta_pos)
        Z_nb     = Z_fwd[:, conn_hh, :]                     # [B,N,K_hh,D]
        Z_struct = (Z_nb * supp_w).sum(dim=2)               # [B,N,D]
        Z_inh    = resonant._phase_inhibit(Z, W_ph_norm, theta_pos)
        Z = F.normalize(
            (Z_struct + resonant.alpha_turing * Z_inh).clamp(-10, 10), dim=-1
        )

    return Z


# ---------------------------------------------------------------------------
# SGNNET_Stacked
# ---------------------------------------------------------------------------

class SGNNET_Stacked(nn.Module):
    """N stacked SGNNET_AntiHebbian layers in series.

    Parameters
    ----------
    layers       : list of SGNNET_AntiHebbian, each independently initialised
    series_mode  : 'passthrough' — Z flows directly; 'bridge' — linear(D,D) between layers
    skip         : if True, layers 2+ apply residual: Z_out = norm(Z_out + Z_in)
    """

    def __init__(
        self,
        layers: List[SGNNET_AntiHebbian],
        series_mode: str = "passthrough",
        skip: bool = False,
    ):
        super().__init__()
        assert len(layers) >= 1, "need at least one layer"
        assert series_mode in ("passthrough", "bridge"), \
            f"unknown series_mode={series_mode!r}"

        self.layers      = nn.ModuleList(layers)
        self.series_mode = series_mode
        self.skip        = skip

        D = layers[0].m.base.D
        if series_mode == "bridge" and len(layers) > 1:
            # One bridge per adjacent pair
            self.bridges = nn.ModuleList([
                nn.Linear(D, D, bias=False) for _ in range(len(layers) - 1)
            ])
        else:
            self.bridges = None

    # -- Trainer compatibility -----------------------------------------------

    @property
    def W_pos(self):
        """Expose first layer W_pos so Trainer safety valve can access it."""
        return self.layers[0].W_pos

    @property
    def W_phase(self):
        return None

    def tick_epoch(self):
        for layer in self.layers:
            layer.tick_epoch()

    # -- Forward ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer0 = self.layers[0]

        # Layer 0: seed → route
        Z = layer0.m.base._seed(x)
        Z = _route_layer(layer0, Z)

        # Layers 1+
        for idx, layer in enumerate(self.layers[1:], start=1):
            Z_in = Z

            # Bridge (optional linear projection between layers)
            if self.bridges is not None:
                Z = F.normalize(self.bridges[idx - 1](Z), dim=-1)

            Z = _route_layer(layer, Z)

            # Skip connection (residual from layer input)
            if self.skip:
                Z = F.normalize(Z + Z_in, dim=-1)

        # Readout from first layer's _readout
        return layer0.m.base._readout(Z)


# ---------------------------------------------------------------------------
# SGNNET_Parallel
# ---------------------------------------------------------------------------

class SGNNET_Parallel(nn.Module):
    """Two independent SGNNET_AntiHebbian streams fused before readout.

    Parameters
    ----------
    stream_a     : SGNNET_AntiHebbian — primary stream (readout is from here)
    stream_b     : SGNNET_AntiHebbian — secondary stream
    fusion_mode  : 'sum' — Z_fused = norm(Z_a + Z_b)
                   'concat' — Z_cat=[Z_a,Z_b], proj linear(2D,D), Z_fused=norm(proj(Z_cat))
    """

    def __init__(
        self,
        stream_a: SGNNET_AntiHebbian,
        stream_b: SGNNET_AntiHebbian,
        fusion_mode: str = "sum",
    ):
        super().__init__()
        assert fusion_mode in ("sum", "concat"), \
            f"unknown fusion_mode={fusion_mode!r}"

        self.stream_a    = stream_a
        self.stream_b    = stream_b
        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            D = stream_a.m.base.D
            self.proj = nn.Linear(2 * D, D, bias=False)
        else:
            self.proj = None

    # -- Trainer compatibility -----------------------------------------------

    @property
    def W_pos(self):
        """Expose stream_a W_pos for Trainer safety valve."""
        return self.stream_a.W_pos

    @property
    def W_phase(self):
        return None

    def tick_epoch(self):
        self.stream_a.tick_epoch()
        self.stream_b.tick_epoch()

    # -- Forward ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stream A: seed → route
        Z_a = self.stream_a.m.base._seed(x)
        Z_a = _route_layer(self.stream_a, Z_a)

        # Stream B: its own seed → route
        Z_b = self.stream_b.m.base._seed(x)
        Z_b = _route_layer(self.stream_b, Z_b)

        # Fusion
        if self.fusion_mode == "sum":
            Z_fused = F.normalize(Z_a + Z_b, dim=-1)
        else:  # concat
            Z_cat   = torch.cat([Z_a, Z_b], dim=-1)   # [B, N, 2D]
            Z_fused = F.normalize(self.proj(Z_cat), dim=-1)

        # Readout from stream_a
        return self.stream_a.m.base._readout(Z_fused)
