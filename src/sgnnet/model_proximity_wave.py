"""SGNNET_ProximityWave: sparse topology + dynamic phase routing.

Unifies the O(N·K) scalability of fixed fan-in with the dynamic phase
modulation of phasor proximity routing.

Key idea
--------
Instead of computing all N×N distances every forward pass (O(N²)),
we maintain a sparse neighbour table conn_hh [N, K] and compute only
the K distances per neuron needed for phase and strength:

    d_hk  = ||W_pos[h] - W_pos[conn_hh[h,k]]||        O(N·K)
    s_hk  = exp(-d²/r*²)                               strength gate
    φ_hk  = 2π · d / λ                                 phase shift
    Z_new[h] = Σ_k  s_hk · (Z_re[k]·cos φ - Z_im[k]·sin φ)

This is 1667× fewer distance ops than full cdist at N=10000, K=6.

Topology update
---------------
conn_hh is rebuilt from W_pos k-NN every `reconnect_every` epochs.
Between rebuilds the topology is fixed (stable gradient flow).
As W_pos trains, neurons drift to better positions and the graph
rewires to match — adaptive topology without per-batch O(N²) cost.

No hard groups
--------------
conn_hh is built purely from W_pos geometry: k nearest neighbours
in the learned 4D position space. "Groups" emerge from where neurons
cluster — no index-based boundaries, so overlap is automatic.
A small fraction K_random of connections are sampled globally (not
from nearest neighbours) to provide long-range shortcuts and ensure
the graph stays well-connected (small-world property).
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import compute_spatial_encoding
from .geometry import personal_volume_radius
from .model_wave import _make_binary_c
from .norm_masked import masked_normalize


# -------------------------------------------------------------------
# Sparse topology builder
# -------------------------------------------------------------------

def build_knn_conn(
    W_pos: torch.Tensor,
    K_local: int,
    K_random: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Build [N, K_local+K_random] connection table from current W_pos.

    K_local nearest neighbours in W_pos space (geometry-based, no hard groups).
    K_random global random shortcuts (Watts-Strogatz long-range links).

    Called at init and periodically during training when reconnect_every > 0.
    Cost: O(N² / chunk) for the k-NN search — done once per reconnect period,
    not every batch.
    """
    N = W_pos.shape[0]
    K = K_local + K_random
    device = W_pos.device

    with torch.no_grad():
        # Pairwise distances on CPU to avoid MPS memory spike
        W_cpu = W_pos.detach().cpu().float()
        dists = torch.cdist(W_cpu, W_cpu)           # [N, N]
        dists.fill_diagonal_(float("inf"))           # exclude self

        # Min-distance guard: neighbours that are nearly coincident (d < r*/4)
        # cause 1/d to explode in phase computation. Mark them as unreachable
        # so k-NN never picks degenerate near-zero edges.
        from .geometry import personal_volume_radius
        r_star = personal_volume_radius(W_pos.shape[0], W_pos.shape[1], 1.0)
        dists[dists < r_star / 4.0] = float("inf")

        _, knn_idx = dists.topk(K_local, dim=1, largest=False)  # [N, K_local]

    rng = np.random.default_rng(seed)
    rand_idx = np.stack([
        rng.choice([j for j in range(N) if j != i], size=K_random, replace=False)
        for i in range(N)
    ])  # [N, K_random]
    rand_idx = torch.tensor(rand_idx, dtype=torch.long)

    conn = torch.cat([knn_idx, rand_idx], dim=1).to(device)  # [N, K]
    return conn


# -------------------------------------------------------------------
# Sparse phasor routing (O(N·K) per forward pass)
# -------------------------------------------------------------------

def sparse_phasor_route(
    Z_re: torch.Tensor,
    Z_im: torch.Tensor,
    W_pos: torch.Tensor,
    conn_hh: torch.Tensor,
    N_hidden: int,
    D: int,
    box_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phasor routing using only the K precomputed neighbours per neuron.

    Parameters
    ----------
    Z_re, Z_im : [B, N, D]   current phasor activations
    W_pos      : [N, D]      neuron positions (learned)
    conn_hh    : [N, K]      neighbour index table
    N_hidden, D, box_size    for r* computation

    Returns
    -------
    Z_out_re, Z_out_im : [B, N, D]
    """
    N, K = conn_hh.shape

    # Neighbour positions: [N, K, D]
    W_nbr = W_pos[conn_hh]                              # gather positions

    # Per-edge distances: ||W_pos[h] - W_pos[k]||  →  [N, K]
    d = (W_pos.unsqueeze(1) - W_nbr).norm(dim=-1)       # [N, K]

    # Gaussian strength + hard gate at r*
    r_star = personal_volume_radius(N_hidden, D, box_size)
    lam    = r_star / 2.0
    s = torch.exp(-d ** 2 / (r_star ** 2 + 1e-8)) * (d < r_star).float()  # [N, K]

    # Normalise per target neuron (sum over K source connections)
    s = s / (s.sum(dim=1, keepdim=True) + 1e-8)         # [N, K]

    # Phase rotation per edge: φ = 2π·d/λ  →  [N, K]
    phi  = 2.0 * math.pi * d / (lam + 1e-8)
    cos_p = torch.cos(phi)                               # [N, K]
    sin_p = torch.sin(phi)                               # [N, K]

    # Weighted phasor: gather source activations, apply strength·phase
    # Z_re_src, Z_im_src: [B, N, K, D]
    Z_re_src = Z_re[:, conn_hh, :]
    Z_im_src = Z_im[:, conn_hh, :]

    # s, cos_p, sin_p: [N, K] → broadcast as [1, N, K, 1]
    w_cos = (s * cos_p).unsqueeze(0).unsqueeze(-1)       # [1, N, K, 1]
    w_sin = (s * sin_p).unsqueeze(0).unsqueeze(-1)

    Z_out_re = (w_cos * Z_re_src - w_sin * Z_im_src).sum(dim=2)  # [B, N, D]
    Z_out_im = (w_sin * Z_re_src + w_cos * Z_im_src).sum(dim=2)

    return Z_out_re, Z_out_im


# -------------------------------------------------------------------
# SGNNET_ProximityWave
# -------------------------------------------------------------------

class SGNNET_ProximityWave(nn.Module):
    """Sparse + dynamic phasor routing network.

    Parameters
    ----------
    N_hidden        : hidden neurons (tested to 20 000)
    K_local         : k-NN connections per neuron (geometry-based)
    K_random        : long-range random shortcuts (≥2 for connectivity)
    reconnect_every : rebuild conn_hh from W_pos every N epochs (0 = never)
    K_in            : fixed fan-in from input per hidden neuron
    K_iter          : routing iterations
    """

    def __init__(
        self,
        N_hidden: int = 1024,
        N_out: int = 10,
        D: int = 4,
        N_in: int = 25088,
        K_local: int = 6,
        K_random: int = 2,
        reconnect_every: int = 10,
        K_in: int = 50,
        K_iter: int = 3,
        sparsity: float = 0.90,
        box_size: float = 1.0,
    ):
        super().__init__()
        self.N_hidden = N_hidden
        self.N_out = N_out
        self.D = D
        self.N_in = N_in
        self.K_iter = K_iter
        self.box_size = box_size
        self.reconnect_every = reconnect_every
        self.K_local = K_local
        self.K_random = K_random
        self._epoch = 0   # incremented by Trainer via .tick_epoch()

        # Learnable positions — the only learned parameters
        self.W_pos = nn.Parameter(torch.rand(N_hidden + N_out, D) * box_size)
        self.W_phase = None   # compatibility shim

        self.register_buffer("spatial_coords", compute_spatial_encoding(N_in))

        # Input fan-in: block-local — rebuilt once at init (input doesn't move)
        from .model_smallworld import _build_fanin_conn
        n_groups = max(8, N_hidden // 8)
        conn_in = _build_fanin_conn(N_hidden, N_in, K_in, n_groups)
        self.register_buffer("conn_in", conn_in)

        # Hidden topology: k-NN on initial W_pos + random shortcuts
        conn_hh = build_knn_conn(self.W_pos[:N_hidden], K_local, K_random)
        self.register_buffer("conn_hh", conn_hh)

        # Output mask (small — N_out=10, stays dense)
        self.register_buffer("C_ho_mask", _make_binary_c(N_hidden, N_out, sparsity))

    # ---------------------------------------------------------------

    def tick_epoch(self):
        """Call once per epoch from training loop to trigger reconnection."""
        self._epoch += 1
        if self.reconnect_every > 0 and self._epoch % self.reconnect_every == 0:
            new_conn = build_knn_conn(
                self.W_pos[:self.N_hidden], self.K_local, self.K_random,
                seed=self._epoch,
            )
            self.conn_hh.copy_(new_conn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z_re, Z_im = self._seed(x)
        Z_re, Z_im = self._route(Z_re, Z_im)
        return self._readout(Z_re, Z_im)

    def _seed(self, x):
        B = x.shape[0]
        spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        A = torch.cat([x.unsqueeze(-1), spatial], dim=-1)   # [B, N_in, D]
        Z_re = masked_normalize(A[:, self.conn_in, :].sum(dim=2))
        return Z_re, torch.zeros_like(Z_re)

    def _route(self, Z_re, Z_im):
        W_h = self.W_pos[:self.N_hidden]
        for _ in range(self.K_iter):
            Z_re, Z_im = sparse_phasor_route(
                Z_re, Z_im, W_h, self.conn_hh,
                self.N_hidden, self.D, self.box_size,
            )
            Z_re = masked_normalize(Z_re)
            Z_im = masked_normalize(Z_im)
        return Z_re, Z_im

    def _readout(self, Z_re, Z_im):
        C_ho = self.C_ho_mask.float()
        S_re = torch.einsum("bhd,ho->bod", Z_re, C_ho)
        S_im = torch.einsum("bhd,ho->bod", Z_im, C_ho)
        A_out = torch.sqrt(S_re ** 2 + S_im ** 2 + 1e-8)
        W_out = F.normalize(self.W_pos[self.N_hidden:], dim=-1)
        return (A_out * W_out.unsqueeze(0)).sum(dim=-1)
