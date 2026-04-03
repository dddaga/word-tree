"""SGNNET_ProximityWave: sparse k-NN topology + dynamic phasor routing.

Unifies the O(N*K) scalability of fixed fan-in with the dynamic phase
modulation of phasor proximity routing.

Key idea
--------
Instead of computing all NxN distances every forward pass (O(N**2)),
we maintain a sparse neighbour table conn_hh [N, K] and compute only
the K distances per neuron needed for phase and strength:

    d_hk  = ||W_pos[h] - W_pos[conn_hh[h,k]]||        O(N*K)
    s_hk  = exp(-d**2/r***2)                            strength gate
    phi_hk = 2*pi * d / lambda                          phase shift
    Z_new[h] = sum_k  s_hk * (Z_re[k]*cos(phi) - Z_im[k]*sin(phi))

This is 1667x fewer distance ops than full cdist at N=10000, K=6.

Topology update
---------------
conn_hh is rebuilt from W_pos k-NN every `reconnect_every` epochs.
Between rebuilds the topology is fixed (stable gradient flow).
As W_pos trains, neurons drift to better positions and the graph
rewires to match -- adaptive topology without per-batch O(N**2) cost.

No hard groups
--------------
conn_hh is built purely from W_pos geometry: k nearest neighbours
in the learned position space. "Groups" emerge from where neurons
cluster -- no index-based boundaries, so overlap is automatic.
K_random connections are sampled globally for long-range shortcuts.

Anti-Hebbian suppression (optional)
------------------------------------
When anti_hebb_alpha > 0, phasor routing weights are modulated by
position-similarity suppression: similar neighbours contribute less.
This decorrelates representations (Mexican-hat profile) without
requiring a separate wrapper -- keeps phasor routing pure O(N*K).
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import compute_spatial_encoding, compute_fourier_encoding
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

    K_local nearest neighbours in W_pos space (geometry-based, no groups).
    K_random global random shortcuts (Watts-Strogatz long-range links).

    Called at init and periodically during training when reconnect_every > 0.
    Cost: O(N**2) for the k-NN search -- done once per reconnect period,
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

        # Min-distance guard: nearly coincident neighbours cause 1/d explosion
        r_star = personal_volume_radius(N, W_pos.shape[1], 1.0)
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
# Sparse phasor routing (O(N*K) per forward pass)
# -------------------------------------------------------------------

def sparse_phasor_route(
    Z_re: torch.Tensor,
    Z_im: torch.Tensor,
    W_pos: torch.Tensor,
    conn_hh: torch.Tensor,
    N_hidden: int,
    D: int,
    box_size: float,
    anti_hebb_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phasor routing using only the K precomputed neighbours per neuron.

    Parameters
    ----------
    Z_re, Z_im      : [B, N, D]   current phasor activations
    W_pos            : [N, D]      neuron positions (learned)
    conn_hh          : [N, K]      neighbour index table
    N_hidden, D, box_size          for r* computation
    anti_hebb_weights: [N, K]      optional suppression weights (1 - alpha*sim)

    Returns
    -------
    Z_out_re, Z_out_im : [B, N, D]
    """
    N, K = conn_hh.shape

    # Neighbour positions: [N, K, D]
    W_nbr = W_pos[conn_hh]

    # Per-edge distances: ||W_pos[h] - W_pos[k]||  ->  [N, K]
    d = (W_pos.unsqueeze(1) - W_nbr).norm(dim=-1)

    # Gaussian strength + hard gate at r*
    r_star = personal_volume_radius(N_hidden, D, box_size)
    lam    = r_star / 2.0
    s = torch.exp(-d ** 2 / (r_star ** 2 + 1e-8)) * (d < r_star).float()

    # Apply anti-Hebbian suppression if provided
    if anti_hebb_weights is not None:
        s = s * anti_hebb_weights  # [N, K]

    # Normalise per target neuron (sum over K source connections)
    s = s / (s.sum(dim=1, keepdim=True) + 1e-8)

    # Phase rotation per edge: phi = 2*pi*d/lambda  ->  [N, K]
    phi   = 2.0 * math.pi * d / (lam + 1e-8)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)

    # Weighted phasor: gather source activations, apply strength*phase
    Z_re_src = Z_re[:, conn_hh, :]   # [B, N, K, D]
    Z_im_src = Z_im[:, conn_hh, :]

    # s, cos_p, sin_p: [N, K] -> broadcast as [1, N, K, 1]
    w_cos = (s * cos_p).unsqueeze(0).unsqueeze(-1)
    w_sin = (s * sin_p).unsqueeze(0).unsqueeze(-1)

    Z_out_re = (w_cos * Z_re_src - w_sin * Z_im_src).sum(dim=2)
    Z_out_im = (w_sin * Z_re_src + w_cos * Z_im_src).sum(dim=2)

    return Z_out_re, Z_out_im


# -------------------------------------------------------------------
# SGNNET_ProximityWave
# -------------------------------------------------------------------

class SGNNET_ProximityWave(nn.Module):
    """Sparse k-NN topology + dynamic phasor routing network.

    Parameters
    ----------
    N_hidden        : hidden neurons (tested to 20 000)
    D               : geometric dimensionality (4 or 64)
    K_local         : k-NN connections per neuron (geometry-based)
    K_random        : long-range random shortcuts (>=2 for connectivity)
    reconnect_every : rebuild conn_hh from W_pos every N epochs (0=never)
    K_in            : fixed fan-in from input per hidden neuron
    K_iter          : routing iterations
    encoding_mode   : 'linear' (D=4 only) or 'fourier' (any D>=4)
    norm_mode       : 'masked', 'l2', or 'relu'
    anti_hebb_alpha : anti-Hebbian suppression strength (0=disabled)
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
        encoding_mode: str = "linear",
        norm_mode: str = "masked",
        anti_hebb_alpha: float = 0.0,
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
        self.norm_mode = norm_mode
        self.anti_hebb_alpha = anti_hebb_alpha
        self._epoch = 0
        self._topology_changes = 0   # count of topology rebuilds

        # Learnable positions
        self.W_pos = nn.Parameter(torch.rand(N_hidden + N_out, D) * box_size)
        self.W_phase = None   # compatibility shim

        # Spatial encoding
        self.encoding_mode = encoding_mode
        if encoding_mode == "fourier":
            spatial = compute_fourier_encoding(N_in, D=D)
        else:
            spatial = compute_spatial_encoding(N_in)  # [N_in, 3], D=4 only
        self.register_buffer("spatial_coords", spatial)

        # Input fan-in: block-local
        from .model_smallworld import _build_fanin_conn
        n_groups = max(8, N_hidden // 8)
        conn_in = _build_fanin_conn(N_hidden, N_in, K_in, n_groups)
        self.register_buffer("conn_in", conn_in)

        # Hidden topology: k-NN on initial W_pos + random shortcuts
        conn_hh = build_knn_conn(self.W_pos[:N_hidden], K_local, K_random)
        self.register_buffer("conn_hh", conn_hh)

        # Output mask (small -- N_out=10, stays dense)
        self.register_buffer("C_ho_mask", _make_binary_c(N_hidden, N_out, sparsity))

    # ---------------------------------------------------------------

    def tick_epoch(self):
        """Call once per epoch from training loop to trigger reconnection."""
        self._epoch += 1
        if self.reconnect_every > 0 and self._epoch % self.reconnect_every == 0:
            old_conn = self.conn_hh.clone()
            new_conn = build_knn_conn(
                self.W_pos[:self.N_hidden], self.K_local, self.K_random,
                seed=self._epoch,
            )
            changed = (old_conn != new_conn).any(dim=1).sum().item()
            self.conn_hh.copy_(new_conn)
            self._topology_changes += 1
            print(f"    [tick_epoch {self._epoch}] topology rebuild #{self._topology_changes}"
                  f"  changed_rows={changed}/{self.N_hidden}")

    def _normalise(self, Z: torch.Tensor) -> torch.Tensor:
        """Apply normalisation according to self.norm_mode."""
        if self.norm_mode == "l2":
            return F.normalize(Z, dim=-1)
        if self.norm_mode == "relu":
            return F.normalize(F.relu(Z), dim=-1)
        return masked_normalize(Z)

    def _compute_anti_hebb_weights(self) -> torch.Tensor | None:
        """Pre-compute anti-Hebbian suppression weights from W_pos similarity.

        Returns [N_hidden, K] weights where similar neighbours are suppressed.
        Static (depends on W_pos, not Z), so computed once before routing loop.
        """
        if self.anti_hebb_alpha <= 0.0:
            return None
        W_h = F.normalize(self.W_pos[:self.N_hidden], dim=-1)  # [N, D]
        W_nbr = W_h[self.conn_hh]                              # [N, K, D]
        sim = (W_h.unsqueeze(1) * W_nbr).sum(dim=-1)           # [N, K]
        return (1.0 - self.anti_hebb_alpha * sim.clamp(min=0))  # [N, K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z_re, Z_im = self._seed(x)
        Z_re, Z_im = self._route(Z_re, Z_im)
        return self._readout(Z_re, Z_im)

    def _seed(self, x):
        B = x.shape[0]
        spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        A = torch.cat([x.unsqueeze(-1), spatial], dim=-1)   # [B, N_in, D]
        Z_re = self._normalise(A[:, self.conn_in, :].sum(dim=2))
        return Z_re, torch.zeros_like(Z_re)

    def _route(self, Z_re, Z_im):
        W_h = self.W_pos[:self.N_hidden]
        ahw = self._compute_anti_hebb_weights()
        for _ in range(self.K_iter):
            Z_re, Z_im = sparse_phasor_route(
                Z_re, Z_im, W_h, self.conn_hh,
                self.N_hidden, self.D, self.box_size,
                anti_hebb_weights=ahw,
            )
            Z_re = self._normalise(Z_re)
            Z_im = self._normalise(Z_im)
        return Z_re, Z_im

    def _readout(self, Z_re, Z_im):
        C_ho = self.C_ho_mask.float()
        S_re = torch.einsum("bhd,ho->bod", Z_re, C_ho)
        S_im = torch.einsum("bhd,ho->bod", Z_im, C_ho)
        A_out = torch.sqrt(S_re ** 2 + S_im ** 2 + 1e-8)
        W_out = F.normalize(self.W_pos[self.N_hidden:], dim=-1)
        return (A_out * W_out.unsqueeze(0)).sum(dim=-1)
