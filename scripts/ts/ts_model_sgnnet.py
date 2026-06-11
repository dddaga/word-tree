"""SGNNET-TS recurrent graph model for time-series prediction.
# See ts_common.py (data/losses), ts_step030_sgnnet_ts.py (main)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeedProjection(nn.Module):
    """Linear(F → D) per stock, output normalized to unit hypersphere."""
    def __init__(self, F: int, D: int):
        super().__init__()
        self.proj = nn.Linear(F, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class AntiHebbianRouter(nn.Module):
    """Anti-Hebbian message passing: msg(i←j) = W_msg@Z[j] - W_anti@Z[i]."""
    def __init__(self, D: int, dropout: float = 0.1):
        super().__init__()
        self.W_msg = nn.Linear(D, D, bias=False)
        self.W_anti = nn.Linear(D, D, bias=False)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, D = Z.shape
        src = edge_index[0]; tgt = edge_index[1]

        msg = self.W_msg(Z[:, src, :]) - self.W_anti(Z[:, tgt, :])

        agg = torch.zeros(B, N, D, device=Z.device, dtype=Z.dtype)
        agg.scatter_add_(1, tgt.view(1, -1, 1).expand(B, -1, D), msg)

        count = torch.zeros(N, device=Z.device, dtype=Z.dtype)
        count.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=Z.device, dtype=Z.dtype))
        agg = agg / count.clamp(min=1.0).view(1, N, 1)

        return self.norm(Z + self.drop(agg))


class SGNNETRecurrent(nn.Module):
    """SGNNET-TS: recurrent GNN for stock return prediction.

    At each timestep t:
        x_emb = seed_proj(X[:, t, :, :])          # [B, N, D]
        inp   = normalize(x_emb + beta * Z)        # additive feedback (no gate — gate-death safe)
        Z_r   = K_iter rounds of anti-Hebbian MP
        Z     = alpha*Z + (1-alpha)*Z_r             # EMA hidden state update

    PredictionHead: Linear(N*D → N).
    alpha, beta: learned scalar nn.Parameters.
    """

    def __init__(self, N: int, F: int = 16, D: int = 16,
                 K_wiring: int = 4, K_iter: int = 3, dropout: float = 0.1):
        super().__init__()
        self.N = N; self.D = D; self.K_iter = K_iter

        self.seed_proj = SeedProjection(F, D)
        self.router = AntiHebbianRouter(D, dropout=dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.head = nn.Linear(N * D, N)

        self.register_buffer("edge_index", self._build_ring_graph(N, K_wiring))

    @staticmethod
    def _build_ring_graph(N: int, K: int) -> torch.Tensor:
        if K % 2 != 0:
            raise ValueError(f"K_wiring must be even, got {K}")
        half = K // 2
        offsets = list(range(-half, 0)) + list(range(1, half + 1))
        src_list, tgt_list = [], []
        for i in range(N):
            for off in offsets:
                j = (i + off) % N
                src_list.append(i); tgt_list.append(j)
        return torch.stack([
            torch.tensor(src_list, dtype=torch.long),
            torch.tensor(tgt_list, dtype=torch.long)
        ], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, F] → [B, N]"""
        B, T, N, _ = x.shape
        alpha = self.alpha.clamp(0.0, 1.0)
        beta = self.beta
        Z = torch.zeros(B, N, self.D, device=x.device, dtype=x.dtype)

        for t in range(T):
            x_emb = self.seed_proj(x[:, t, :, :])
            inp = F.normalize(x_emb + beta * Z, dim=-1)
            Z_r = inp
            for _ in range(self.K_iter):
                Z_r = self.router(Z_r, self.edge_index)
            Z = alpha * Z + (1.0 - alpha) * Z_r

        return torch.tanh(self.head(Z.reshape(B, N * self.D))) * 0.05
