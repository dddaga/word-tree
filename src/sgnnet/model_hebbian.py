"""HebbianRewirer: epoch-boundary Hebbian prune-and-grow dynamic topology.

Tests brief §9: edges carrying high ΔW-proj alignment stay; low-alignment
edges get pruned each epoch. Safe pattern — epoch-boundary outer-loop update
mirrors AH (CONFIRMED load-bearing since step218). In-forward multiplicative
gating = gate-death (KILLED steps 873–916); epoch-boundary = safe.

v2 (2026-06-10): controller object, NOT an nn.Module forward wrapper.
v1 wrapped SmallWorld.forward — bypassed by the canonical AH chain, so edge
scores never accumulated AND the bare base never learned (step991 v1 invalid,
Ref=14%). v2 scores edges via a diagnostic batch run through base-only routing
under no_grad at each epoch boundary, then prunes/grows base.conn_hh in place.

Edge score: mean over K_iter of |c_ij| = |dot(Z_nb, normalize(W[j]-W[i]))|
High |c_ij| → strong information flow along geometric path → keep.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld


class HebbianRewirer:
    """Epoch-boundary prune-and-grow on base.conn_hh, scored by ΔW-proj.

    Parameters
    ----------
    base       : SGNNET_SmallWorld whose conn_hh gets rewired in place
    prune_frac : fraction of edges pruned (and regrown) per call
    grow_mode  : "random" — random new neighbors (not self, not duplicate)
                 "wpos"   — nearest W_pos neighbors not already connected
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        prune_frac: float = 0.10,
        grow_mode: str = "random",
    ):
        self.base = base
        self.prune_frac = prune_frac
        self.grow_mode = grow_mode
        self.n_rewired_total = 0

    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """ΔW-proj edge scores from a diagnostic batch. Returns [N, K_hh]."""
        base = self.base
        N = base.N_hidden
        Z = base._seed(x)                                   # [B, N, D]
        W_n = F.normalize(base.W_pos[:N], dim=-1)           # [N, D]
        dW = F.normalize(W_n[base.conn_hh] - W_n.unsqueeze(1), dim=-1)  # [N,K,D]
        score = torch.zeros(
            base.conn_hh.shape, dtype=torch.float32, device=Z.device
        )
        for _ in range(base.K_iter):
            Z_nb = Z[:, base.conn_hh, :]                    # [B, N, K, D]
            score += (Z_nb * dW.unsqueeze(0)).sum(-1).abs().mean(0)
            Z = F.normalize(Z_nb.sum(dim=2), dim=-1)
        return score / base.K_iter

    # ------------------------------------------------------------------

    @torch.no_grad()
    def rewire(self, x: torch.Tensor) -> int:
        """Prune bottom prune_frac edges by score, grow replacements.
        Returns number of edges rewired."""
        scores = self.score(x)
        N, K_hh = self.base.conn_hh.shape
        n_prune = max(1, int(N * K_hh * self.prune_frac))

        _, bottom = scores.flatten().topk(n_prune, largest=False)
        row_idx = (bottom // K_hh).cpu()
        col_idx = (bottom % K_hh).cpu()

        conn = self.base.conn_hh.clone()
        if self.grow_mode == "wpos":
            self._grow_wpos(conn, row_idx, col_idx, N)
        else:
            self._grow_random(conn, row_idx, col_idx, N)
        self.base.conn_hh.copy_(conn)
        self.n_rewired_total += n_prune
        return n_prune

    # ------------------------------------------------------------------

    def _grow_random(self, conn, row_idx, col_idx, N: int) -> None:
        g = torch.Generator().manual_seed(self.n_rewired_total)
        for r, c in zip(row_idx.tolist(), col_idx.tolist()):
            existing = set(conn[r].tolist()) | {r}
            for nb in torch.randint(0, N, (32,), generator=g).tolist():
                if nb not in existing:
                    conn[r, c] = nb
                    break
            else:
                for nb in range(N):
                    if nb not in existing:
                        conn[r, c] = nb
                        break

    def _grow_wpos(self, conn, row_idx, col_idx, N: int) -> None:
        W = self.base.W_pos[:N].detach()
        dist = torch.cdist(W, W)
        dist.fill_diagonal_(float("inf"))
        for r, c in zip(row_idx.tolist(), col_idx.tolist()):
            existing = set(conn[r].tolist()) | {r}
            for nb in dist[r].argsort()[:16].tolist():
                if nb not in existing:
                    conn[r, c] = nb
                    break
