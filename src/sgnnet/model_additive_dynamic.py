"""SGNNET_AdditiveDynamic: wrapper adding r*-threshold dynamic connectivity.

Tests brief §3.5 additive dynamic connectivity:
    Z = normalize(Z_static + alpha_dyn * Z_dyn)
where Z_static = static small-world gather and Z_dyn = dynamic_connectivity_hh
using cdist(Z, W_pos) with Gaussian kernel + hard r* gate.

Key distinction from gate-death experiments:
  - MULTIPLICATIVE gating: Z = gate * Z  → gate-death theorem applies (K_iter≥4 → g^K→0)
  - ADDITIVE contribution: Z = normalize(Z_static + alpha * Z_dyn)  → no gate-death risk
    The dynamic term adds to static; never suppresses it multiplicatively.

COST: O(N²) per iteration per batch (cdist). Use N=512, not N=2048.

Trainer compatibility: exposes model.W_pos and model.W_phase shims so the
standard Trainer can update W_pos without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld
from .geometry import dynamic_connectivity_hh


class SGNNET_AdditiveDynamic(nn.Module):
    """Wraps SGNNET_SmallWorld with additive r*-threshold dynamic connectivity.

    Parameters
    ----------
    base       : SGNNET_SmallWorld instance (provides topology + readout)
    alpha_dyn  : float, scaling weight for dynamic contribution (default 1.0)
    learn_alpha: bool, if True alpha_dyn is a learned nn.Parameter
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        alpha_dyn: float = 1.0,
        learn_alpha: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.learn_alpha = learn_alpha

        if learn_alpha:
            self._alpha_dyn = nn.Parameter(torch.tensor(alpha_dyn))
        else:
            self._alpha_dyn = alpha_dyn  # plain float, not a Parameter

        # Trainer compatibility shims: Trainer looks for model.W_pos
        self.W_pos = base.W_pos    # nn.Parameter reference; grads flow to base
        self.W_phase = None        # compatibility only; unused

    @property
    def alpha_dyn(self) -> torch.Tensor | float:
        return self._alpha_dyn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base

        # Step 1: seed — input → hidden [B, N_hidden, D]
        Z = base._seed(x)

        # Step 2: K_iter rounds of additive static + dynamic routing
        W_pos_norm = F.normalize(base.W_pos[:base.N_hidden], dim=-1)  # [N, D]

        for _ in range(base.K_iter):
            # Static gather: fixed small-world topology [B, N, K_hh, D] → [B, N, D]
            Z_static = Z[:, base.conn_hh, :].sum(dim=2)

            # Dynamic contribution: r*-gated Gaussian kernel over cdist
            Z_dyn, _ = dynamic_connectivity_hh(
                Z, W_pos_norm, base.N_hidden, base.D
            )

            # Additive combination + L2 normalise per neuron
            Z = F.normalize(Z_static + self.alpha_dyn * Z_dyn, dim=-1)

        # Step 3: readout
        return base._readout(Z)

    def extra_repr(self) -> str:
        return (
            f"N_hidden={self.base.N_hidden}, D={self.base.D}, "
            f"K_iter={self.base.K_iter}, "
            f"alpha_dyn={self._alpha_dyn!r}, learn_alpha={self.learn_alpha}"
        )


class SGNNET_AH_AdditiveDynamic(nn.Module):
    """Eager canonical chain (theta + reflection + AH wpos) + additive dynamic.

    v2 (2026-06-10): v1 above wraps bare SmallWorld — invalid Ref: bare base
    does not learn (AH prerequisite, step218). v2 reimplements the canonical
    routing step eagerly and adds the dynamic term inside it.
    Ref = alpha_dyn=0 (chain-equivalent). O(N^2) cdist per iter — use N=512.

    Per iteration:
      Z_fwd    = relu(Z - |theta|)
      Z_struct = sum_k supp_w * Z_fwd[conn_hh]           (AH-suppressed gather)
      Z_dyn    = dynamic_connectivity_hh(Z_fwd, W_norm)  (Gaussian, r* gate)
      Z_refl   = alpha_reflect * Z_refl + (Z_fwd - Z)
      Z        = normalize(clamp(Z_struct + alpha_dyn*Z_dyn + Z_refl))
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        alpha_dyn: float = 1.0,
        learn_alpha: bool = False,
        alpha_reflect: float = 0.5,
        alpha_ahebb: float = 1.0,
        theta_init: float = 0.1,
    ):
        super().__init__()
        self.base = base
        self.alpha_reflect = alpha_reflect
        self.alpha_ahebb = alpha_ahebb
        self.learn_alpha = learn_alpha
        if learn_alpha:
            self._alpha_dyn = nn.Parameter(torch.tensor(float(alpha_dyn)))
        else:
            self._alpha_dyn = float(alpha_dyn)
        self.theta = nn.Parameter(torch.full((base.N_hidden,), theta_init))
        self.W_phase = None  # Trainer shim

    @property
    def W_pos(self):
        return self.base.W_pos

    def tick_epoch(self):
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    def _supp_w(self) -> torch.Tensor:
        """AH suppression weights [1, N, K_hh, 1] from W_pos cosine sim."""
        base = self.base
        W_n = F.normalize(base.W_pos[:base.N_hidden], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[base.conn_hh]).sum(-1)
        return (
            1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
        ).unsqueeze(0).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        N, Dd = base.N_hidden, base.D
        Z = base._seed(x)                                    # [B, N, D]
        Z_refl = torch.zeros_like(Z)
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)
        supp_w = self._supp_w()
        alpha = self._alpha_dyn
        W_norm = F.normalize(base.W_pos[:N], dim=-1)         # [N, D]
        use_dyn = self.learn_alpha or (isinstance(alpha, float) and alpha != 0.0)

        for _ in range(base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)                    # [B, N, D]
            Z_nb = Z_fwd[:, base.conn_hh, :]                 # [B, N, K, D]
            Z_struct = (Z_nb * supp_w).sum(dim=2)            # [B, N, D]
            if use_dyn:
                Z_dyn, _ = dynamic_connectivity_hh(Z_fwd, W_norm, N, Dd)
                Z_struct = Z_struct + alpha * Z_dyn
            Z_refl = self.alpha_reflect * Z_refl + (Z_fwd - Z)
            Z = F.normalize((Z_struct + Z_refl).clamp(-10, 10), dim=-1)

        return base._readout(Z)
