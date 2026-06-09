"""SGNNET_PerNeighborGate: pure per-neighbor softmax routing (step988).

Mechanism:
  w_n[j] ∈ R^D  — learned phase direction for each neighbor j, L2-normalised.
  For each neuron i, given its K_hh neighbors' activations Z_nb[b,i,k,:]:

    s[b,i,k]    = dot(Z_nb[b,i,k], w_n_norm[conn_hh[i,k]])   # [B, N, K_hh]
    gate[b,i,k] = softmax(s[b,i,:], dim=-1)                   # [B, N, K_hh]
    Z_agg[b,i]  = sum_k gate[b,i,k] * Z_nb[b,i,k,:]          # [B, N, D]

  Then L2-normalise Z_agg.

Key properties:
  - No dead zones: softmax always sums to 1
  - No randomness
  - Replaces ΔW-proj entirely (not compounded on same path)
  - w_n[j] is the phase direction OF neighbor j — gate_j measures how
    much neighbor j's current state aligns with its own phase

Variants (all in make_configs):
  A_softmax_gate    — pure per-neighbor softmax (above)
  B_softmax_gate_dw — compound: softmax(s) * |ΔW-proj|, then renorm
                      (same signal path — expect noise per compounding rule)
  C_temperature     — per-neighbor softmax with learned temperature tau

Compounding note: A replaces ΔW-proj. B compounds on same path — may cancel.

See: scripts/train_step988_perneighbor_gate_t0.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld


class SGNNET_PerNeighborGate(nn.Module):
    """Per-neighbor softmax gating replacing ΔW-proj aggregation.

    Parameters
    ----------
    base          : SGNNET_SmallWorld backbone
    alpha_reflect : self-inhibition factor (default 0.5, matches step199)
    theta_init    : initial per-neuron threshold
    dw_compound   : if True, multiply softmax gate by |ΔW-proj| weight and renorm
                    (creates compound B_softmax_gate_dw — same signal path)
    temperature   : if not None, use learned temperature (scalar nn.Parameter)
                    initialized to this value; otherwise temperature=1.0 fixed
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        alpha_reflect: float = 0.5,
        theta_init: float = 0.1,
        dw_compound: bool = False,
        temperature: float | None = None,
    ):
        super().__init__()
        self.base          = base
        self.alpha_reflect = alpha_reflect
        self.dw_compound   = dw_compound
        self.W_phase       = None  # Trainer compatibility shim

        N = base.N_hidden
        D = base.W_pos.shape[1]

        # Per-neuron threshold (shared with step985/Ref convention)
        self.theta = nn.Parameter(torch.full((N,), theta_init))

        # Per-neighbor phase direction [N, D]
        # w_n[j] is the phase direction of neuron j (as a neighbor)
        self.w_n = nn.Parameter(torch.rand(N, D))

        # Optional learned temperature (C_temperature variant)
        if temperature is not None:
            self.tau = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.tau = None

        # Pre-compute ΔW-proj if needed for B variant
        # Stored as buffer so it moves with model.to(device)
        if dw_compound:
            with torch.no_grad():
                W_h = base.W_pos[:N]
                dw  = F.normalize(
                    W_h.unsqueeze(1) - W_h[base.conn_hh], dim=-1
                ).unsqueeze(0)  # [1, N, K_hh, D]
            self.register_buffer("dw_proj", dw)
        else:
            self.dw_proj = None

    # ------------------------------------------------------------------
    # Trainer compatibility shims
    # ------------------------------------------------------------------

    @property
    def W_pos(self):
        return self.base.W_pos

    def tick_epoch(self):
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    # ------------------------------------------------------------------
    # Per-neighbor softmax gate aggregation
    # ------------------------------------------------------------------

    def _softmax_gate_agg(
        self, Z: torch.Tensor, debug: bool = False
    ) -> torch.Tensor:
        """Compute softmax-gated aggregation over K_hh neighbors.

        Args:
            Z     : [B, N, D] L2-normalised activations
            debug : if True, print gate stats (entropy, mean, std)

        Returns:
            Z_agg : [B, N, D] weighted aggregate (un-normalised)
        """
        conn_hh  = self.base.conn_hh                            # [N, K_hh]
        assert conn_hh.shape[1] == 2, (
            f"SGNNET_PerNeighborGate requires K_hh=2, got K_hh={conn_hh.shape[1]}"
        )
        K_hh     = conn_hh.shape[1]

        # L2-normalise w_n at forward time
        w_n_norm = F.normalize(self.w_n, dim=-1)                # [N, D]

        # Gather neighbor activations
        Z_nb = Z[:, conn_hh, :]                                 # [B, N, K_hh, D]

        # Per-neighbor score: Z[b,i,k] · w_n_norm[conn_hh[i,k]]
        # w_n_norm[conn_hh] → [N, K_hh, D] (phase dir of each neighbor)
        neighbor_dirs = w_n_norm[conn_hh]                       # [N, K_hh, D]
        s = (Z_nb * neighbor_dirs.unsqueeze(0)).sum(-1)         # [B, N, K_hh]

        # Apply temperature if present
        if self.tau is not None:
            tau = self.tau.abs().clamp(min=1e-4)
            s   = s / tau

        # Softmax over K_hh dim — no dead zones
        gate = F.softmax(s, dim=-1)                             # [B, N, K_hh]

        # Optional compound with |ΔW-proj| weight (B variant)
        if self.dw_compound and self.dw_proj is not None:
            dw_coeff = (Z_nb * self.dw_proj).sum(-1).abs()      # [B, N, K_hh]
            gate     = gate * dw_coeff
            gate_sum = gate.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            gate     = gate / gate_sum

        if debug:
            # Entropy of gate distribution (max = log(K_hh))
            eps     = 1e-8
            entropy = -(gate * (gate + eps).log()).sum(-1)       # [B, N]
            print(
                f"  [PerNeighborGate debug]  "
                f"gate.mean={gate.mean().item():.4f}  "
                f"gate.std={gate.std().item():.4f}  "
                f"entropy.mean={entropy.mean().item():.4f}  "
                f"entropy.max_possible={torch.tensor(K_hh, dtype=torch.float).log().item():.4f}  "
                f"mean|gate0-0.5|={(gate[...,0]-0.5).abs().mean().item():.4f}"
            )

        Z_agg = (gate.unsqueeze(-1) * Z_nb).sum(dim=2)         # [B, N, D]
        return Z_agg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        Z = self.base._seed(x)                                  # [B, N, D]

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1) # [1, N, 1]
        Z_ref     = torch.zeros_like(Z)
        _debug_done = False

        for _ in range(self.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)

            # Per-neighbor softmax gate aggregation (replaces ΔW-proj)
            _print = debug and not _debug_done
            Z_agg = self._softmax_gate_agg(Z_fwd, debug=_print)
            _debug_done = _debug_done or _print

            # Self-inhibition reflection
            Z_ref = self.alpha_reflect * Z_ref + (Z_fwd - Z)

            # Combine and L2-normalise
            Z = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.base._readout(Z)
