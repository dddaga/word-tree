"""SGNNET_HubInterneuron: high fan-in mixing hub neurons.

Design
------
N_mix = N - n_input neurons act as mixing hubs:
  - Seeded to zero each forward (blank start)
  - Receive via conn_mix [N_mix, fan_in]: high fan-in from random input neurons
  - Broadcast back through existing conn_hh (regular neurons have hubs as neighbours)
  - NOT constrained to be inhibitory — pure mixing/aggregation

This tests whether a global mixing layer (CLS-token / global-node style) helps
the network form long-range representations beyond what the small-world graph provides.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.sgnnet.model_resonant import SGNNET_Resonant


class SGNNET_HubInterneuron(nn.Module):
    def __init__(self, base_model: SGNNET_Resonant, n_input: int,
                 fan_in: int = 512, ah_alpha: float = 0.0, seed: int = 42):
        super().__init__()
        self.m        = base_model
        self.n_input  = n_input
        self.ah_alpha = ah_alpha

        N     = base_model.base.N_hidden
        N_mix = N - n_input
        assert N_mix > 0, f"n_input={n_input} must be < N={N}"

        # conn_mix [N_mix, fan_in]: each hub randomly samples fan_in input neurons
        rng      = np.random.default_rng(seed)
        fan_in_  = min(fan_in, n_input)
        conn_mix = np.array([
            rng.choice(n_input, size=fan_in_, replace=False)
            for _ in range(N_mix)
        ], dtype=np.int64)
        self.register_buffer("conn_mix", torch.tensor(conn_mix, dtype=torch.long))
        # [N_mix, fan_in_]

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)                      # [B, N, D]
        Z[:, self.n_input:, :] = 0.0                  # hubs start blank

        B, N, D   = Z.shape
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh

        # AntiHebb suppression weights (if enabled)
        if self.ah_alpha > 0:
            N_h    = self.m.base.N_hidden
            W_n    = F.normalize(self.m.W_pos[:N_h], dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)        # [N, K_hh]
            supp_w  = (1.0 - self.ah_alpha * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)                      # [1,N,K_hh,1]

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)                              # [B, N, D]

            # Structural routing for all neurons
            Z_nb     = Z_fwd[:, conn_hh, :]                            # [B, N, K_hh, D]
            if self.ah_alpha > 0:
                Z_struct = (Z_nb * supp_w).sum(2)
            else:
                Z_struct = Z_nb.sum(2)                                 # [B, N, D]

            # Hub neurons ADDITIONALLY receive from conn_mix (high fan-in)
            # conn_mix [N_mix, fan_in] indexes into the first n_input neurons
            Z_hub_in = Z_fwd[:, self.conn_mix, :].sum(2)              # [B, N_mix, D]
            Z_struct[:, self.n_input:, :] = (
                Z_struct[:, self.n_input:, :] + Z_hub_in
            )

            Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z_new = Z_struct + self.m.alpha_turing * Z_inh
            Z     = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)
