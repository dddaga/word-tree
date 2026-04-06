"""Inhibitory routing mechanism wrappers for SGNNET_Resonant.

Three lateral inhibition mechanisms for step16 experiments.
Each wraps a trained SGNNET_Resonant, overriding only the routing loop.

SGNNET_DivisiveNorm   — lateral shunting: structural contribution scaled by
                        neighbourhood activity (Heeger 1992 V1 normalization).
SGNNET_Refractory     — temporal suppression: recently-fired neurons are
                        exponentially dampened the next routing step.
SGNNET_AntiHebbian    — decorrelation: structurally similar neurons suppress
                        each other (Mexican-hat profile).
                        variant='wpos': similarity in W_pos direction space (static)
                        variant='zact': similarity in current Z space (dynamic)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_resonant import SGNNET_Resonant


class SGNNET_DivisiveNorm(nn.Module):
    """Lateral shunting: structural contribution scaled by neighbourhood activity.

    Z_struct = Z_nb.sum(2) / (1 + alpha_div * mean_nb_magnitude)

    Prevents a single dominant neighbourhood from monopolising routing.
    """
    def __init__(self, base: SGNNET_Resonant, alpha_div: float = 1.0):
        super().__init__()
        self.m         = base
        self.alpha_div = alpha_div

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]                          # [B,N,K_hh,D]
            nb_mag   = Z_nb.norm(dim=-1).mean(dim=2, keepdim=True)   # [B,N,1]
            Z_struct = Z_nb.sum(dim=2) / (1.0 + self.alpha_div * nb_mag)
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z = F.normalize((Z_struct + self.m.alpha_turing * Z_inh).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


class SGNNET_Refractory(nn.Module):
    """Refractory inhibition: recently active neurons are suppressed next step.

    Refractory trace r (EMA of per-neuron activity per routing step):
        r = beta * r + (1-beta) * ||Z||
    Suppression applied before excitatory gate:
        Z_available = Z * exp(-alpha_refract * r)

    Forces information to spread — no neuron can dominate two consecutive steps.
    """
    def __init__(self, base: SGNNET_Resonant, beta: float = 0.9, alpha_refract: float = 1.0):
        super().__init__()
        self.m             = base
        self.beta          = beta
        self.alpha_refract = alpha_refract

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        B, N_h, _ = Z.shape
        r = torch.zeros(B, N_h, device=Z.device)   # refractory trace [B,N]

        for _ in range(self.m.base.K_iter):
            suppression  = torch.exp(-self.alpha_refract * r).unsqueeze(-1)   # [B,N,1]
            Z_avail      = Z * suppression
            Z_fwd        = F.relu(Z_avail - theta_pos)
            Z_struct     = Z_fwd[:, conn_hh, :].sum(dim=2)
            Z_inh        = self.m._phase_inhibit(Z_avail, W_ph_norm, theta_pos)
            Z_new = F.normalize((Z_struct + self.m.alpha_turing * Z_inh).clamp(-10, 10), dim=-1)
            r = self.beta * r + (1.0 - self.beta) * Z.norm(dim=-1)
            Z = Z_new

        return self.m.base._readout(Z)


class SGNNET_AntiHebbian(nn.Module):
    """Anti-Hebbian lateral inhibition: similar neurons suppress each other.

    suppress(h, k) = 1 - alpha_ahebb * cosine_sim(h, k).clamp(0)
    Z_struct = (Z_nb * suppress.unsqueeze(-1)).sum(dim=2)

    Creates Mexican-hat response — suppresses redundant structural neighbours,
    forcing diverse representations and enhancing feature contrast.

    Parameters
    ----------
    variant : 'wpos'  similarity in W_pos direction space (static, spatial surround)
              'zact'  similarity in current Z direction space (dynamic, decorrelation)
    """
    def __init__(self, base: SGNNET_Resonant, alpha_ahebb: float = 0.3, variant: str = "wpos"):
        super().__init__()
        self.m           = base
        self.alpha_ahebb = alpha_ahebb
        self.variant     = variant

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh

        # Pre-compute static suppression weights for wpos variant (outside loop)
        if self.variant == "wpos":
            # W_pos contains hidden + output rows; slice to hidden neurons only
            N_h     = self.m.base.N_hidden
            W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)         # [N_hidden, D]
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N_hidden, K_hh]
            supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)  # [N,K_hh]
                      ).unsqueeze(0).unsqueeze(-1)                     # [1,N,K_hh,1]

        # Reflection accumulator — leaky memory of what relu suppressed each step.
        # alpha_reflect=0.5 was calibrated in step22b (+5pp). Must persist across K_iter.
        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                              # [B,N,K_hh,D]

            if self.variant == "wpos":
                Z_struct = (Z_nb * supp_w).sum(dim=2)
            else:   # zact: dynamic cosine suppression in current Z space
                Z_n    = F.normalize(Z, dim=-1)
                z_sim  = (Z_n.unsqueeze(2) * F.normalize(Z_nb, dim=-1)).sum(-1)  # [B,N,K_hh]
                supp_z = (1.0 - self.alpha_ahebb * z_sim.clamp(min=0)).unsqueeze(-1)
                Z_struct = (Z_nb * supp_z).sum(dim=2)

            # Reflection: accumulate what the threshold suppressed
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            # Phase inhibition — skip expensive computation when alpha_turing=0
            if self.m.alpha_turing != 0.0:
                Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
                Z_new = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh
            else:
                Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)
