"""SGNNET_PhaseRouting: distance-based phase shift routing.

Architecture
------------
Z = [Z_mag | Z_phase_1 ... Z_phase_63]  (D=64 total)
  Z_mag   [B,N]    : activation magnitude, normalized across N neurons each step
  Z_phase [B,N,63] : cyclic phases in [-pi, pi), shifted by freq*dist each routing step

Phase shift when activation travels from neuron A to B:
  delta_phi_k = freq_k * ||W_pos[A] - W_pos[B]||
  freq_mode="capped_exp": freq_k = 2^min(k//2, 10) for k in range(D-1)
  freq_mode="harmonic_primes": freq_k = 2*pi / prime_k  (first D-1 primes)
  Z_phase = torch.remainder(Z_phase + delta_phi, 2*pi) - pi

Magnitude modes (6 total):
  "independent"       : gate=sum;              phase=raw_sum
  "weighted_phase"    : gate=sum;              phase=mag-weighted_sum
  "coherent"          : gate=coherence-weight; phase=raw_sum
  "coherent_weighted" : gate=coherence-weight; phase=mag-weighted_sum
  "decay_coherence"   : gate=decay*coherence;  phase=raw_sum
  "decay_weighted"    : gate=decay*coherence;  phase=decay-weighted_sum
                        decay weights: Σ_j decay[h,j,k]*Z_phase_arr[j,k] / Σ_j decay[h,j,k]
                        nearby sources (low decay) dominate phase update over far sources

coherence_ref controls the reference in coherence calculation:
  "dynamic"  : ref = Z_phase (h's current activation phase) — original design
  "anchor"   : ref = W_phase[h, 1:] (h's learned static phase anchor)
  "absolute" : no reference — cos(Z_phase_arr)

AntiHebb optionally applied to gate/magnitude (ah_alpha > 0).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.sgnnet.model_smallworld import SGNNET_SmallWorld


def _sieve_primes(n: int) -> list[int]:
    """Return list of primes up to n (inclusive)."""
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]


class SGNNET_PhaseRouting(nn.Module):
    def __init__(self, base: SGNNET_SmallWorld, K_phase=8, beam_size=16,
                 theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
                 magnitude_mode="independent",   # "independent" | "weighted_phase" | "coherent" | "coherent_weighted" | "decay_coherence" | "decay_weighted"
                 coherence_ref="dynamic",         # "dynamic" | "anchor" | "absolute"
                 freq_mode="capped_exp",          # "capped_exp" | "harmonic_primes"
                 lambda_decay=1.0,               # for decay_coherence / decay_weighted modes
                 ah_alpha=0.0):
        super().__init__()
        self.base           = base
        self.K_phase        = K_phase
        self.beam_size      = beam_size
        self.alpha_reflect  = alpha_reflect
        self.alpha_turing   = alpha_turing
        self.magnitude_mode = magnitude_mode
        self.coherence_ref  = coherence_ref
        self.freq_mode      = freq_mode
        self.lambda_decay   = lambda_decay
        self.ah_alpha       = ah_alpha

        N = base.N_hidden
        D = base.W_pos.shape[1]  # D=64

        self.theta   = nn.Parameter(torch.full((N,), theta_init))
        self.W_phase = nn.Parameter(torch.rand(N, D) * 2 * math.pi - math.pi)  # init in [-pi, pi]

        # Fourier frequencies for phase shift
        if freq_mode == "capped_exp":
            freqs = torch.tensor(
                [2 ** min(k // 2, 10) for k in range(D - 1)], dtype=torch.float32
            )
        elif freq_mode == "harmonic_primes":
            primes = _sieve_primes(400)[: D - 1]   # first D-1 primes (63 for D=64)
            freqs = torch.tensor(
                [2 * math.pi / p for p in primes], dtype=torch.float32
            )
        else:
            raise ValueError(f"Unknown freq_mode: {freq_mode!r}")

        self.register_buffer("phase_freqs", freqs)  # [D-1]

        self._build_phase_graph()

    def _build_phase_graph(self):
        """K-NN graph over W_phase (same logic as SGNNET_Resonant)."""
        with torch.no_grad():
            Wp = F.normalize(self.W_phase.detach(), dim=-1)
            device = Wp.device
            try:
                import faiss
                Wp_np = Wp.cpu().float().numpy()
                N, D  = Wp_np.shape
                fi    = faiss.IndexFlatIP(D)
                fi.add(Wp_np)
                _, I  = fi.search(Wp_np, self.K_phase + 1)
                conn  = np.array([[j for j in row if j != i][:self.K_phase]
                                  for i, row in enumerate(I)], dtype=np.int64)
                idx = torch.tensor(conn, dtype=torch.long, device=device)
            except ImportError:
                # Cast to float32 to avoid float16 overflow on MPS
                sim = (Wp @ Wp.T).float()
                sim.fill_diagonal_(-1e9)
                _, idx = sim.topk(self.K_phase, dim=-1)
        self.register_buffer("conn_phase", idx)

    def tick_epoch(self):
        self._build_phase_graph()
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    @property
    def W_pos(self):
        return self.base.W_pos

    def forward(self, x):
        Z = self.base._seed(x)   # [B, N, D]
        B, N, D = Z.shape

        theta_pos = self.theta.abs()          # [N]
        W_pos_h   = self.base.W_pos[:N]       # [N, D]
        nb_idx    = self.base.conn_hh         # [N, K_hh]

        # Precompute W_pos distances for phase shifts (static per forward)
        W_pos_nb  = W_pos_h[nb_idx]           # [N, K_hh, D]
        dist      = (W_pos_h.unsqueeze(1) - W_pos_nb).norm(dim=-1)  # [N, K_hh]
        # delta_phi: [N, K_hh, D-1]
        delta_phi = dist.unsqueeze(-1) * self.phase_freqs.unsqueeze(0).unsqueeze(0)

        # Precompute decay matrix for decay modes (static per forward)
        use_decay = self.magnitude_mode in ("decay_coherence", "decay_weighted")
        if use_decay:
            # decay: [N, K_hh, D-1]
            decay = torch.exp(-self.lambda_decay * delta_phi / (2 * math.pi))

        # AntiHebb suppression weights (static per forward if ah_alpha > 0)
        if self.ah_alpha > 0:
            W_n     = F.normalize(W_pos_h, dim=-1)
            W_n_nb  = F.normalize(W_pos_nb, dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n_nb).sum(-1)           # [N, K_hh]
            supp_w  = (1.0 - self.ah_alpha * pos_sim.clamp(min=0))  # [N, K_hh]
            supp_w  = supp_w.unsqueeze(0)                            # [1, N, K_hh]

        # W_phase anchor: [N, D-1] — phase channels only (slice off dim 0 = magnitude dim)
        # W_phase is [N, D]; we treat dim 0 as magnitude-like init, dims 1: as phases.
        # For anchor coherence_ref we use W_phase[:, 1:] reshaped to [1, N, 1, D-1].
        if self.coherence_ref == "anchor":
            W_ph_anchor = F.normalize(self.W_phase[:N, 1:], dim=-1)  # [N, D-1]
            anchor_ref  = W_ph_anchor.unsqueeze(0).unsqueeze(2)       # [1, N, 1, D-1]

        for _ in range(self.base.K_iter):
            Z_mag   = Z[:, :, 0]    # [B, N]
            Z_phase = Z[:, :, 1:]   # [B, N, D-1]

            # Threshold gate on magnitude
            active      = (Z_mag > theta_pos.unsqueeze(0)).float()       # [B, N]
            Z_mag_fwd   = Z_mag   * active                                # [B, N]
            Z_phase_fwd = Z_phase * active.unsqueeze(-1)                  # [B, N, D-1]

            # Gather neighbours
            Z_mag_nb   = Z_mag_fwd[:, nb_idx]    # [B, N, K_hh]
            Z_phase_nb = Z_phase_fwd[:, nb_idx]  # [B, N, K_hh, D-1]

            # Apply phase shift to arriving signals
            Z_phase_arr = torch.remainder(
                Z_phase_nb + delta_phi.unsqueeze(0), 2 * math.pi
            ) - math.pi                            # [B, N, K_hh, D-1]

            # ----------------------------------------------------------------
            # Build coherence reference for coherent / decay modes
            # ----------------------------------------------------------------
            if self.magnitude_mode in ("coherent", "coherent_weighted",
                                       "decay_coherence", "decay_weighted"):
                if self.coherence_ref == "dynamic":
                    ref = Z_phase.unsqueeze(2)          # [B, N, 1, D-1]
                elif self.coherence_ref == "anchor":
                    ref = anchor_ref                    # [1, N, 1, D-1]
                else:  # "absolute"
                    ref = 0.0

            # ----------------------------------------------------------------
            # Magnitude gate computation
            # ----------------------------------------------------------------
            if self.magnitude_mode == "independent":
                # Gate: sum; Phase: raw sum
                if self.ah_alpha > 0:
                    Z_mag_new = (Z_mag_nb * supp_w).sum(2)              # [B, N]
                else:
                    Z_mag_new = Z_mag_nb.sum(2)                         # [B, N]
                Z_phase_new = Z_phase_arr.sum(2)                        # [B, N, D-1]

            elif self.magnitude_mode == "weighted_phase":
                # Gate: sum; Phase: magnitude-weighted sum
                if self.ah_alpha > 0:
                    Z_mag_new = (Z_mag_nb * supp_w).sum(2)              # [B, N]
                else:
                    Z_mag_new = Z_mag_nb.sum(2)                         # [B, N]
                weights = Z_mag_nb / Z_mag_nb.sum(2, keepdim=True).clamp(min=1e-6)
                # weights: [B, N, K_hh]; Z_phase_arr: [B, N, K_hh, D-1]
                Z_phase_new = (Z_phase_arr * weights.unsqueeze(-1)).sum(2)  # [B, N, D-1]

            elif self.magnitude_mode == "coherent":
                # Gate: coherence-weighted; Phase: raw sum
                coherence = torch.cos(Z_phase_arr - ref).mean(-1)       # [B, N, K_hh]
                if self.ah_alpha > 0:
                    coherence = coherence * supp_w
                Z_mag_new   = (Z_mag_nb * coherence).sum(2)             # [B, N]
                Z_phase_new = Z_phase_arr.sum(2)                        # [B, N, D-1]

            elif self.magnitude_mode == "coherent_weighted":
                # Gate: coherence-weighted; Phase: magnitude-weighted sum
                coherence = torch.cos(Z_phase_arr - ref).mean(-1)       # [B, N, K_hh]
                if self.ah_alpha > 0:
                    coherence = coherence * supp_w
                Z_mag_new   = (Z_mag_nb * coherence).sum(2)             # [B, N]
                weights = Z_mag_nb / Z_mag_nb.sum(2, keepdim=True).clamp(min=1e-6)
                Z_phase_new = (Z_phase_arr * weights.unsqueeze(-1)).sum(2)  # [B, N, D-1]

            elif self.magnitude_mode == "decay_coherence":
                # Gate: decay*coherence; Phase: raw sum
                per_ch = torch.cos(Z_phase_arr - ref)                   # [B, N, K_hh, D-1]
                gate   = (decay.unsqueeze(0) * per_ch).sum(-1)          # [B, N, K_hh]
                norm   = decay.sum(-1).clamp(min=1e-6).unsqueeze(0)     # [1, N, K_hh]
                gate   = gate / norm                                     # [B, N, K_hh]
                if self.ah_alpha > 0:
                    gate = gate * supp_w
                Z_mag_new   = (Z_mag_nb * gate).sum(2)                  # [B, N]
                Z_phase_new = Z_phase_arr.sum(2)                        # [B, N, D-1]

            elif self.magnitude_mode == "decay_weighted":
                # Gate: decay*coherence; Phase: decay-weighted sum
                # Magnitude gate — same as decay_coherence
                per_ch = torch.cos(Z_phase_arr - ref)                   # [B, N, K_hh, D-1]
                gate   = (decay.unsqueeze(0) * per_ch).sum(-1)          # [B, N, K_hh]
                norm   = decay.sum(-1).clamp(min=1e-6).unsqueeze(0)     # [1, N, K_hh]
                gate   = gate / norm                                     # [B, N, K_hh]
                if self.ah_alpha > 0:
                    gate = gate * supp_w
                Z_mag_new   = (Z_mag_nb * gate).sum(2)                  # [B, N]
                # Phase update — decay-weighted: nearby sources dominate
                # decay: [N, K_hh, D-1], Z_phase_arr: [B, N, K_hh, D-1]
                Z_phase_new = (decay.unsqueeze(0) * Z_phase_arr).sum(2)          # [B, N, D-1]
                Z_phase_new = Z_phase_new / decay.sum(-2).clamp(min=1e-6).unsqueeze(0)  # [B, N, D-1]

            else:
                raise ValueError(f"Unknown magnitude_mode: {self.magnitude_mode!r}")

            # Normalize magnitude across N neurons (not across D)
            Z_mag_new = F.normalize(Z_mag_new, dim=1)                   # [B, N]

            # Phase: keep in [-pi, pi)
            Z_phase_new = torch.remainder(Z_phase_new, 2 * math.pi) - math.pi  # [B, N, D-1]

            Z = torch.cat([Z_mag_new.unsqueeze(-1), Z_phase_new], dim=-1)  # [B, N, D]

        # Normalize per-neuron vector to unit norm before readout.
        # SGNNET_Resonant always passes L2-normalized vectors to _readout; without
        # this, the phase channels (up to ±π) produce logit scales 30-165× larger
        # than the readout was designed for, causing gradient explosion.
        Z = F.normalize(Z, dim=-1)
        return self.base._readout(Z)
