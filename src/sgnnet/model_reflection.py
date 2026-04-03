"""SGNNET_Reflection: sign-based conditional propagation routing.

Signal reflection routing tests whether input-dependent active-path routing
via negative-activation bounce-back improves classification accuracy.

Reflection hypothesis
---------------------
Standard routing only propagates positive activations (via relu gate).
Strongly negative activations are discarded — they carry no signal.

Reflection reclaims that signal: strongly negative activations at neuron h
are inverted and re-injected into h itself (self-inhibition) and/or into
h's neighbours. This creates an input-dependent routing pattern where:
  - positive activations propagate forward through the graph
  - strongly negative activations "bounce back" as a self-correcting signal

Two variants:
  - Leaky reflect (alpha_reflect=0.1, theta=0.0):
    Small strength bounce-back; all negative activations participate equally.
    Low risk of dead neurons.
  - Hard reflect (alpha_reflect=1.0, theta=0.5):
    Only strongly negative activations (below -theta) bounce back;
    full strength. Higher risk of dead neurons (cascade suppression).

Dead neuron tracking
--------------------
Per routing step, we track the fraction of neurons with |Z| < 1e-6
(effectively zero activation). If > 5%, a warning is logged — this
signals a potential dying neuron cascade that will suppress training.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGNNET_Reflection(nn.Module):
    """Reflection routing wrapper for SGNNET_SmallWorld.

    Wraps an SGNNET_SmallWorld base model (or a Resonant/AntiHebbian wrapper
    around one) and overrides the routing loop to add signal reflection.

    After the excitatory gate (relu), the discarded below-threshold activations
    are bounced back as self-inhibitory signal:
        Z_prop    = relu(Z)                           positive activations propagate
        Z_reflect = alpha_reflect * relu(-Z - theta)  strongly negative bounce back
        Z_new     = gather_sum(Z_prop, conn_hh) - Z_reflect

    Note: the sign of Z_reflect is negative (self-inhibitory) to prevent
    cascading amplification of negative signals.

    Parameters
    ----------
    base          : SGNNET_SmallWorld (provides _seed, _route, _readout, conn_hh)
    alpha_reflect : bounce-back strength [0, 1]. 0=disabled, 0.1=leaky, 1.0=hard
    theta         : reflection threshold. Only Z < -theta reflects.
                    theta=0.0 means all negative activations reflect.
                    theta=0.5 means only strongly negative activations.
    """

    def __init__(
        self,
        base: nn.Module,
        alpha_reflect: float = 0.1,
        theta: float = 0.0,
    ):
        super().__init__()
        self.base          = base
        self.alpha_reflect = alpha_reflect
        self.theta         = theta
        # Track dead neuron diagnostics across the last forward pass
        self._last_dead_fracs: list[float] = []

    # -- Compatibility shims so Trainer can access W_pos/W_phase ---------------

    @property
    def W_pos(self) -> torch.Tensor:
        return self.base.W_pos

    @property
    def W_phase(self):
        return getattr(self.base, "W_phase", None)

    def tick_epoch(self):
        """Delegate topology rebuilds to base model if supported."""
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    # -- Forward ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reflection routing forward pass.

        Replaces the standard _route loop with a reflection-augmented version.
        Dead neuron fraction is tracked per iteration step as a diagnostic.
        """
        base = self.base
        Z = base._seed(x)                           # [B, N_hidden, D]
        conn_hh = base.conn_hh                      # [N_hidden, K_hh]

        dead_fracs: list[float] = []

        for step in range(base.K_iter):
            # ── Dead neuron tracking ──────────────────────────────────────────
            # Fraction of neurons where |Z| < 1e-6 (essentially zero activation)
            with torch.no_grad():
                dead_frac = (Z.norm(dim=-1) < 1e-6).float().mean().item()
                dead_fracs.append(dead_frac)

            if dead_frac > 0.05:
                warnings.warn(
                    f"[SGNNET_Reflection] step={step}: dead_frac={dead_frac:.1%} "
                    f"(>5% — dying neuron cascade risk). "
                    f"alpha_reflect={self.alpha_reflect}, theta={self.theta}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # ── Reflection routing ────────────────────────────────────────────
            # Positive activations propagate forward (standard excitatory gate)
            Z_prop = F.relu(Z)                                     # [B, N, D]

            # Strongly negative activations bounce back (reflection signal)
            # relu(-Z - theta): nonzero only where Z < -theta
            Z_reflect = self.alpha_reflect * F.relu(-Z - self.theta)  # [B, N, D]

            # Gather-sum structural neighbours (O(N*K) — same as SmallWorld)
            Z_struct = Z_prop[:, conn_hh, :].sum(dim=2)            # [B, N, D]

            # Combine: neighbour excitation minus self-reflection
            # The minus sign makes reflection self-inhibitory (stabilising)
            Z_new = Z_struct - Z_reflect                           # [B, N, D]
            Z = base._normalise(Z_new)

        self._last_dead_fracs = dead_fracs
        return base._readout(Z)

    def dead_neuron_report(self) -> dict:
        """Return dead neuron diagnostics from last forward pass.

        Returns
        -------
        dict with:
            max_dead_frac  : worst fraction across all routing steps
            mean_dead_frac : average fraction
            per_step       : list of per-step fractions
            warning        : True if max_dead_frac > 0.05
        """
        if not self._last_dead_fracs:
            return {"max_dead_frac": 0.0, "mean_dead_frac": 0.0,
                    "per_step": [], "warning": False}
        max_d = max(self._last_dead_fracs)
        mean_d = sum(self._last_dead_fracs) / len(self._last_dead_fracs)
        return {
            "max_dead_frac":  max_d,
            "mean_dead_frac": mean_d,
            "per_step":       self._last_dead_fracs,
            "warning":        max_d > 0.05,
        }
