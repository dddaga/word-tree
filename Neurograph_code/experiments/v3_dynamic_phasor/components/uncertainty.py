"""
Issue 9: Uncertainty Estimation
MC Dropout in discrete space, learnable null activation, temperature scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional


class MCDropoutUncertainty(nn.Module):
    """Uncertainty estimation via MC Dropout + null activations + temperature scaling."""

    def __init__(
        self,
        vector_dim: int = 5,
        phase_bins: int = 512,
        mag_bins: int = 1024,
        dropout_rate: float = 0.1,
        num_mc_samples: int = 10,
    ):
        super().__init__()
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.dropout_rate = dropout_rate
        self.num_mc_samples = num_mc_samples

        # Learnable null activation (replaces -inf for inactive outputs)
        self.null_phase = nn.Parameter(
            torch.randint(0, phase_bins, (vector_dim,)).float()
        )
        self.null_mag = nn.Parameter(
            torch.randint(0, mag_bins, (vector_dim,)).float()
        )

        # Temperature for calibration (grid-searched on val set)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def apply_phase_dropout(
        self,
        phase_indices: torch.Tensor,
        mag_indices: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly zero out phase/mag dimensions during training.
        At eval time, this is a no-op unless called explicitly for MC sampling.
        """
        if not training and not self._mc_mode:
            return phase_indices, mag_indices

        mask = torch.bernoulli(
            torch.full(phase_indices.shape, 1.0 - self.dropout_rate,
                       device=phase_indices.device)
        ).long()

        masked_phase = phase_indices * mask
        masked_mag = mag_indices * mask
        return masked_phase, masked_mag

    def estimate_uncertainty(
        self,
        forward_fn: Callable,
        input_context: Dict,
    ) -> Dict:
        """
        Run N stochastic forward passes and compute output statistics.

        Args:
            forward_fn: Callable that takes input_context and returns logits [num_classes]
            input_context: The input to the model

        Returns:
            mean_logits, variance, entropy, confidence
        """
        self._mc_mode = True
        all_logits = []

        for _ in range(self.num_mc_samples):
            logits = forward_fn(input_context)
            if isinstance(logits, torch.Tensor):
                all_logits.append(logits.detach())

        self._mc_mode = False

        if not all_logits:
            return {
                "mean_logits": torch.zeros(1),
                "variance": torch.zeros(1),
                "entropy": torch.tensor(0.0),
                "confidence": torch.tensor(0.0),
            }

        stacked = torch.stack(all_logits)  # [N, num_classes]
        mean_logits = stacked.mean(dim=0)
        variance = stacked.var(dim=0)

        # Predictive entropy
        mean_probs = F.softmax(mean_logits, dim=-1)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10))

        confidence = mean_probs.max()

        return {
            "mean_logits": mean_logits,
            "variance": variance,
            "entropy": entropy,
            "confidence": confidence,
        }

    def get_null_activation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return learnable null phase/mag as long tensors (clamped to valid range)."""
        null_p = torch.clamp(self.null_phase.detach().long(), 0, self.phase_bins - 1)
        null_m = torch.clamp(self.null_mag.detach().long(), 0, self.mag_bins - 1)
        return null_p, null_m

    def calibrate_temperature(
        self,
        logits_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ) -> float:
        """
        Grid-search optimal temperature T on validation set.
        Minimizes NLL of softmax(logits/T) against true labels.
        """
        all_logits = torch.stack(logits_list)  # [N, C]
        all_labels = torch.stack(labels_list)  # [N]

        best_t = 1.0
        best_nll = float("inf")

        for t in np.linspace(0.1, 5.0, 50):
            scaled = all_logits / t
            nll = F.cross_entropy(scaled, all_labels).item()
            if nll < best_nll:
                best_nll = nll
                best_t = t

        with torch.no_grad():
            self.temperature.fill_(best_t)

        return best_t

    @property
    def _mc_mode(self):
        return getattr(self, "__mc_mode", False)

    @_mc_mode.setter
    def _mc_mode(self, val):
        self.__mc_mode = val
