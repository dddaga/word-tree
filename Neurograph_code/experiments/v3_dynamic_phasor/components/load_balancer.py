"""
Issue 7: Load Balancing (MoE-style)
Score penalty, capacity limits, and temperature-annealed softmax for radiation targets.
"""

import torch
import math
from typing import Dict


class RadiationLoadBalancer:
    """MoE-style load balancing for radiation target selection."""

    def __init__(
        self,
        total_nodes: int,
        beta: float = 0.001,
        capacity_limit: int = 5,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        temperature_anneal_rate: float = 0.995,
        cache_invalidation_frequency: int = 50,
    ):
        self.total_nodes = total_nodes
        self.beta = beta
        self.capacity_limit = capacity_limit
        self.temperature = temperature_init
        self.temperature_min = temperature_min
        self.temperature_anneal_rate = temperature_anneal_rate
        self.cache_invalidation_frequency = cache_invalidation_frequency

        # Per-epoch hit counts (for balance loss)
        self.epoch_hit_counts = torch.zeros(total_nodes)
        # Per-step hit counts (for capacity check)
        self.step_hit_counts = torch.zeros(total_nodes)
        self._iteration_count = 0

    def adjust_scores_for_balance(
        self,
        scores: torch.Tensor,
        candidate_indices: list,
    ) -> torch.Tensor:
        """
        Apply score penalty based on cumulative hit counts.
        effective_score = alignment_score - beta * log(1 + hit_count[target])
        """
        adjusted = scores.clone().float()
        for i, idx in enumerate(candidate_indices):
            node_idx = int(idx[1:]) if isinstance(idx, str) else idx
            if 0 <= node_idx < self.total_nodes:
                penalty = self.beta * torch.log(
                    torch.tensor(1.0 + self.epoch_hit_counts[node_idx].item())
                )
                adjusted[i] = adjusted[i] - penalty
        return adjusted

    def check_capacity(self, target_idx) -> bool:
        """Return True if this node can accept another radiation hit this step."""
        node_idx = int(target_idx[1:]) if isinstance(target_idx, str) else target_idx
        if 0 <= node_idx < self.total_nodes:
            return self.step_hit_counts[node_idx].item() < self.capacity_limit
        return False

    def record_hit(self, target_idx) -> None:
        """Record a radiation hit for both step and epoch tracking."""
        node_idx = int(target_idx[1:]) if isinstance(target_idx, str) else target_idx
        if 0 <= node_idx < self.total_nodes:
            self.step_hit_counts[node_idx] += 1
            self.epoch_hit_counts[node_idx] += 1

    def compute_balance_loss(self) -> torch.Tensor:
        """beta * Var(hit_counts) -- penalizes uneven distribution."""
        if self.epoch_hit_counts.sum() == 0:
            return torch.tensor(0.0)
        variance = torch.var(self.epoch_hit_counts.float())
        return self.beta * variance

    def anneal_temperature(self) -> None:
        """Reduce temperature once per epoch."""
        self.temperature = max(
            self.temperature_min,
            self.temperature * self.temperature_anneal_rate,
        )

    def step_iteration(self) -> None:
        """Reset per-step counters; increment iteration for cache invalidation."""
        self.step_hit_counts.zero_()
        self._iteration_count += 1

    def should_invalidate_cache(self) -> bool:
        """Whether the radiation cache should be cleared this iteration."""
        return (
            self._iteration_count > 0
            and self._iteration_count % self.cache_invalidation_frequency == 0
        )

    def reset_epoch(self) -> None:
        """Reset epoch-level counters."""
        self.epoch_hit_counts.zero_()
        self.step_hit_counts.zero_()

    def get_metrics(self) -> Dict:
        """Diagnostic metrics for load balancing."""
        counts = self.epoch_hit_counts
        nonzero = counts[counts > 0]
        return {
            "hit_count_variance": torch.var(counts.float()).item() if counts.sum() > 0 else 0.0,
            "hit_count_max": counts.max().item(),
            "hit_count_mean": counts.float().mean().item(),
            "temperature": self.temperature,
            "unique_targets": (counts > 0).sum().item(),
            "capacity_violations": 0,  # tracked externally if needed
        }
