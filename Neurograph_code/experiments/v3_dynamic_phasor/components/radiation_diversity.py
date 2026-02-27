"""
Issue 5: Auxiliary Loss for Radiation Diversity
Entropy regularization on radiation target distribution to prevent hub collapse.
"""

import torch
import math
from typing import Dict


class RadiationDiversityTracker:
    """Tracks radiation target distribution and computes entropy-based diversity loss."""

    def __init__(self, total_nodes: int, alpha: float = 0.01):
        self.total_nodes = total_nodes
        self.alpha = alpha
        self.hit_counts = torch.zeros(total_nodes)
        self.total_hits = 0

    def record_radiation_targets(self, target_indices: list) -> None:
        """Record which nodes were selected as radiation targets during a propagation step."""
        for idx in target_indices:
            if isinstance(idx, str):
                idx = int(idx[1:])  # "n42" -> 42
            if 0 <= idx < self.total_nodes:
                self.hit_counts[idx] += 1
                self.total_hits += 1

    def compute_entropy_loss(self) -> torch.Tensor:
        """
        Compute diversity penalty: L_diversity = -alpha * H(p) / H_max.
        Higher entropy = more diverse = lower penalty (more negative * -alpha = smaller loss).
        Returns a positive loss value that should be minimized (want high entropy).
        """
        if self.total_hits == 0:
            return torch.tensor(0.0)

        # Compute probability distribution over nodes
        p = self.hit_counts / self.total_hits
        # Filter to non-zero entries for log
        mask = p > 0
        if mask.sum() <= 1:
            return torch.tensor(0.0)

        # Shannon entropy
        H = -torch.sum(p[mask] * torch.log(p[mask]))

        # Maximum possible entropy (uniform distribution over all nodes)
        H_max = math.log(self.total_nodes)
        if H_max == 0:
            return torch.tensor(0.0)

        # Normalized entropy in [0, 1]; higher = more diverse
        normalized_entropy = H / H_max

        # Loss: penalize low diversity (1 - normalized_entropy)
        # So that minimizing loss = maximizing entropy
        loss = self.alpha * (1.0 - normalized_entropy)
        return loss

    def get_metrics(self) -> Dict:
        """Return diagnostic metrics about radiation diversity."""
        if self.total_hits == 0:
            return {
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "gini_coefficient": 0.0,
                "top10_hit_fraction": 0.0,
                "unique_targets": 0,
            }

        p = self.hit_counts / self.total_hits
        mask = p > 0
        H = -torch.sum(p[mask] * torch.log(p[mask])).item()
        H_max = math.log(self.total_nodes) if self.total_nodes > 1 else 1.0

        # Gini coefficient
        sorted_counts = torch.sort(self.hit_counts)[0]
        n = self.total_nodes
        index = torch.arange(1, n + 1, dtype=torch.float)
        gini = (2.0 * torch.sum(index * sorted_counts) / (n * torch.sum(sorted_counts)) - (n + 1) / n).item() if torch.sum(sorted_counts) > 0 else 0.0

        # Top-10 hit fraction
        top10_hits = torch.topk(self.hit_counts, min(10, self.total_nodes)).values.sum().item()
        top10_frac = top10_hits / self.total_hits if self.total_hits > 0 else 0.0

        unique = (self.hit_counts > 0).sum().item()

        return {
            "entropy": H,
            "normalized_entropy": H / H_max if H_max > 0 else 0.0,
            "gini_coefficient": gini,
            "top10_hit_fraction": top10_frac,
            "unique_targets": unique,
        }

    def reset(self) -> None:
        """Reset counters at epoch boundary."""
        self.hit_counts.zero_()
        self.total_hits = 0
