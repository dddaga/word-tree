import torch
from typing import Optional, Tuple

class GNNGradientSink:
    """
    Process-local container for GNN gradients from the last backward pass.
    Stores flat tensors: (active_idx, phase_grads, mag_grads).
    """

    def __init__(self):
        self._idx: Optional[torch.Tensor] = None
        self._pg: Optional[torch.Tensor] = None
        self._mg: Optional[torch.Tensor] = None

    def add(self, active_idx: torch.Tensor, phase_grads: torch.Tensor, mag_grads: torch.Tensor):
        """Add flat tensors from a backward pass. Replaces any existing gradients."""
        self._idx = active_idx
        self._pg = phase_grads
        self._mg = mag_grads

    def get_and_clear(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get current gradient tensors, then clear."""
        idx, pg, mg = self._idx, self._pg, self._mg
        self.clear()
        return idx, pg, mg

    def clear(self):
        """Clear stored gradients without returning them."""
        self._idx = None
        self._pg = None
        self._mg = None