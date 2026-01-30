"""
Gradient sink for collecting GNN gradients from distributed layer backward
and making them available to the optimizer on the main process.
"""

import torch
from typing import Dict, Optional, Tuple


class GNNGradientSink:
    """
    Process-local container for GNN gradients from the last backward pass.
    Stores (phase_grads, mag_grads) and per-node frequencies (how many samples contributed).
    """

    def __init__(self):
        self._phase_grads: Optional[Dict[int, torch.Tensor]] = None
        self._mag_grads: Optional[Dict[int, torch.Tensor]] = None
        self._phase_grad_freq: Optional[Dict[int, int]] = None
        self._mag_grad_freq: Optional[Dict[int, int]] = None

    def add(
        self,
        phase_grads: Dict[int, torch.Tensor],
        mag_grads: Dict[int, torch.Tensor],
        phase_grad_freq: Optional[Dict[int, int]] = None,
        mag_grad_freq: Optional[Dict[int, int]] = None,
    ):
        """
        Add gradients from a backward pass. Replaces any existing gradients.
        phase_grad_freq / mag_grad_freq: node_id -> number of samples that contributed.
        If omitted, accumulator treats each node as count 1 (backwards compatible).
        """
        self._phase_grads = phase_grads
        self._mag_grads = mag_grads
        self._phase_grad_freq = phase_grad_freq
        self._mag_grad_freq = mag_grad_freq

    def get_and_clear(self) -> Tuple[
        Optional[Dict[int, torch.Tensor]],
        Optional[Dict[int, torch.Tensor]],
        Optional[Dict[int, int]],
        Optional[Dict[int, int]],
    ]:
        """
        Get current gradients and frequencies, then clear. Returns
        (phase_grads, mag_grads, phase_grad_freq, mag_grad_freq); freqs may be None.
        """
        phase = self._phase_grads
        mag = self._mag_grads
        phase_f = self._phase_grad_freq
        mag_f = self._mag_grad_freq
        self._phase_grads = None
        self._mag_grads = None
        self._phase_grad_freq = None
        self._mag_grad_freq = None
        return phase, mag, phase_f, mag_f

    def clear(self):
        """Clear stored gradients without returning them."""
        self._phase_grads = None
        self._mag_grads = None
        self._phase_grad_freq = None
        self._mag_grad_freq = None
