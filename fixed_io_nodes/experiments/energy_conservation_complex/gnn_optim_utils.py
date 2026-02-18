"""
Shared sink/accumulator logic matching distributed GNNAdam behavior.
Used by EnergyConservationGNNAdam so grad accumulation and backprop match the PyTorch-based original.
"""

from typing import List


def clear_gnn_sinks(gradient_sinks: List) -> None:
    """Clear all gradient sinks (same as GNNAdam.zero_grad side effect)."""
    for sink in gradient_sinks:
        if hasattr(sink, "clear"):
            sink.clear()


def step_gnn_from_sinks(gradient_sinks: List, accumulators: List) -> None:
    """Get grads from each sink, feed to accumulator, run step (same as GNNAdam.step GNN part)."""
    for sink, accumulator in zip(gradient_sinks, accumulators):
        phase_grads, mag_grads, phase_grad_freq, mag_grad_freq = sink.get_and_clear()
        if phase_grads is not None or mag_grads is not None:
            if phase_grads is None:
                phase_grads = {}
            if mag_grads is None:
                mag_grads = {}
            accumulator.receive_gradients(
                phase_grads,
                mag_grads,
                phase_grad_freq=phase_grad_freq,
                mag_grad_freq=mag_grad_freq,
            )
            accumulator.step()
