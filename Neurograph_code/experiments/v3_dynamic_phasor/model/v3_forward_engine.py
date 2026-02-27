"""
V3 Forward Engine
Wraps VectorizedForwardEngine with load balancing, diversity tracking,
magnitude clamping, and per-step timing.
"""

import torch
import time
from typing import Dict, List, Tuple, Optional

from core.modular_forward_engine import VectorizedForwardEngine
from core.node_store import NodeStore
from core.radiation import get_radiation_neighbors, clear_radiation_cache

from experiments.v3_dynamic_phasor.components.signal_normalization import PhasorNormalization
from experiments.v3_dynamic_phasor.components.radiation_diversity import RadiationDiversityTracker
from experiments.v3_dynamic_phasor.components.load_balancer import RadiationLoadBalancer


class V3ForwardEngine:
    """Forward engine with V3 components integrated."""

    def __init__(
        self,
        base_engine: VectorizedForwardEngine,
        node_store: NodeStore,
        normalizer: PhasorNormalization,
        diversity_tracker: RadiationDiversityTracker,
        load_balancer: RadiationLoadBalancer,
        enable_diversity: bool = True,
        enable_balance: bool = True,
        enable_normalization: bool = True,
    ):
        self.engine = base_engine
        self.node_store = node_store
        self.normalizer = normalizer
        self.diversity_tracker = diversity_tracker
        self.load_balancer = load_balancer

        self.enable_diversity = enable_diversity
        self.enable_balance = enable_balance
        self.enable_normalization = enable_normalization

        # Timing
        self.step_times: List[float] = []
        self.last_forward_time: float = 0.0

    def forward_pass(
        self,
        input_context: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        Run the forward pass through the base engine, then apply V3 post-processing.

        Returns the activation table from the base engine.
        """
        t0 = time.perf_counter()

        # Pre-step: invalidate cache if needed
        if self.enable_balance and self.load_balancer.should_invalidate_cache():
            clear_radiation_cache()

        # Run base forward pass
        activation_table = self.engine.forward_pass_vectorized(input_context)

        # Post-processing: clamp magnitude indices on output nodes
        if self.enable_normalization:
            self._clamp_output_magnitudes()

        # Record radiation targets for diversity tracking
        if self.enable_diversity:
            self._record_radiation_from_stats()

        # Load balancer step iteration
        if self.enable_balance:
            self.load_balancer.step_iteration()

        elapsed = time.perf_counter() - t0
        self.step_times.append(elapsed)
        self.last_forward_time = elapsed

        return activation_table

    def _clamp_output_magnitudes(self) -> None:
        """Clamp magnitude indices on all active output nodes in the node store."""
        for node_id in self.engine.output_nodes:
            try:
                mag = self.node_store.get_mag(node_id)
                clamped = self.normalizer.clamp_magnitude_indices(mag)
                if not torch.equal(mag, clamped):
                    self.node_store.mag_table[node_id].data.copy_(clamped)
            except (KeyError, AttributeError):
                pass

    def _record_radiation_from_stats(self) -> None:
        """Extract radiation targets from engine stats and record for diversity."""
        stats = self.engine.get_performance_stats()
        # The base engine doesn't expose individual targets, so we record
        # which output nodes activated as a proxy
        active_outputs = self.engine.get_active_output_nodes()
        if active_outputs:
            self.diversity_tracker.record_radiation_targets(active_outputs)

    def get_active_output_nodes(self) -> List[str]:
        return self.engine.get_active_output_nodes()

    def get_timing_stats(self) -> Dict:
        if not self.step_times:
            return {"forward_time_ms": 0.0, "num_steps": 0}
        return {
            "forward_time_ms": self.last_forward_time * 1000,
            "mean_step_time_ms": sum(self.step_times) / len(self.step_times) * 1000,
            "num_steps": len(self.step_times),
        }

    def reset_timing(self) -> None:
        self.step_times.clear()
