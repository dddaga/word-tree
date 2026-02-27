"""
Complexity Tracker
Measures space (parameter count, memory) and time (train/inference latency) complexity.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional, Callable


class ComplexityTracker:
    """Tracks space and time complexity of V3 experiment."""

    def __init__(self):
        self.epoch_train_times: List[float] = []
        self.sample_train_times: List[float] = []
        self.inference_latencies: List[float] = []
        self.forward_times: List[float] = []
        self.backward_times: List[float] = []

    # ------------------------------------------------------------------ space
    def measure_space_complexity(self, modules: Dict[str, torch.nn.Module]) -> Dict:
        """
        Measure parameter count and memory for a set of named modules.

        Args:
            modules: {"node_store": ..., "input_adapter": ..., ...}

        Returns:
            Space complexity dict with totals and breakdown.
        """
        breakdown = {}
        total_params = 0
        total_bytes = 0

        for name, mod in modules.items():
            params = sum(p.numel() for p in mod.parameters())
            mem = sum(p.numel() * p.element_size() for p in mod.parameters())
            breakdown[name] = {"parameters": params, "memory_bytes": mem}
            total_params += params
            total_bytes += mem

        return {
            "total_parameters": total_params,
            "memory_bytes": total_bytes,
            "memory_mb": total_bytes / (1024 * 1024),
            "breakdown": breakdown,
        }

    # ------------------------------------------------------------------ time
    def record_epoch_time(self, seconds: float) -> None:
        self.epoch_train_times.append(seconds)

    def record_sample_train_time(self, seconds: float) -> None:
        self.sample_train_times.append(seconds)

    def measure_inference_latency(
        self,
        forward_fn: Callable,
        input_context: Dict,
        num_runs: int = 200,
    ) -> float:
        """
        Measure median inference latency over num_runs.

        Args:
            forward_fn: Callable that takes input_context and returns output.
            input_context: A single sample input context.
            num_runs: Number of runs for timing.

        Returns:
            Median latency in microseconds.
        """
        latencies = []
        # Warm up
        for _ in range(5):
            forward_fn(input_context)

        for _ in range(num_runs):
            t0 = time.perf_counter()
            forward_fn(input_context)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed * 1e6)  # seconds to microseconds

        self.inference_latencies = latencies
        return float(np.median(latencies))

    def get_time_complexity(self) -> Dict:
        return {
            "train_time_per_epoch_s": self.epoch_train_times,
            "train_time_per_sample_ms": [t * 1000 for t in self.sample_train_times],
            "inference_latency_us": float(np.median(self.inference_latencies)) if self.inference_latencies else 0.0,
            "inference_latency_std_us": float(np.std(self.inference_latencies)) if self.inference_latencies else 0.0,
        }

    def get_full_report(self, modules: Dict[str, torch.nn.Module]) -> Dict:
        space = self.measure_space_complexity(modules)
        time_c = self.get_time_complexity()
        return {"space_complexity": space, "time_complexity": time_c}
