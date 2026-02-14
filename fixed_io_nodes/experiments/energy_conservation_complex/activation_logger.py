"""Log activations and components (real, imag, theta, magnitude) per step for conservation analysis."""

import csv
import math
from pathlib import Path
from typing import Dict, Any

import torch

from .utils import activation_real_imag


class ActivationLogger:
    """Appends per (step, node_id) activations and components to CSV."""

    def __init__(
        self,
        log_path: str,
        vector_dim: int,
        gamma: float,
        include_phase_mag: bool = True,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dim = vector_dim
        self.gamma = gamma
        self.include_phase_mag = include_phase_mag
        self._file = None
        self._writer = None
        self._header_written = False

    def _ensure_header(self):
        if self._header_written:
            return
        self._file = open(self.log_path, "w", newline="")
        fieldnames = [
            "step",
            "node_id",
            "real",
            "imag",
            "theta",
            "magnitude",
            "activation_strength",
        ]
        if self.include_phase_mag:
            fieldnames += [f"phase_{d}" for d in range(self.vector_dim)]
            fieldnames += [f"mag_{d}" for d in range(self.vector_dim)]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()
        self._header_written = True

    def log_step(self, step: int, active_nodes: Dict[Any, Any], gamma: float = None):
        gamma = gamma if gamma is not None else self.gamma
        self._ensure_header()
        for node_id, node in active_nodes.items():
            phase = node.phase_activation.detach()
            mag = node.mag_activation.detach()
            strength = node.activation_strength.detach()
            if strength.dim() > 0:
                strength = strength.sum().item()
            else:
                strength = strength.item()
            real, imag = activation_real_imag(
                phase.unsqueeze(0), mag.unsqueeze(0), gamma
            )
            real = real.squeeze(0).item()
            imag = imag.squeeze(0).item()
            magnitude = (real * real + imag * imag + 1e-12) ** 0.5
            theta = math.atan2(imag, real)
            row = {
                "step": step,
                "node_id": node_id,
                "real": f"{real:.8f}",
                "imag": f"{imag:.8f}",
                "theta": f"{theta:.8f}",
                "magnitude": f"{magnitude:.8f}",
                "activation_strength": f"{strength:.8f}",
            }
            if self.include_phase_mag:
                for d in range(self.vector_dim):
                    row[f"phase_{d}"] = f"{phase[d].item():.8f}"
                    row[f"mag_{d}"] = f"{mag[d].item():.8f}"
            self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
