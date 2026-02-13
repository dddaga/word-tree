"""
Log activations of all active nodes at each time step to a CSV for stability analysis.
Forward-only experiment: no backward pass.
"""

import csv
from pathlib import Path
from typing import Any

import torch


def log_step(
    step: int,
    active_nodes: dict,
    vector_dim: int,
    csv_writer: csv.DictWriter,
    file_handle: Any,
) -> None:
    """Append one row per active node for this step. Detaches tensors."""
    for node_id, node in active_nodes.items():
        phase = node.phase_activation.detach()
        mag = node.mag_activation.detach()
        strength = node.activation_strength.detach()
        if phase.dim() == 0:
            phase = phase.unsqueeze(0)
        if mag.dim() == 0:
            mag = mag.unsqueeze(0)
        row = {"step": step, "node_id": node.id}
        for d in range(vector_dim):
            row[f"phase_{d}"] = f"{phase[d].item():.6f}" if d < phase.numel() else ""
        for d in range(vector_dim):
            row[f"mag_{d}"] = f"{mag[d].item():.6f}" if d < mag.numel() else ""
        row["activation_strength"] = f"{strength.item():.6f}"
        csv_writer.writerow(row)
    file_handle.flush()


def open_activation_log(log_path: Path, vector_dim: int):
    """Create CSV and return (file_handle, csv_writer) with header."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "node_id"] + [f"phase_{d}" for d in range(vector_dim)] + [f"mag_{d}" for d in range(vector_dim)] + ["activation_strength"]
    f = open(log_path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    return f, w
