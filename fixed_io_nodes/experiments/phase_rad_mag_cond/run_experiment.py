"""
Forward-only experiment: phase-only radiation, magnitude-only conduction, beam search.
No backward pass. Logs activations of all nodes at each time step to check stability.
Run from repo root (word-tree):

  python fixed_io_nodes/experiments/phase_rad_mag_cond/run_experiment.py
"""

import sys
from pathlib import Path

import torch
from sklearn.datasets import load_iris

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fixed_io_nodes.distributed._config_utils import get_config
from fixed_io_nodes.experiments.phase_rad_mag_cond import create_experiment_layer
from fixed_io_nodes.experiments.phase_rad_mag_cond.activation_logger import (
    open_activation_log,
    log_step,
)


def make_input_values(input_nodes: int, vector_dim: int, device: torch.device, seed: int = 42):
    """One Iris sample as (input_nodes, vector_dim) phase input; mag = 0 at input."""
    iris_data = load_iris()
    X = torch.tensor(iris_data.data[:1], dtype=torch.float32).squeeze(0)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    X = torch.arccos(X.clamp(0, 1))
    # Broadcast to (input_nodes, vector_dim); pad or repeat if needed
    if X.numel() >= input_nodes * vector_dim:
        return X[: input_nodes * vector_dim].reshape(input_nodes, vector_dim).to(device)
    X = X.repeat((input_nodes * vector_dim + X.numel() - 1) // X.numel())[: input_nodes * vector_dim]
    return X.reshape(input_nodes, vector_dim).to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    cfg = get_config(
        input_nodes=4,
        output_nodes=3,
        vector_dim=4,
        total_nodes=100,
        cardinality=4,
        radiation_targets=4,
        iterations=3,
        activation_threshold=0.1,
        beam_top_frac=0.1,
        gamma=1.5,
        temporal_decay=0.9,
        lr=0.001,
        accumulation_steps=4,
        device=str(device),
    )

    layer = create_experiment_layer(
        input_nodes=cfg["graph"]["input_nodes"],
        output_nodes=cfg["graph"]["output_nodes"],
        vector_dim=cfg["model"]["vector_dim"],
        total_nodes=cfg["graph"]["total_nodes"],
        cardinality=cfg["graph"]["cardinality"],
        radiation_targets=cfg["graph"]["radiation_targets"],
        iterations=cfg["model"]["iterations"],
        activation_threshold=cfg["model"]["activation_threshold"],
        beam_top_frac=cfg["model"].get("beam_top_frac", 0.1),
        gamma=cfg["model"]["gamma"],
        temporal_decay=cfg["model"]["temporal_decay"],
        lr=cfg["training"]["lr"],
        accumulation_steps=cfg["training"]["accumulation_steps"],
        device=cfg["system"]["device"],
    )
    gnn = layer.gnn
    vector_dim = cfg["model"]["vector_dim"]

    input_values = make_input_values(
        cfg["graph"]["input_nodes"],
        vector_dim,
        device,
    )

    num_steps = 30
    log_dir = Path("training_runs/phase_rad_mag_cond_experiment")
    log_path = log_dir / "activations.csv"
    f, csv_writer = open_activation_log(log_path, vector_dim)

    try:
        for step in range(num_steps):
            gnn.one_step_forward(input_values if step == 0 else None)
            log_step(step, gnn.active_nodes, vector_dim, csv_writer, f)
            print(f"Step {step + 1}/{num_steps}  active_nodes={len(gnn.active_nodes)}")
    finally:
        f.close()

    print(f"Done. Activations log: {log_path}")


if __name__ == "__main__":
    main()
