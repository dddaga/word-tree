"""
Run equilibrium experiment: real/imag split on conduction/radiation, forward-only, Iris input.
Logs activations each step for later analysis. No existing files modified.

Run from repo root (word-tree):
  python fixed_io_nodes/experiments/equilibrium_real_imag/run_equilibrium_experiment.py
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
from sklearn.datasets import load_iris

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fixed_io_nodes.distributed._config_utils import get_config, get_node_store_from_config
from fixed_io_nodes.core.full_model import get_dtype

from .gnn import RealImagConductionRadiationGNN
from .activation_logger import ActivationLogger


def load_iris_sample(device, dtype=torch.float32):
    """One Iris sample as (input_node_count, vector_dim) for injection. Same preprocessing as phase_rad_mag_cond."""
    iris_data = load_iris()
    X_full = torch.tensor(iris_data.data, dtype=dtype)
    X = torch.tensor(iris_data.data[:1], dtype=dtype)
    X = (X - X_full.min(dim=0).values) / (
        X_full.max(dim=0).values - X_full.min(dim=0).values + 1e-8
    )
    X = torch.arccos(X.clamp(0, 1)).squeeze(0)
    input_nodes = 4
    vector_dim = 4
    input_values = X.unsqueeze(0).expand(input_nodes, vector_dim).to(device)
    return input_values


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
        iterations=1,
        activation_threshold=0.1,
        gamma=1.5,
        temporal_decay=0.9,
        lr=0.001,
        accumulation_steps=4,
        device=str(device),
    )

    node_store = get_node_store_from_config(cfg)
    dtype = get_dtype(cfg["model"].get("dtype", "float32"))

    gnn = RealImagConductionRadiationGNN(
        node_store=node_store,
        cardinality=cfg["graph"]["cardinality"],
        radiation_targets=cfg["graph"]["radiation_targets"],
        total_nodes=cfg["graph"]["total_nodes"],
        input_nodes=cfg["graph"]["input_nodes"],
        output_nodes=cfg["graph"]["output_nodes"],
        phase_bins=cfg["model"].get("phase_bins", 256),
        mag_bins=cfg["model"].get("mag_bins", 256),
        vector_dim=cfg["model"]["vector_dim"],
        iterations=cfg["model"]["iterations"],
        activation_threshold=cfg["model"]["activation_threshold"],
        gamma=cfg["model"].get("gamma", 1.5),
        temporal_decay=cfg["model"].get("temporal_decay", 0.9),
        device=cfg["system"]["device"],
        verbose=False,
        dtype=dtype,
    )
    if hasattr(gnn, "sync_weights"):
        gnn.sync_weights()

    num_steps = 100
    run_dir = _here / "runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "activations.csv"

    input_values = load_iris_sample(device, dtype)
    gamma = cfg["model"].get("gamma", 1.5)
    vector_dim = cfg["model"]["vector_dim"]

    with ActivationLogger(
        str(log_path), vector_dim=vector_dim, gamma=gamma, include_phase_mag=False
    ) as logger:
        gnn.one_step_forward(input_values)
        logger.log_step(0, dict(gnn.active_nodes), gamma=gamma)
        for step in range(1, num_steps):
            gnn.one_step_forward()
            logger.log_step(step, dict(gnn.active_nodes), gamma=gamma)

    print(f"Logged {num_steps} steps to {log_path}")
    print(f"Run directory: {run_dir}")
    print("Run analyze_equilibrium.py on the log to check equilibrium.")


if __name__ == "__main__":
    main()
