"""
Full training for real/imag activation experiment: forward + backward, Iris, loss logging.
Uses RealImagConductionRadiationGNN (theta=sin(m), real on conduction, imag on radiation).

Run from repo root (word-tree):
  python -m fixed_io_nodes.experiments.equilibrium_real_imag.run_training

Or with path on PYTHONPATH:
  python fixed_io_nodes/experiments/equilibrium_real_imag/run_training.py
"""

import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fixed_io_nodes.distributed._config_utils import get_config
from fixed_io_nodes.experiments.equilibrium_real_imag.layer import create_real_imag_layer
from fixed_io_nodes.experiments.equilibrium_real_imag.optim import RealImagGNNAdam


def load_iris_dataset():
    """Iris: (N, 1, 4) after arccos, labels (N,) 0/1/2."""
    iris_data = load_iris()
    X = torch.tensor(iris_data.data, dtype=torch.float32)
    X = (X - X.min(dim=0).values) / (X.max(dim=0).values - X.min(dim=0).values + 1e-8)
    X = torch.arccos(X.clamp(0, 1)).reshape(-1, 1, 4)
    y = torch.tensor(iris_data.target, dtype=torch.long)
    return TensorDataset(X, y)


class MLPRealImagGNNModel(nn.Module):
    """MLP -> RealImag GNN layer. Same shape as phase_rad_mag_cond."""

    def __init__(
        self,
        input_dim=4,
        input_nodes=4,
        output_nodes=3,
        vector_dim=4,
        lr=0.001,
        accumulation_steps=4,
        **gnn_kwargs,
    ):
        super().__init__()
        self.input_nodes = input_nodes
        self.vector_dim = vector_dim
        self.linear = nn.Linear(input_dim, input_nodes * vector_dim)
        self.tanh = nn.Tanh()
        self.gnn = create_real_imag_layer(
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            vector_dim=vector_dim,
            lr=lr,
            accumulation_steps=accumulation_steps,
            **gnn_kwargs,
        )

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(1) if x.dim() > 2 else x
        h = self.tanh(self.linear(x))
        h = h.view(B, self.input_nodes, self.vector_dim)
        return self.gnn(h)


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
        gamma=1.5,
        temporal_decay=0.9,
        lr=0.001,
        accumulation_steps=4,
        device=str(device),
    )

    model = MLPRealImagGNNModel(
        input_dim=4,
        input_nodes=cfg["graph"]["input_nodes"],
        output_nodes=cfg["graph"]["output_nodes"],
        vector_dim=cfg["model"]["vector_dim"],
        lr=cfg["training"]["lr"],
        accumulation_steps=cfg["training"]["accumulation_steps"],
    )
    model = model.to(device)

    optimizer = RealImagGNNAdam(
        model, lr=cfg["training"]["lr"], betas=(0.9, 0.999), eps=1e-8
    )
    dataset = load_iris_dataset()
    dataloader = DataLoader(
        dataset, batch_size=cfg["training"]["accumulation_steps"], shuffle=True
    )
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    log_dir = _here / "training_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "loss.csv"

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss"])
        w.writeheader()
        step = 0
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                w.writerow({"step": step, "loss": f"{loss.item():.6f}"})
                f.flush()
                step += 1
            print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")

    try:
        model.gnn.shutdown()
    except Exception:
        pass
    print(f"Done. Log: {log_path}")


if __name__ == "__main__":
    main()
