"""
Save/load full model (MLP + GNN) in a single file.
Expects model to have model.gnn._node_store (e.g. IrisGNNModel with DistributedNeurographLayer).
"""

import torch


def save_full_model(model, path: str):
    """
    Save MLP + GNN weights in one file. Model part excludes gnn._node_store to avoid duplication.
    """
    model_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("gnn._node_store")
    }
    node_store_state = model.gnn._node_store.state_dict()
    torch.save({"model": model_state, "node_store": node_store_state}, path)


def load_full_model(model, path: str, map_location=None):
    """
    Load MLP + GNN from one file. Model must have same architecture as when saved.
    """
    if map_location is None:
        map_location = "cpu"
    data = torch.load(path, map_location=map_location)
    model.load_state_dict(data["model"], strict=False)
    model.gnn._node_store.load_state_dict(data["node_store"])
