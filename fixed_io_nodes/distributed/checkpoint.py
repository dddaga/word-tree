"""
Save/load full model (MLP + GNN) in a single file.
Expects model to have model.gnn._node_store (e.g. IrisGNNModel with DistributedNeurographLayer).
"""

import torch
from distributed.layer import DistributedNeurographLayer


def save_full_model(model, path: str):
    """
    Save MLP + GNN weights in one file. Automatically discovers all GNN layers to avoid duplication.
    """
    node_stores_state = {}
    for name, module in model.named_modules():
        if isinstance(module, DistributedNeurographLayer):
            node_stores_state[name] = module._node_store.get_custom_state()
            
    model_state = {
        k: v for k, v in model.state_dict().items()
        if not any(k.startswith(f"{name}._node_store") for name in node_stores_state.keys())
    }
    torch.save({"model": model_state, "node_stores": node_stores_state}, path)


def load_full_model(model, path: str, map_location=None):
    """
    Load MLP + GNN from one file. Model must have same architecture as when saved.
    """
    if map_location is None:
        map_location = "cpu"
    data = torch.load(path, map_location=map_location)
    model.load_state_dict(data["model"], strict=False)
    
    # Support the new multi-layer format
    if "node_stores" in data:
        for name, module in model.named_modules():
            if isinstance(module, DistributedNeurographLayer) and name in data["node_stores"]:
                module._node_store.load_custom_state(data["node_stores"][name])
                
    # Support backward compatibility for the old single-layer format
    elif "node_store" in data and hasattr(model, "gnn"):
        model.gnn._node_store.load_custom_state(data["node_store"])