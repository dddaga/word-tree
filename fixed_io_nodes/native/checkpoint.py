import torch
import torch.nn as nn
from .layer import NativeNeurographLayer


def save_full_model(model: nn.Module, path: str) -> None:
    """
    Save model state_dict (includes GNN nn.Parameters) + graph topology.
    """
    topology = {}
    for name, module in model.named_modules():
        if isinstance(module, NativeNeurographLayer):
            topology[name] = module._node_store.get_custom_state()

    torch.save({
        "model_state_dict": model.state_dict(),
        "topology": topology,
    }, path)


def load_full_model(model: nn.Module, path: str, map_location=None) -> None:
    """
    Load model weights and graph topology.
    """
    data = torch.load(path, map_location=map_location or "cpu", weights_only=False)

    if "model_state_dict" not in data:
        print(f"Warning: checkpoint at {path} is not in native format, skipping load")
        return

    model.load_state_dict(data["model_state_dict"], strict=False)

    if "topology" in data:
        for name, module in model.named_modules():
            if isinstance(module, NativeNeurographLayer) and name in data["topology"]:
                module._node_store.load_custom_state(data["topology"][name])
