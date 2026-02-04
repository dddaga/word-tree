"""
fixed_io_nodes: NeuroGraph GNN layer and training utilities.

Use from parent directory (e.g. word-tree):
    import sys; sys.path.insert(0, ".")
    from fixed_io_nodes import create_gnn_layer, get_config, GNNAdam

Or from inside fixed_io_nodes: from distributed import ... ; from simple_api import ...

Simple usage:
    from fixed_io_nodes import create_gnn_layer, get_config, GNNAdam
    layer = create_gnn_layer(input_nodes=14, output_nodes=10, vector_dim=14, lr=0.001, accumulation_steps=8)
    optimizer = GNNAdam(model, lr=0.001)
"""

from torch import nn

from .distributed import (
    get_config,
    DistributedNeurographLayer,
    DistributedNeurographStack,
    GNNAdam,
    save_full_model,
    load_full_model,
)


def create_gnn_layer(**kwargs) -> nn.Module:
    """Build GNN layer with merged config. Required: input_nodes, output_nodes, vector_dim, lr, accumulation_steps."""
    return DistributedNeurographLayer(get_config(**kwargs))


__all__ = [
    "create_gnn_layer",
    "get_config",
    "DistributedNeurographLayer",
    "DistributedNeurographStack",
    "GNNAdam",
    "save_full_model",
    "load_full_model",
]
