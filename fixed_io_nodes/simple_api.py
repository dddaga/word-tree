"""
Simple API for beginners: create a GNN layer with minimal kwargs; rest use defaults.
"""

from torch import nn

from distributed._config_utils import get_config
from distributed import DistributedNeurographLayer


def create_gnn_layer(**kwargs) -> nn.Module:
    """Build DistributedNeurographLayer with merged config from defaults + kwargs.

    Required (pass every time): input_nodes, output_nodes, vector_dim, lr, accumulation_steps.
    All other kwargs are optional and use defaults. Use GNNAdam (from distributed) to train, not torch.optim.Adam.

    Graph:
        input_nodes: Number of input nodes (required).
        output_nodes: Number of output nodes (required).
        total_nodes: Total nodes in the graph (default 200).
        cardinality: Graph cardinality (default 5).
        radiation_targets: Number of radiation targets (default 5).

    Model:
        vector_dim: Node vector dimension (required).
        phase_bins: Phase quantization bins; unused in unquantized mode (default 256).
        mag_bins: Magnitude quantization bins; unused in unquantized mode (default 256).
        iterations: GNN message-passing iterations (default 3).
        activation_threshold: Activation threshold (default 0.05).
        gamma: Gamma for signal/activation (default 1.0).
        temporal_decay: Temporal decay factor (default 0.9).
        radiation_similarity_threshold: Radiation similarity threshold (default 0.5).
        dtype: "float32" or "float16" (default "float32").

    Training (required for layer; epochs/batch size are for your own loop):
        lr: Learning rate (required).
        accumulation_steps: Gradient accumulation steps (required).

    System:
        device: "cuda" or "cpu" (default "cuda").
        random_seed: Random seed (default 42).

    Storage:
        collection_name: Qdrant collection name for node store (default "example_collection").
    """
    cfg = get_config(**kwargs)
    return DistributedNeurographLayer(cfg)
