"""
Helpers to build node_store and related objects from config.
Used by DistributedNeurographLayer so the layer can own its own node_store and accumulator.
"""

from main import get_qdrant_params
from core import initialize_model_and_nodestore


def get_node_store_from_config(cfg):
    """
    Build node_store from config dict using the same kwargs as distributed_training and workers.
    Does not modify main or core; calls their existing APIs.
    """
    qdrant_params = get_qdrant_params(cfg) or {}
    _, node_store = initialize_model_and_nodestore(
        qdrant_url=cfg["qdrant"]["url"],
        collection_name=cfg["qdrant"]["collection_name"],
        total_nodes=cfg["graph"]["total_nodes"],
        input_nodes=cfg["graph"]["input_nodes"],
        output_nodes=cfg["graph"]["output_nodes"],
        cardinality=cfg["graph"]["cardinality"],
        radiation_targets=cfg["graph"]["radiation_targets"],
        vector_dim=cfg["model"]["vector_dim"],
        phase_bins=cfg["model"].get("phase_bins") or 256,
        mag_bins=cfg["model"].get("mag_bins") or 256,
        iterations=cfg["model"]["iterations"],
        activation_threshold=cfg["model"]["activation_threshold"],
        gamma=cfg["model"]["gamma"],
        device=cfg["system"]["device"],
        temporal_decay=cfg["model"].get("temporal_decay", 1.0),
        radiation_similarity_threshold=cfg["model"].get("radiation_similarity_threshold", 0.0),
        qdrant_params=qdrant_params,
    )
    return node_store
