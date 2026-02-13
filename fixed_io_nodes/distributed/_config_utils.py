"""
Config resolution: defaults, deep merge, and get_config.
Also builds node_store from a full config (get_node_store_from_config).
"""

import copy
import yaml

try:
    from ..main import get_qdrant_params
    from ..core import initialize_model_and_nodestore
except ImportError:
    from main import get_qdrant_params
    from core import initialize_model_and_nodestore

# Default config: only what the layer needs; no logging/training clutter (user defines those in their script).
# Required: input_nodes, output_nodes, vector_dim, lr, accumulation_steps — user must pass via kwargs or overrides (or load from path).
# No qdrant index parameters (main.get_qdrant_params supplies those).
DEFAULT_CONFIG = {
    "qdrant": {
        "url": "http://localhost:6333",
        "collection_name": "example_collection",
    },
    "graph": {
        # "total_nodes": 200,
        # "input_nodes": 14,
        # "output_nodes": 10,
        "cardinality": 5,
        "radiation_targets": 5,
    },
    "model": {
        # "vector_dim": 14,
        "phase_bins": 256,
        "mag_bins": 256,
        "gamma": 1.0,
        "iterations": 3,
        "radiation_similarity_threshold": 0.5,
        "temporal_decay": 0.9,
        "activation_threshold": 0.05,
        "dtype": "float32",
    },
    "system": {
        "device": "cuda",
        "random_seed": 42,
        "logging": {
            # "log_path": "training_runs/example_config/example_config.csv",
            # "save_interval_seconds": 10,
            # "fieldnames": ["worker_id", "loss", "class", "skipped"],
            # "tensorboard_dir": None,
            "verbose": False,
        },
    },
}

# Keys that must be present after merge; get_config() raises if any are missing.
REQUIRED_CONFIG_PATHS = (
    "graph.input_nodes",
    "graph.output_nodes",
    "graph.total_nodes",
    "model.vector_dim",
    "training.lr",
    "training.accumulation_steps",
)

# Flat kwarg name -> dotted path for get_config(**kwargs).
PARAM_MAP = {
    "input_nodes": "graph.input_nodes",
    "output_nodes": "graph.output_nodes",
    "total_nodes": "graph.total_nodes",
    "cardinality": "graph.cardinality",
    "radiation_targets": "graph.radiation_targets",
    "vector_dim": "model.vector_dim",
    "phase_bins": "model.phase_bins",
    "mag_bins": "model.mag_bins",
    "iterations": "model.iterations",
    "activation_threshold": "model.activation_threshold",
    "gamma": "model.gamma",
    "temporal_decay": "model.temporal_decay",
    "radiation_similarity_threshold": "model.radiation_similarity_threshold",
    "dtype": "model.dtype",
    "lr": "training.lr",
    "accumulation_steps": "training.accumulation_steps",
    "worker_count": "training.worker_count",
    "epochs": "training.epochs",
    "timeout": "training.timeout",
    "device": "system.device",
    "random_seed": "system.random_seed",
    "collection_name": "qdrant.collection_name",
}


def deep_merge(base, override):
    """Recursively merge override into base. Override values replace; lists are replaced."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _kwargs_to_nested(kwargs):
    """Build nested dict from flat kwargs using PARAM_MAP."""
    nested = {}
    for key, value in kwargs.items():
        if key not in PARAM_MAP:
            continue
        path = PARAM_MAP[key].split(".")
        d = nested
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = value
    return nested


def _validate_required(cfg):
    """Raise ValueError if any REQUIRED_CONFIG_PATHS are missing."""
    missing = []
    for path in REQUIRED_CONFIG_PATHS:
        d = cfg
        for key in path.split("."):
            d = d.get(key) if isinstance(d, dict) else None
            if d is None:
                missing.append(path)
                break
    if missing:
        raise ValueError(
            "Config missing required keys (pass via kwargs or overrides): " + ", ".join(missing)
        )


def get_config(path=None, overrides=None, **kwargs):
    """
    Return a full config dict: defaults, optionally loaded from path, then merged with overrides and kwargs.
    Requires input_nodes, output_nodes, vector_dim (from path, overrides, or kwargs).
    """
    if path is not None:
        with open(path, "r") as f:
            base = yaml.safe_load(f)
    else:
        base = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        base = deep_merge(base, overrides)
    if kwargs:
        base = deep_merge(base, _kwargs_to_nested(kwargs))
    _validate_required(base)
    return base


def get_node_store_from_config(cfg):
    """
    Build node_store from config dict using the same kwargs as distributed_training and workers.
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
        verbose=cfg["system"].get("verbose", False),
        dtype=cfg["model"].get("dtype", "float32"),
    )
    return node_store
