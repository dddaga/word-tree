"""
Resolve GA individual to full config; inject paths for candidate dir; config hash.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    d = config
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value


def resolve_config(individual: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge individual (flat dict of dotted_key -> single value) into a deep copy of base_config.
    Returns full resolved config with single values throughout.
    """
    resolved = copy.deepcopy(base_config)
    for key, value in individual.items():
        _set_nested(resolved, key, value)
    return resolved


def _canonical_json(config: dict) -> str:
    """Deterministic JSON for hashing (sorted keys, no extra whitespace)."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_hash(resolved_config: Dict[str, Any]) -> str:
    """SHA256 hash of resolved config for folder naming and reuse check."""
    return hashlib.sha256(_canonical_json(resolved_config).encode()).hexdigest()[:16]


def inject_paths(
    resolved_config: Dict[str, Any],
    candidate_dir: str,
    run_name: str = "ga",
) -> Dict[str, Any]:
    """
    Set log_path, tensorboard_dir, weights path, and collection_name under candidate_dir.
    Modifies resolved_config in place and returns it. Uses candidate_dir basename for collection_name.
    """
    candidate_path = Path(candidate_dir)
    candidate_path.mkdir(parents=True, exist_ok=True)
    h = candidate_path.name

    resolved_config.setdefault("system", {})
    resolved_config["system"].setdefault("logging", {})
    resolved_config["system"]["logging"]["log_path"] = str(candidate_path / "loss.csv")
    resolved_config["system"]["logging"]["tensorboard_dir"] = str(candidate_path / "tensorboard")
    resolved_config["system"]["weights_save_path"] = str(candidate_path / "weights.pt")

    resolved_config.setdefault("qdrant", {})
    resolved_config["qdrant"]["collection_name"] = f"{run_name}_{h}"
    return resolved_config
