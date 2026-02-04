"""
Load GA run config (list-valued search space) and split into search_space and base_config.
"""

import yaml
from pathlib import Path


def _is_list_of_scalars(val):
    if not isinstance(val, list) or len(val) == 0:
        return False
    return all(
        v is None or isinstance(v, (bool, int, float, str))
        for v in val
    )


def _collect_search_space(config, prefix=""):
    out = {}
    for key, val in config.items():
        if not prefix and key == "gene_expression":
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and not _is_list_of_scalars(val):
            out.update(_collect_search_space(val, path))
        elif _is_list_of_scalars(val):
            out[path] = val
    return out


def load_run_config(config_path):
    """
    Load run config YAML. Returns (base_config, search_space).
    base_config: full nested dict. search_space: flat dict dotted_key -> list of values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        base_config = yaml.safe_load(f)
    if not base_config:
        raise ValueError(f"Empty config: {config_path}")
    search_space = _collect_search_space(base_config)
    return base_config, search_space
