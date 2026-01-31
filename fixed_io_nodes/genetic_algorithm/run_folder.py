"""
Per-candidate folder: get path, check if done, read/write results.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .config_resolver import config_hash


def get_candidate_dir(run_dir: str, resolved_config: Dict[str, Any]) -> str:
    """Return run_dir/<hash>; create dir if needed."""
    h = config_hash(resolved_config)
    path = Path(run_dir) / h
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def is_training_done(candidate_dir: str) -> bool:
    """True if results.json exists with validation_accuracy key."""
    p = Path(candidate_dir) / "results.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return "validation_accuracy" in d
    except (json.JSONDecodeError, KeyError):
        return False


def read_results(candidate_dir: str) -> Dict[str, Any]:
    """Load results.json; return dict with validation_accuracy, validation_loss."""
    p = Path(candidate_dir) / "results.json"
    with open(p) as f:
        return json.load(f)


def write_results(candidate_dir: str, results_dict: Dict[str, Any]) -> None:
    """Write results.json."""
    p = Path(candidate_dir) / "results.json"
    with open(p, "w") as f:
        json.dump(results_dict, f, indent=2)
