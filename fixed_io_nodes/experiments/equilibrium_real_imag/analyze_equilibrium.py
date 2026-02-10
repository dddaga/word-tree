"""
Load activation log and report whether node activations reached equilibrium:
how many nodes stabilized and optionally at which step. No existing files modified.
Uses only stdlib (no pandas).
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict


def load_log(log_path: str):
    """Returns list of dicts with step, node_id, strength (float)."""
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["step"] = int(row["step"])
            row["node_id"] = row["node_id"]
            try:
                row["strength"] = float(row["strength"])
            except (ValueError, KeyError):
                row["strength"] = None
            rows.append(row)
    return rows


def analyze_equilibrium(
    rows,
    window: int = 10,
    strength_var_threshold: float = 1e-6,
    strength_max_diff_threshold: float = 1e-5,
):
    """
    For each node, check if strength stabilizes in the last `window` steps.
    Returns dict with equilibrium_observed, num_nodes_equilibrium, num_nodes_total, equilibrium_at_step.
    """
    by_node = defaultdict(list)
    for r in rows:
        if r.get("strength") is not None:
            by_node[r["node_id"]].append((r["step"], r["strength"]))

    node_ids = list(by_node.keys())
    if not node_ids:
        return {
            "equilibrium_observed": False,
            "num_nodes_equilibrium": 0,
            "num_nodes_total": 0,
            "equilibrium_at_step": {},
        }

    if window < 1:
        window = 1

    equilibrium_at_step = {}
    for nid in node_ids:
        seq = sorted(by_node[nid], key=lambda x: x[0])
        vals = [x[1] for x in seq]
        steps = [x[0] for x in seq]
        found = None
        for i in range(window, len(vals) + 1):
            w = vals[i - window : i]
            mean = sum(w) / len(w)
            var = sum((x - mean) ** 2 for x in w) / len(w)
            max_diff = abs(w[-1] - w[0]) if len(w) > 1 else 0
            if var < strength_var_threshold or max_diff < strength_max_diff_threshold:
                found = steps[i - 1]
                break
        if found is not None:
            equilibrium_at_step[nid] = found

    return {
        "equilibrium_observed": len(equilibrium_at_step) > 0,
        "num_nodes_equilibrium": len(equilibrium_at_step),
        "num_nodes_total": len(node_ids),
        "equilibrium_at_step": equilibrium_at_step,
    }


def main(
    log_path: str,
    window: int = 10,
    strength_var_threshold: float = 1e-6,
    strength_max_diff_threshold: float = 1e-5,
):
    rows = load_log(log_path)
    result = analyze_equilibrium(
        rows,
        window=window,
        strength_var_threshold=strength_var_threshold,
        strength_max_diff_threshold=strength_max_diff_threshold,
    )
    print("Equilibrium analysis")
    print("  Equilibrium observed:", result["equilibrium_observed"])
    print("  Nodes in equilibrium:", result["num_nodes_equilibrium"], "/", result["num_nodes_total"])
    if result["equilibrium_at_step"]:
        items = list(result["equilibrium_at_step"].items())[:10]
        print("  First equilibrium step per node (sample):", dict(items))
    return result


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not log_path:
        print("Usage: python -m fixed_io_nodes.experiments.equilibrium_real_imag.analyze_equilibrium <log_path>")
        print("Example: after run_equilibrium_experiment.py, pass runs/run_<timestamp>/activations.csv")
        sys.exit(1)
    main(log_path)
