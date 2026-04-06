"""
Graph persistence — save/load/search named knowledge graphs as JSON files.

Storage: infranodus_trial/graphs/{slug}.json
"""

import re
import json
from datetime import datetime
from pathlib import Path

GRAPHS_DIR = Path(__file__).parent / "graphs"


def _slugify(name: str) -> str:
    """Convert a graph name to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower().strip())
    return slug.strip("-") or "unnamed"


def save_graph(name: str, tn_dict: dict, metadata: dict = None) -> Path:
    """
    Save a TextNetwork (as dict from tn.to_dict()) to disk.

    Args:
        name: Human-readable graph name
        tn_dict: Output of TextNetwork.to_dict()
        metadata: Optional dict with source_files, corpus_chars, etc.

    Returns: Path to the saved JSON file.
    """
    GRAPHS_DIR.mkdir(exist_ok=True)

    record = {
        "name": name,
        "slug": _slugify(name),
        "created": datetime.now().isoformat(),
        "metadata": metadata or {},
        "graph": tn_dict,
    }

    path = GRAPHS_DIR / f"{_slugify(name)}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def load_graph(name: str) -> dict:
    """Load a saved graph record by name or slug."""
    slug = _slugify(name)
    path = GRAPHS_DIR / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"No graph named '{name}' (looked for {path})")
    return json.loads(path.read_text())


def list_graphs() -> list[dict]:
    """Return summary info for all saved graphs."""
    GRAPHS_DIR.mkdir(exist_ok=True)
    results = []
    for path in sorted(GRAPHS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            stats = data.get("graph", {}).get("stats", {})
            results.append({
                "name": data.get("name", path.stem),
                "slug": data.get("slug", path.stem),
                "created": data.get("created", ""),
                "nodes": stats.get("nodes", 0),
                "edges": stats.get("edges", 0),
                "communities": stats.get("communities", 0),
                "modularity": stats.get("modularity", 0),
                "file": str(path),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def search_graphs(keyword: str) -> list[dict]:
    """
    Search across all saved graphs for a keyword.
    Matches against: graph name, node names, community labels, metadata.
    Returns list of matches with context.
    """
    keyword_lower = keyword.lower()
    matches = []

    for path in sorted(GRAPHS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue

        graph_name = data.get("name", path.stem)
        graph_data = data.get("graph", {})

        # Search in graph name
        name_match = keyword_lower in graph_name.lower()

        # Search in node names
        node_metrics = graph_data.get("node_metrics", {})
        matching_nodes = [n for n in node_metrics if keyword_lower in n.lower()]

        # Search in community labels
        communities = graph_data.get("communities", [])
        matching_communities = [
            c for c in communities
            if keyword_lower in c.get("label", "").lower()
        ]

        # Search in metadata
        metadata = data.get("metadata", {})
        meta_match = keyword_lower in json.dumps(metadata).lower()

        if name_match or matching_nodes or matching_communities or meta_match:
            # Get node details for matches
            node_details = []
            for n in matching_nodes[:10]:
                m = node_metrics.get(n, {})
                node_details.append({
                    "node": n,
                    "betweenness": m.get("betweenness", 0),
                    "community": m.get("community", -1),
                })

            matches.append({
                "name": graph_name,
                "slug": data.get("slug", path.stem),
                "created": data.get("created", ""),
                "name_match": name_match,
                "matching_nodes": node_details,
                "matching_communities": [
                    {"id": c.get("id"), "label": c.get("label", "")}
                    for c in matching_communities
                ],
                "stats": graph_data.get("stats", {}),
            })

    return matches


def delete_graph(name: str) -> bool:
    """Delete a saved graph by name or slug."""
    slug = _slugify(name)
    path = GRAPHS_DIR / f"{slug}.json"
    if path.exists():
        path.unlink()
        return True
    return False
