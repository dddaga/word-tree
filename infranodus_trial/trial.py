"""
InfraNodus trial — feed SGNNET learnings into knowledge graph API.
Usage:
    INFRANODUS_API_KEY=your_key python scripts/infranodus_trial.py

Optional flags:
    --files   comma-separated list of learnings files (default: all learnings/*.md)
    --save    save the graph to your InfraNodus account (default: dry-run, no save)
    --output  path to dump full JSON response (default: prints summary only)
"""

import os
import sys
import json
import glob
import argparse
import requests

API_URL = "https://infranodus.com/api/v1/graphAndStatements"


def load_learnings(file_paths):
    parts = []
    for path in file_paths:
        with open(path) as f:
            content = f.read().strip()
        label = os.path.basename(path).replace(".md", "")
        parts.append(f"## {label}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def call_infranodus(text, api_key, save=False):
    payload = {
        "name": "sgnnet-learnings",
        "text": text,
        "aiTopics": True,
        "doNotSave": not save,
        "addStats": True,
        "includeStatements": False,  # keep response small for trial
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def print_summary(data):
    # Topical clusters
    clusters = data.get("clusters") or data.get("topics") or []
    if clusters:
        print("\n=== TOPICAL CLUSTERS ===")
        for i, c in enumerate(clusters, 1):
            name = c.get("label") or c.get("name") or c.get("id", f"cluster-{i}")
            pct = c.get("percentage") or c.get("weight") or ""
            pct_str = f"  ({pct}%)" if pct else ""
            print(f"  {i}. {name}{pct_str}")

    # Structural gaps
    gaps = data.get("gaps") or data.get("structuralHoles") or []
    if gaps:
        print("\n=== STRUCTURAL GAPS (disconnected concept pairs) ===")
        for g in gaps[:10]:
            a = g.get("nodeA") or g.get("source") or ""
            b = g.get("nodeB") or g.get("target") or ""
            if a and b:
                print(f"  · {a}  ←gap→  {b}")

    # Research questions
    questions = data.get("researchQuestions") or data.get("questions") or []
    if questions:
        print("\n=== BRIDGING RESEARCH QUESTIONS ===")
        for q in questions[:8]:
            text = q if isinstance(q, str) else q.get("question") or q.get("text") or str(q)
            print(f"  ? {text}")

    # Top influential nodes
    nodes = data.get("nodes") or []
    if nodes:
        sorted_nodes = sorted(nodes, key=lambda n: n.get("betweenness") or n.get("influence") or 0, reverse=True)
        print("\n=== TOP BRIDGING CONCEPTS (by betweenness centrality) ===")
        for n in sorted_nodes[:15]:
            label = n.get("label") or n.get("id") or ""
            score = n.get("betweenness") or n.get("influence") or ""
            score_str = f"  [{score:.4f}]" if isinstance(score, float) else ""
            print(f"  · {label}{score_str}")

    # Stats
    stats = data.get("stats") or {}
    if stats:
        print("\n=== GRAPH STATS ===")
        for k in ("nodes", "edges", "modularity", "diversity", "communities"):
            if k in stats:
                print(f"  {k}: {stats[k]}")

    # Fallback: dump top-level keys if nothing matched
    if not clusters and not gaps and not questions and not nodes:
        print("\n[Raw top-level keys in response]")
        for k, v in data.items():
            print(f"  {k}: {str(v)[:120]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", default=None, help="Comma-separated file paths")
    parser.add_argument("--save", action="store_true", help="Save graph to InfraNodus account")
    parser.add_argument("--output", default=None, help="Dump full JSON response to this path")
    args = parser.parse_args()

    api_key = os.environ.get("INFRANODUS_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set INFRANODUS_API_KEY env var first.")
        sys.exit(1)

    if args.files:
        file_paths = [p.strip() for p in args.files.split(",")]
    else:
        # Default: all learnings/*.md files
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_paths = sorted(glob.glob(os.path.join(base, "learnings", "*.md")))
        # Exclude EXPERIMENT_QUEUE (structured table, not prose)
        file_paths = [p for p in file_paths if "EXPERIMENT_QUEUE" not in p]

    print(f"Loading {len(file_paths)} files:")
    for p in file_paths:
        print(f"  {os.path.basename(p)}")

    text = load_learnings(file_paths)
    print(f"\nTotal chars: {len(text):,}")
    print("Calling InfraNodus API...")

    data = call_infranodus(text, api_key, save=args.save)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Full response saved to {args.output}")

    print_summary(data)


if __name__ == "__main__":
    main()
