#!/usr/bin/env python3
"""
SGNNET Knowledge Graph Analyzer — local InfraNodus-equivalent.

Loads ALL files from learnings/ (including EXPERIMENT_QUEUE) + converts
results/train_step*.json to natural language prose, then runs full text
network analysis: concept clusters, structural gaps, bridging concepts.

Usage:
    python infranodus_trial/run.py

Options:
    --no-results     Exclude experiment result JSONs
    --no-queue       Exclude EXPERIMENT_QUEUE.md
    --resolution N   Louvain resolution (default 1.0; higher = more/smaller communities)
    --window N       Co-occurrence window size (default 4)
    --output FILE    Save full graph JSON to file (nodes, edges, communities, holes)
    --top N          Show top N bridging concepts (default 25)
"""

import re
import sys
import json
import argparse
from pathlib import Path

# Allow running from repo root or from infranodus_trial/
ROOT = Path(__file__).parent.parent.parent  # v1_baseline → infranodus_trial → repo root
sys.path.insert(0, str(Path(__file__).parent))

from engine import TextNetwork

BASE = ROOT

GLOBAL_REF_ACC = 0.2922  # step9A baseline


# ─── Data converters ──────────────────────────────────────────────────────────

def step_title(path: Path) -> str:
    """train_step16_inhibition_mechanisms.json → 'Step 16 Inhibition Mechanisms'"""
    m = re.match(r"train_step(\d+)_(.*)", path.stem)
    if m:
        return f"Step {m.group(1)} {m.group(2).replace('_', ' ').title()}"
    return path.stem.replace("_", " ").title()


def result_json_to_prose(path: Path) -> str:
    """
    Convert experiment result JSON to natural language.
    Strips top1_history (pure numerical noise for text analysis).
    Produces one sentence per variant describing what was tested and outcome.
    """
    try:
        data = json.loads(path.read_text())
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""

    title = step_title(path)

    ref = data.get("Ref") or data.get("ref")
    ref_acc = ref.get("top1_best", GLOBAL_REF_ACC) if ref else GLOBAL_REF_ACC

    lines = [f"Experiment {title}."]

    if ref:
        lines.append(
            f"Reference baseline {ref.get('label', 'Ref')} achieved {ref_acc:.2%} accuracy "
            f"at epoch {ref.get('best_epoch', '?')} out of {ref.get('epochs_run', '?')} epochs."
        )

    for vkey, v in data.items():
        if not isinstance(v, dict):
            continue
        if vkey.lower() == "ref":
            continue

        acc = v.get("top1_best")
        if acc is None:
            continue

        label = v.get("label", vkey)
        delta = acc - ref_acc
        sign = "+" if delta >= 0 else ""

        config = v.get("_meta", {}).get("config") or {}
        skip = {"D", "N", "epochs", "seed", "device"}
        cfg_parts = [
            f"{k} {val}" for k, val in config.items()
            if k not in skip and not isinstance(val, (list, dict))
        ]

        if abs(delta) <= 0.005:
            verdict = "neutral result no meaningful change"
        elif delta >= 0.08:
            verdict = "large gain strong winner included in next generation"
        elif delta >= 0.03:
            verdict = "solid gain candidate winner"
        elif delta >= 0.005:
            verdict = "modest gain"
        elif delta <= -0.05:
            verdict = "significant regression rejected"
        else:
            verdict = "mild regression"

        sentence = (
            f"Variant {vkey} tested {label} and reached {acc:.2%} accuracy "
            f"{sign}{delta:.2%} versus reference at epoch "
            f"{v.get('best_epoch', '?')} of {v.get('epochs_run', '?')}. "
            f"Verdict {verdict}."
        )
        if cfg_parts:
            sentence += f" Parameters {' '.join(cfg_parts)}."
        lines.append(sentence)

    return " ".join(lines)


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_learnings(include_queue: bool = True) -> list[tuple[str, str]]:
    """
    Returns list of (filename, text) for all learnings/*.md files.
    All files are included — the engine handles structural analysis.
    """
    files = sorted((BASE / "learnings").glob("*.md"))
    if not include_queue:
        files = [f for f in files if "EXPERIMENT_QUEUE" not in f.name]
    return [(f.name, f.read_text(errors='replace').strip()) for f in files]


def load_results() -> list[tuple[str, str]]:
    """Load all train_step*.json and convert to prose."""
    files = sorted((BASE / "results").glob("train_step*.json"))
    out = []
    for f in files:
        prose = result_json_to_prose(f)
        if prose.strip():
            out.append((f.name, prose))
    return out


# ─── Text assembly ────────────────────────────────────────────────────────────

def assemble_corpus(
    learnings: list[tuple[str, str]],
    results: list[tuple[str, str]],
) -> str:
    """
    Concatenate all sources into one corpus with clear section labels.
    Section labels themselves become nodes if repeated enough — they anchor clusters.
    """
    parts = []

    if learnings:
        parts.append("SECTION RESEARCH LEARNINGS AND EXPERIMENTAL FINDINGS")
        for fname, text in learnings:
            # Use filename as a section marker (underscores become spaces → separate tokens)
            section = fname.replace(".md", "").replace("_", " ")
            parts.append(f"DOCUMENT {section}")
            parts.append(text)

    if results:
        parts.append("SECTION EXPERIMENT RESULTS NUMERICAL OUTCOMES")
        for fname, text in results:
            section = fname.replace(".json", "").replace("_", " ")
            parts.append(f"DOCUMENT {section}")
            parts.append(text)

    return "\n\n".join(parts)


# ─── Output helpers ───────────────────────────────────────────────────────────

def save_graph_json(tn: TextNetwork, path: str):
    """Save graph structure to JSON for external inspection."""
    data = {
        "stats": {
            "nodes": tn.G.number_of_nodes(),
            "edges": tn.G.number_of_edges(),
            "communities": len(tn.communities),
            "modularity": round(tn.mod, 4),
        },
        "communities": [
            {
                "id": i,
                "label": tn.community_labels[i],
                "size": len(c),
                "members": sorted(c),
            }
            for i, c in enumerate(tn.communities)
        ],
        "top_nodes": [
            {
                "term": node,
                "betweenness": round(bc, 5),
                "degree": round(tn.degree.get(node, 0), 1),
                "diversivity": round(tn.diversivity.get(node, 0), 5),
                "community": tn.node_community.get(node, -1),
            }
            for node, bc in tn.top_nodes(50, by="betweenness")
        ],
        "structural_holes": [
            {
                "community_a": h["community_a"],
                "label_a": tn.community_labels[h["community_a"]],
                "community_b": h["community_b"],
                "label_b": tn.community_labels[h["community_b"]],
                "density": round(h["density"], 5),
                "cross_weight": h["cross_weight"],
            }
            for h in tn.holes
        ],
        "edges_sample": [
            {"a": a, "b": b, "weight": round(d["weight"], 1)}
            for a, b, d in sorted(
                tn.G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True
            )[:100]
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"Graph saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SGNNET text network analyzer (local InfraNodus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--no-results", action="store_true", help="Exclude result JSONs")
    p.add_argument("--no-queue", action="store_true", help="Exclude EXPERIMENT_QUEUE.md")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Louvain resolution — higher = more communities (default: 1.0)")
    p.add_argument("--window", type=int, default=4, help="Co-occurrence window size (default: 4)")
    p.add_argument("--output", metavar="FILE", help="Save full graph JSON to file")
    p.add_argument("--top", type=int, default=25, help="Top N bridging concepts (default: 25)")
    args = p.parse_args()

    # ── Load data
    learnings = load_learnings(include_queue=not args.no_queue)
    results = [] if args.no_results else load_results()

    print(f"\n[Loading]")
    print(f"  {len(learnings)} learnings files: {[f for f, _ in learnings]}")
    print(f"  {len(results)} result JSONs converted to prose")

    corpus = assemble_corpus(learnings, results)
    print(f"  total corpus: {len(corpus):,} chars")

    # ── Build text network
    print(f"\n[Building text network]  window={args.window}  resolution={args.resolution}")
    tn = TextNetwork.from_text(corpus, window_size=args.window, resolution=args.resolution)
    print(f"  graph: {tn.G.number_of_nodes()} nodes, {tn.G.number_of_edges()} edges")

    # ── Print full report
    tn.print_report(top_n=args.top)

    # ── Optionally save
    if args.output:
        save_graph_json(tn, args.output)


if __name__ == "__main__":
    main()
