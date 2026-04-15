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
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from engine import TextNetwork, tfidf_reweight
import store
from viz import generate_viz
from ai import generate_research_questions, generate_topical_overview, build_graph_context

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
    # Analysis options
    p.add_argument("--no-results", action="store_true", help="Exclude result JSONs")
    p.add_argument("--no-queue", action="store_true", help="Exclude EXPERIMENT_QUEUE.md")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Louvain resolution — higher = more communities (default: 1.0)")
    p.add_argument("--window", type=int, default=4, help="Co-occurrence window size (default: 4)")
    p.add_argument("--top", type=int, default=25, help="Top N bridging concepts (default: 25)")
    p.add_argument("--tfidf", action="store_true", help="Enable TF-IDF edge reweighting")
    # Output
    p.add_argument("--output", metavar="FILE", help="Save full graph JSON to file")
    # Persistence
    p.add_argument("--save", metavar="NAME", help="Save graph with this name")
    p.add_argument("--load", metavar="NAME", help="Load a previously saved graph (skip analysis)")
    p.add_argument("--list", action="store_true", dest="list_graphs", help="List all saved graphs")
    p.add_argument("--search", metavar="KEYWORD", help="Search across saved graphs")
    # Node query
    p.add_argument("--statements", metavar="NODE", help="Show relations for a specific node")
    p.add_argument("--radius", type=int, default=1, help="Hop radius for --statements (default: 1)")
    # Visualization
    p.add_argument("--viz", metavar="FILE.html", help="Generate interactive HTML visualization")
    # AI features
    p.add_argument("--questions", action="store_true", help="Generate bridging research questions")
    p.add_argument("--overview", action="store_true", help="Generate topical overview")
    p.add_argument("--context", action="store_true", help="Print graph context for LLM injection")
    args = p.parse_args()

    # ── List saved graphs
    if args.list_graphs:
        graphs = store.list_graphs()
        if not graphs:
            print("No saved graphs yet. Use --save NAME to save one.")
            return
        print(f"\n{'─' * 60}")
        print(f"  SAVED GRAPHS ({len(graphs)})")
        print(f"{'─' * 60}")
        for g in graphs:
            print(f"  {g['name']:<30} nodes={g['nodes']:<6} mod={g['modularity']:.4f}  {g['created'][:10]}")
        return

    # ── Search saved graphs
    if args.search:
        matches = store.search_graphs(args.search)
        if not matches:
            print(f"No graphs contain '{args.search}'.")
            return
        print(f"\n{'─' * 60}")
        print(f"  SEARCH RESULTS for '{args.search}' ({len(matches)} graphs)")
        print(f"{'─' * 60}")
        for m in matches:
            print(f"\n  Graph: {m['name']}  ({m['stats'].get('nodes', '?')} nodes)")
            if m['matching_nodes']:
                print(f"    Matching nodes:")
                for n in m['matching_nodes'][:5]:
                    print(f"      · {n['node']}  BC={n['betweenness']:.4f}  cluster={n['community']}")
            if m['matching_communities']:
                for c in m['matching_communities']:
                    print(f"    Matching cluster [{c['id']}]: {c['label']}")
        return

    # ── Load existing graph
    if args.load:
        print(f"\n[Loading graph '{args.load}']")
        record = store.load_graph(args.load)
        tn = TextNetwork.from_dict(record["graph"])
        print(f"  graph: {tn.G.number_of_nodes()} nodes, {tn.G.number_of_edges()} edges")
        print(f"  created: {record.get('created', '?')}")

        if args.statements:
            _print_statements(tn, args.statements, args.radius)
            return

        tn.print_report(top_n=args.top)

        if args.viz:
            generate_viz(tn, args.viz)
        if args.context:
            print(build_graph_context(tn))
        if args.questions:
            generate_research_questions(tn)
        if args.overview:
            generate_topical_overview(tn)
        return

    # ── Build fresh analysis
    learnings = load_learnings(include_queue=not args.no_queue)
    results = [] if args.no_results else load_results()

    print(f"\n[Loading]")
    print(f"  {len(learnings)} learnings files: {[f for f, _ in learnings]}")
    print(f"  {len(results)} result JSONs converted to prose")

    corpus = assemble_corpus(learnings, results)
    print(f"  total corpus: {len(corpus):,} chars")

    # ── Build text network
    tfidf_label = " + TF-IDF" if args.tfidf else ""
    print(f"\n[Building text network]  window={args.window}  resolution={args.resolution}{tfidf_label}")
    tn = TextNetwork.from_text(corpus, window_size=args.window, resolution=args.resolution)

    # ── Optional TF-IDF reweighting (re-run community detection after)
    if args.tfidf:
        documents = [text for _, text in learnings] + [text for _, text in results]
        tfidf_reweight(tn.G, documents)
        # Re-run community detection on reweighted graph
        from engine import detect_communities, compute_betweenness, compute_degree, compute_modularity
        tn.communities = detect_communities(tn.G, resolution=args.resolution)
        tn.bc = compute_betweenness(tn.G)
        tn.degree = compute_degree(tn.G)
        tn.mod = compute_modularity(tn.G, tn.communities)
        from engine import compute_diversivity, label_community, find_structural_holes
        tn.diversivity = compute_diversivity(tn.bc, tn.degree)
        tn.node_community = {}
        for idx, comm in enumerate(tn.communities):
            for node in comm:
                tn.node_community[node] = idx
        tn.community_labels = [label_community(c, tn.bc, tn.degree) for c in tn.communities]
        tn.holes = find_structural_holes(tn.G, tn.communities)

    print(f"  graph: {tn.G.number_of_nodes()} nodes, {tn.G.number_of_edges()} edges")

    # ── Node query
    if args.statements:
        _print_statements(tn, args.statements, args.radius)
        return

    # ── Print full report
    tn.print_report(top_n=args.top)

    # ── Save outputs
    if args.output:
        save_graph_json(tn, args.output)

    if args.save:
        metadata = {
            "source_files": [f for f, _ in learnings] + [f for f, _ in results],
            "corpus_chars": len(corpus),
            "tfidf": args.tfidf,
            "resolution": args.resolution,
            "window": args.window,
        }
        path = store.save_graph(args.save, tn.to_dict(), metadata)
        print(f"Graph saved → {path}")

    # ── Visualization
    if args.viz:
        generate_viz(tn, args.viz)

    # ── AI features
    if args.context:
        print(build_graph_context(tn))
    if args.questions:
        generate_research_questions(tn)
    if args.overview:
        generate_topical_overview(tn)


def _print_statements(tn: TextNetwork, node: str, radius: int):
    """Print relations for a specific node."""
    relations = tn.get_statements(node, radius=radius)
    if not relations:
        print(f"  Node '{node}' not found in graph.")
        return
    print(f"\n{'─' * 60}")
    print(f"  RELATIONS for '{node}'  (radius={radius}, {len(relations)} edges)")
    print(f"{'─' * 60}")
    # Sort by weight descending
    for r in sorted(relations, key=lambda x: x["weight"], reverse=True)[:30]:
        cross = "→" if r["community_source"] != r["community_target"] else "·"
        print(f"  {cross} {r['source']:<25} — {r['target']:<25} w={r['weight']:<8} "
              f"[{r['community_source']}→{r['community_target']}]")


if __name__ == "__main__":
    main()
