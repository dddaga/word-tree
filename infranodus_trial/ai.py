"""
AI-powered features for text network analysis.

Two modes:
  - Claude API mode: if ANTHROPIC_API_KEY is set, calls API directly
  - Prompt-only mode: prints formatted prompt to stdout for manual paste
"""

import os
import json
import textwrap

try:
    import requests as _requests
except ImportError:
    _requests = None


def build_graph_context(tn, max_nodes: int = 30) -> str:
    """
    Build a compact text representation of the graph structure.
    Suitable for injecting into any LLM conversation as context.
    """
    lines = []
    lines.append("=== KNOWLEDGE GRAPH CONTEXT ===")
    lines.append(f"Nodes: {tn.G.number_of_nodes()}, Edges: {tn.G.number_of_edges()}, "
                 f"Modularity: {tn.mod:.4f}, Communities: {len(tn.communities)}")

    lines.append("\n--- TOPICAL CLUSTERS ---")
    total_w = sum(tn.community_weight(i) for i in range(len(tn.communities)))
    for i, comm in enumerate(tn.communities):
        if len(comm) < 2:
            continue
        cw = tn.community_weight(i)
        pct = 100 * cw / total_w if total_w > 0 else 0
        lines.append(f"[{i}] {pct:.1f}% — {tn.community_labels[i]}")

    lines.append("\n--- STRUCTURAL GAPS ---")
    for h in tn.holes[:10]:
        la = tn.community_labels[h["community_a"]][:35]
        lb = tn.community_labels[h["community_b"]][:35]
        density = h["density"]
        gap_type = "ZERO-CONNECTION" if h["cross_weight"] == 0 else f"density={density:.3f}"
        lines.append(f"[{h['community_a']}] {la}  <->  [{h['community_b']}] {lb}  ({gap_type})")

    lines.append("\n--- TOP BRIDGING CONCEPTS ---")
    for node, bc in tn.top_nodes(max_nodes, by="betweenness"):
        ci = tn.node_community.get(node, -1)
        lines.append(f"  {node} (BC={bc:.4f}, cluster={ci})")

    return "\n".join(lines)


def _build_questions_prompt(tn, n: int = 5) -> str:
    """Build prompt for research question generation."""
    context = build_graph_context(tn)

    prompt = f"""You are a research advisor analyzing a knowledge graph of experiment notes about Sparse Geometric Neural Networks (SGNNET).

{context}

Based on the STRUCTURAL GAPS above, generate exactly {n} research questions that would BRIDGE the identified gaps. Each question should:
1. Reference specific concepts from the disconnected clusters
2. Propose a concrete experiment or investigation that would connect them
3. Be actionable (a researcher could immediately design an experiment to answer it)

Format: Number each question. Be specific — use the actual terms from the graph."""

    return prompt


def _build_overview_prompt(tn) -> str:
    """Build prompt for topical overview generation."""
    context = build_graph_context(tn)

    prompt = f"""You are a research analyst. Based on this knowledge graph structure from SGNNET (Sparse Geometric Neural Network) experimental research notes, write a concise topical overview (3-5 paragraphs).

{context}

Your overview should:
1. Describe the main research themes (using the cluster labels)
2. Identify which areas are well-connected vs. isolated
3. Highlight the most important bridging concepts and what they connect
4. Point out the structural gaps as potential research opportunities
5. Be written as a research summary, not a description of graph metrics"""

    return prompt


def generate_research_questions(tn, n: int = 5) -> str:
    """
    Generate research questions bridging structural gaps.
    Uses Claude API if ANTHROPIC_API_KEY is set, otherwise returns the prompt.
    """
    prompt = _build_questions_prompt(tn, n)
    return _call_or_print(prompt, "RESEARCH QUESTIONS")


def generate_topical_overview(tn) -> str:
    """
    Generate a narrative overview of the knowledge graph.
    Uses Claude API if ANTHROPIC_API_KEY is set, otherwise returns the prompt.
    """
    prompt = _build_overview_prompt(tn)
    return _call_or_print(prompt, "TOPICAL OVERVIEW")


def _call_or_print(prompt: str, label: str) -> str:
    """Call Claude API if key is set, otherwise print the prompt for manual use."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if api_key and _requests:
        print(f"\n[Calling Claude API for {label}...]")
        try:
            resp = _requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            if resp.ok:
                data = resp.json()
                text = data["content"][0]["text"]
                print(f"\n{'─' * 60}")
                print(f"  {label}")
                print(f"{'─' * 60}")
                print(text)
                return text
            else:
                print(f"[API error {resp.status_code}]: {resp.text[:200]}")
                print("[Falling back to prompt-only mode]")
        except Exception as e:
            print(f"[API error: {e}]")
            print("[Falling back to prompt-only mode]")

    # Prompt-only mode
    print(f"\n{'─' * 60}")
    print(f"  {label} — PROMPT (paste into any LLM)")
    print(f"{'─' * 60}")
    print(prompt)
    return prompt
