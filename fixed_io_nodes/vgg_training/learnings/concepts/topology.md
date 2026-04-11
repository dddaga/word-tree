# Concept: Graph Topology (flat vs. layered)

## What it is
Controls how edges are assigned between nodes at graph initialization in `native/node_store.py`.

**flat** (default): Any node can connect to any other node. Output nodes receive edges directly from input nodes, intermediate nodes, or other outputs.

**layered**: Output nodes connect only from intermediate nodes. Intermediate nodes connect from any node (including inputs and other intermediates). Input→output direct edges are disallowed.

Config: `graph.topology: "layered"` (or absent/"flat" for flat).

## Parameter impact
In flat topology with cardinality=200, output nodes receive ~41 direct edges from input nodes (~20% of cardinality, by probability). In layered topology, this is 0.

## Results

| Run | Topology | Iterations | Other | Val best | Epochs done | Notes |
|-----|----------|------------|-------|----------|-------------|-------|
| run6 | flat | 5 | dropout=0.2, no radiation | **85.81%** | 40/40 | Best result; monotonically improving |
| run7 | layered | 7 | temporal_decay=0.8 | 23.08% (ep9) | 16 (stopped) | Confounded (3 vars); peaked early, declined |
| run8 | layered | 15 | beam_width=500, CPU | 13.32% | 23 (stopped) | Highly confounded |
| run9 | layered | 7 | beam_width=0 | 25.45% (ep29) | 38/40 | Nearest-controlled vs run6; erratic curve |

## Key finding
HYPOTHESIS (2026-04-05, run9 post-mortem): Layered topology severely underperforms flat (25.45% vs 85.81%, 60pp gap). Most likely cause: output nodes have no direct edges from inputs, creating a mandatory 2-hop path. This may starve output nodes of gradient signal early in training, preventing them from learning class-discriminative patterns.

Evidence: erratic, non-monotonic training curve in run9 (flat was monotonically increasing). Run7 also peaked early (epoch 9) and declined. Activation flow analysis in `diagnose.ipynb`.

**Status: HYPOTHESIS. Needs controlled diagnostic experiment.**

## Diagnostic plan
1. Verify in-degree distribution of output nodes (flat vs layered) — `diagnose.ipynb` already does this
2. Track per-iteration activation strength of output nodes — `_iter_stats_hook` added for this
3. If gradient starvation confirmed: experiment with "partially layered" topology (allow some input→output direct edges) or warm-start layered runs from flat weights

## Cross-references
- `training_runs/run9/notes.md` — run9 post-mortem
- `native/node_store.py` — topology implementation
- `diagnose.ipynb` — activation flow visualization
