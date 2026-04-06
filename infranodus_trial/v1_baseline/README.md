# V1 Baseline — Frozen 2026-03-31

Stable baseline of the InfraNodus-equivalent text network analysis engine.
Frozen before Phase 1 enhancements (TF-IDF, persistence, visualization, MCP).

## Run

```bash
python infranodus_trial/v1_baseline/run.py --output infranodus_trial/v1_baseline/output.json
```

## Baseline Stats (from graph_output.json)

- Nodes: 1,777
- Edges: 15,439
- Communities: 16
- Modularity: 0.4295 (strong structure)

## Files

- `engine.py` — Text network analysis engine (4-gram co-occurrence, Louvain, structural holes)
- `run.py` — CLI runner (loads learnings/ + results/, prints report)
- `trial.py` — Cloud InfraNodus API client (reference only)
- `graph_output.json` — Full graph JSON from baseline run
