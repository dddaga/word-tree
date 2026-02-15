# Run1: Iris + GA

- **Dataset**: Iris (4 features, 3 classes). Train/val split with fixed seed (80/20).
- **Search space**: `graph.cardinality`, `model.vector_dim`, `model.iterations`, `training.lr` (see `config.yaml` list-valued keys).
- **Entry**: From project root, `python genetic_run1.py` (or `python -m genetic_run1`). Writes per-candidate folders under `run1/<hash>/` and `best_configs.json` here.
