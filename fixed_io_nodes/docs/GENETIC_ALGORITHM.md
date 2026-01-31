# Genetic Algorithm for Hyperparameter Tuning

Single reference for the GA implementation: overview, layout, config, model, weights, package, flow, and how to run.

## Overview

The GA tunes GNN hyperparameters via evolutionary search. It lives in:

- **`genetic_algorithm/`** — Package: config loading, resolving, run folders, fitness evaluation, GA loop.
- **`genetic_algorithm_runs/`** — One folder per run (e.g. `run1`); each run has a list-valued `config.yaml` (search space) and `README.md`.

Training uses the **distributed_training** flow: **IrisGNNModel** (MLP + tanh + GNN), **GNNAdam**, DataLoader, CSV + TensorBoard. No Qdrant; PyTorch NodeStore only.

## Directory layout

- **`genetic_algorithm_runs/<run_name>/`**
  - **`config.yaml`** — Same structure as training configs; **list-valued** keys = search space, single values = fixed.
  - **`README.md`** — Short description of the run (dataset, goal).
  - **`<hash>/`** — One folder per evaluated config (hash = SHA256 of resolved config, truncated).
    - **`config.yaml`** — Resolved config (single values) used for this training.
    - **`weights.pt`** — Single file: MLP + GNN weights (see Weights below).
    - **`loss.csv`** — Training loss log.
    - **`tensorboard/`** — TensorBoard logs.
    - **`results.json`** — `{"validation_accuracy": float, "validation_loss": float}`.
  - **`best_configs.json`** — Top-k configs from the GA run (written at end of `tuner.run()`).
  - **`log.txt`** — Full run log (stdout/stderr from the GA process, including training output). Tee’d so it also appears in the terminal.

If `results.json` exists for a config hash, that config is not re-run; fitness is read from the file.

## Config format

- Run config mirrors `training_runs` YAML (e.g. `qdrant`, `graph`, `model`, `training`, `system`).
- Any **list-valued** leaf (e.g. `vector_dim: [4, 8, 14]`) is a search-space dimension; other keys are fixed.
- Resolved config per candidate has single values everywhere; paths (`log_path`, `tensorboard_dir`, `weights_save_path`, `qdrant.collection_name`) are set by `config_resolver.inject_paths()` under `run_dir/<hash>/`.

## Model architecture

- **IrisGNNModel** (used by GA and `distributed_training.py`):
  - **MLP**: `nn.Linear(4, input_nodes * vector_dim)` — input 4 (Iris), output `input_nodes * vector_dim`.
  - **Activation**: `tanh`.
  - **Reshape**: `h.view(B, input_nodes, vector_dim)` into GNN.
  - **GNN**: `DistributedNeurographLayer(cfg)` (unchanged).
  - **Output**: Logits (CrossEntropyLoss; no softmax in forward).

`input_nodes` and `vector_dim` come from resolved config (`graph.input_nodes`, `model.vector_dim`).

## Weights

- **One file per candidate**: `weights.pt` contains **MLP + GNN** in one place.
- **NodeStore**: `PytorchNodeStore` has **`state_dict()`** and **`load_state_dict(state_dict)`**; **`save_weights`** / **`load_weights`** use them.
- **Full model**: **`distributed.checkpoint.save_full_model(model, path)`** and **`load_full_model(model, path)`** save/load `{"model": model_part_state_dict, "node_store": model.gnn._node_store.state_dict()}`. Model part excludes `gnn._node_store` keys to avoid duplication.
- Training (fitness) calls **`save_full_model`** at the end; validation calls **`load_full_model`** before evaluating on the val set.

## Package layout (`genetic_algorithm/`)

| Module | Role |
|--------|------|
| **config_loader** | Load run `config.yaml`; return `(base_config, search_space)` (list-valued keys → dotted_key → list). |
| **config_resolver** | `resolve_config(individual, base_config)` → full config; `inject_paths(resolved, candidate_dir, run_name)`; `config_hash(resolved)` for folder name. |
| **run_folder** | `get_candidate_dir(run_dir, resolved_config)`; `is_training_done(candidate_dir)`; `read_results` / `write_results`. |
| **fitness** | `evaluate_fitness(run_dir, resolved_config, get_train_val_datasets, run_name)` — if done, return cached accuracy; else train, validate, write `results.json`, return accuracy. |
| **ga** | **GeneticTuner**: `generate_individual`, `evaluate_fitness`, `select_top_k`, **`run(run_dir, get_train_val_datasets, run_name)`** — generations, crossover, mutation, elitism; returns top-k and writes `best_configs.json`. |

**Entry for run1**: **`genetic_run1.py`** — loads `genetic_algorithm_runs/run1/config.yaml`, defines **`get_train_val_datasets()`** (Iris 80/20 split, fixed seed), instantiates **GeneticTuner**, calls **`tuner.run(run_dir="genetic_algorithm_runs/run1", ...)`**.

## Flow

1. Load run config → `base_config`, `search_space`.
2. **GeneticTuner.run()**: for each generation, evaluate fitness for each individual (resolve config → candidate dir → if `results.json` exists use it, else train + validate, write results).
3. Select elites, crossover, mutate → next generation.
4. Return top-k; save **`best_configs.json`** under run dir.

## How to run

From the **fixed_io_nodes** directory (project root for this codebase):

```bash
python genetic_run1.py
```

This uses `genetic_algorithm_runs/run1/config.yaml` and writes outputs under `genetic_algorithm_runs/run1/`. Adjust `generations`, `population_size`, etc. in `genetic_run1.py` as needed.

## Modifications to existing code

- **`core/nodestore.py`** (PytorchNodeStore): Added **`state_dict()`** and **`load_state_dict(state_dict)`**; **`save_weights`** / **`load_weights`** refactored to use them.
- **`distributed/checkpoint.py`** (new): **`save_full_model(model, path)`** and **`load_full_model(model, path)`** for IrisGNNModel (MLP + GNN in one file).
- **`distributed_training.py`**: **IrisGNNModel** — MLP size set to `input_nodes * vector_dim` from config; reshape to `(B, input_nodes, vector_dim)`.
