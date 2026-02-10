# Experiment: Phase on radiation, magnitude on conduction

Activation is split by connection type:

- **Radiation** (similarity-based): only **phase** and activation strength propagate; magnitude is sent as zero.
- **Conduction** (direct edges): only **magnitude** and activation strength propagate; phase is sent as zero.

No changes are made to the original `core/` or `distributed/` code; this experiment adds separate modules and uses the same PyTorch-only stack (PytorchNodeStore, no Qdrant at runtime).

## Requirements

Same as the main package. From repo root:

```bash
pip install -r word-tree/fixed_io_nodes/requirements.txt
```

(Or install `qdrant-client` if you hit `ModuleNotFoundError` when importing `fixed_io_nodes`.)

## Run

From repo root (word-tree):

```bash
python fixed_io_nodes/experiments/phase_rad_mag_cond/run_experiment.py
```

Uses Iris (MLP → experiment GNN), in-process training, and `ExperimentGNNAdam`. Logs to `training_runs/phase_rad_mag_cond_experiment/loss.csv`.

**Plot loss vs step** (from repo root):

```bash
python fixed_io_nodes/experiments/phase_rad_mag_cond/plot_loss.py
```

Saves `training_runs/phase_rad_mag_cond_experiment/loss_plot.png` and shows the figure. Requires `pandas` and `matplotlib`. Optional: pass a custom CSV path as the first argument.

## Files

| File | Role |
|------|------|
| `gnn.py` | `PhaseRadMagConductionGNN`: subclasses `UnquantizedGNN`, overrides `one_step_forward` to mask phase/mag by connection type. |
| `layer.py` | `InProcessNeurographLayer`: builds the experiment GNN + gradient_sink + accumulator; forward/backward in main process. `create_experiment_layer(**kwargs)` same API as `create_gnn_layer`. |
| `optim.py` | `ExperimentGNNAdam`: discovers `InProcessNeurographLayer`, pushes grads to accumulator and steps (like `GNNAdam`). |
| `run_experiment.py` | Example: MLP + experiment layer, Iris, training loop, CSV logging. |
