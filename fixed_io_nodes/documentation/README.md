# fixed_io_nodes — Documentation

Sparse graph neural network replacing VGG16's dense MLP FC layers for imagenette-10 classification. The central architecture is `NativeNeurographLayer`: a message-passing GNN where nodes hold complex-valued state vectors, routing is softmax-weighted by activation strength, and weights are trained with a per-node Adam accumulator.

## Module map

```
fixed_io_nodes/
├── native/               # Active single-process implementation (use this)
│   ├── layer.py          # NativeNeurographLayer — main GNN forward pass
│   ├── node_store.py     # NativeNodeStore — graph topology + nn.Parameters
│   ├── optimizer.py      # NativeGNNOptimizer — per-node Adam accumulator
│   └── checkpoint.py     # save_full_model / load_full_model
├── core/                 # Shared math kernels
│   ├── custom_functions.py   # update_activations, activation_strength_forward
│   └── nodestore.py          # SimpleCosineSearch (cosine-similarity search)
├── distributed/          # Multi-process variant (not active — kept for reference)
├── vgg_training/         # VGG16 + GNN training pipeline
│   ├── training.py           # Shared training script (runs 1–9)
│   ├── training_runs/runN/   # Per-run config.yaml + training_runN.py (runs 10+)
│   ├── data/                 # imagenette_train_data.pt, imagenette_val_data.pt
│   ├── learnings/            # Experiment audit trail + concept wiki
│   │   ├── EXPERIMENT_QUEUE.md
│   │   ├── LEARNINGS_p*.md
│   │   └── concepts/         # Per-concept wiki pages
│   ├── gradient_starvation_analysis.py   # Diagnostic tool — grad flow by node
│   └── diagnose.ipynb        # Interactive activation/weight-delta visualisation
├── documentation/        # This directory — code-level reference
├── experiments/          # Older standalone experiments (not part of active pipeline)
└── viz/                  # Visualisation helpers (notebook + server)
```

## Quick start — run an experiment

```bash
# 1. Create training_runs/runN/config.yaml and training_runs/runN/training_runN.py
# 2. Add entry to vgg_training/learnings/EXPERIMENT_QUEUE.md
# 3. Launch in tmux (one run at a time on Mac Mini):
cd /Volumes/T9/IndraAstra/sudarshan/word-tree/fixed_io_nodes/vgg_training
tmux new-session -d -s runN \
  '/Volumes/T9/IndraAstra/sudarshan/.venv/bin/python -u \
   training_runs/runN/training_runN.py training_runs/runN/config.yaml \
   2>&1 | tee training_runs/runN/train.log'

# Monitor:
tmux attach -t runN
```

See `documentation/training_guide.md` for the full pipeline and experiment standards.

## Key documents

| What | Where |
|---|---|
| Architecture deep-dive | `documentation/architecture.md` |
| All config parameters | `documentation/architecture.md#config-reference` |
| Training pipeline & standards | `documentation/training_guide.md` |
| Gradient flow & temperature param | `documentation/gradient_flow.md` |
| Experiment history | `vgg_training/learnings/EXPERIMENT_QUEUE.md` |
| Concept wiki | `vgg_training/learnings/concepts/` |
