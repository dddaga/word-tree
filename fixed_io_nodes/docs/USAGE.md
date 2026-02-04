# Usage: Simple model with MLP before and GNN after

This example builds a small model: **MLP → GNN**, suitable for tabular or small feature inputs (e.g. Iris). You pass all arguments as **kwargs** (no nested dicts); required ones must be given, the rest use defaults. **Logging, epochs, and batch size** are defined by you in your training script, not by the layer config.

## Required when creating the layer

You must pass (as kwargs):

- **input_nodes**, **output_nodes**, **vector_dim** (model shape)
- **lr**, **accumulation_steps** (needed by the GNN layer’s internal optimizer)

## Important: use GNNAdam, not torch.optim.Adam

The GNN layer manages its own parameters and gradient sink. You **must** use **`GNNAdam`** to train models that contain this layer. `torch.optim.Adam` will not update the GNN parameters correctly. See `distributed_training.py` for the canonical example.

---

## Imports

Run your script from **outside** `fixed_io_nodes` (e.g. from `word-tree`), so the **parent** of `fixed_io_nodes` is on `sys.path`. Then:

```python
from fixed_io_nodes import create_gnn_layer, get_config, GNNAdam
```

---

## 1. Create the GNN layer (simple API)

Pass all arguments as **kwargs** (no nested dictionaries):

```python
from fixed_io_nodes import create_gnn_layer

# Required: input_nodes, output_nodes, vector_dim, lr, accumulation_steps
gnn = create_gnn_layer(
    input_nodes=14,
    output_nodes=10,
    vector_dim=14,
    lr=0.001,
    accumulation_steps=8,
)
```

There are optional kwargs which can be passed. See docstring of `create_gnn_layer` for more info

```python
gnn = create_gnn_layer(
    input_nodes=14,
    output_nodes=10,
    vector_dim=14,
    lr=0.001,
    accumulation_steps=8,
)
```


## 2. Wrap GNN with an MLP (e.g. for 4-feature input like Iris)

```python
import torch
from torch import nn
from fixed_io_nodes import create_gnn_layer

class MLPGNNModel(nn.Module):
    """MLP maps input features to (batch, input_nodes, vector_dim); GNN consumes that."""

    def __init__(self, input_dim=4, input_nodes=14, output_nodes=10, vector_dim=14, lr=0.001, accumulation_steps=8, **gnn_kwargs):
        super().__init__()
        self.input_nodes = input_nodes
        self.vector_dim = vector_dim
        self.linear = nn.Linear(input_dim, input_nodes * vector_dim)
        self.tanh = nn.Tanh()
        self.gnn = create_gnn_layer(
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            vector_dim=vector_dim,
            lr=lr,
            accumulation_steps=accumulation_steps,
            **gnn_kwargs,
        )

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(1) if x.dim() > 2 else x
        h = self.tanh(self.linear(x))
        h = h.view(B, self.input_nodes, self.vector_dim)
        return self.gnn(h)
```

## 3. Training loop: use GNNAdam and your own logging

```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from fixed_io_nodes import GNNAdam, create_gnn_layer

#this block is necessary because this uses multi-processing
if __name__=="__main__:
    mp.set_start_method("spawn", force=True)

    model = MLPGNNModel(
        input_dim=4,
        input_nodes=14,
        output_nodes=10,
        vector_dim=14,
        lr=0.001,
        accumulation_steps=8,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Required: GNNAdam (not torch.optim.Adam) so GNN parameters are updated
    optimizer = GNNAdam(model, lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    dataset = ...  # your Dataset
    batch_size = 8  # your choice; can use accumulation_steps as batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    epochs = 10  # your choice

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        # Your logging here (TensorBoard, CSV, etc.)
```

When you are done training, shut down the layer’s worker pool (e.g. before process exit):

```python
try:
    model.gnn.shutdown()
except Exception:
    pass
```

See `distributed_training.py` for a full runnable example (Iris, TensorBoard, CSV logging).

## 4. Summary

| What | Use |
|------|-----|
| **Imports** | Run from outside `fixed_io_nodes`; `from fixed_io_nodes import create_gnn_layer, get_config, GNNAdam` |
| **Arguments** | Always pass as **kwargs** (e.g. `iterations=5`, `device="cpu"`); no nested dicts |
| Create layer | `create_gnn_layer(input_nodes=..., output_nodes=..., vector_dim=..., lr=..., accumulation_steps=...)` — these five required; any other options as kwargs |
| Config for custom wrapper | `get_config(...)` with the same kwargs |
| MLP before GNN | `Linear(in_dim, input_nodes * vector_dim)` → reshape → GNN |
| **Optimizer** | **`GNNAdam`** (do not use `torch.optim.Adam`) |
| Epochs, batch size, logging | Defined by you in your training script; not part of the layer config |
