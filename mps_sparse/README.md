# mps_sparse

Efficient sparse matrix multiplication for PyTorch on Apple Silicon (MPS).

Two strategies are available via a single `strategy=` argument. Pick based on whether your network has dense layers.

---

## Which strategy to use

| Condition | Strategy | Expected speedup |
|---|---|---|
| Network has dense layers (CNN heads, classifiers, etc.) | `"auto"` (default) | 4–7x vs dense MPS at ≤0.1% density |
| Network is **all-sparse** (pure GNNs, sparse transformers) | `"scipy"` | 10–37x vs dense MPS at ≤0.1% density |
| Density > 1.5% | `"auto"` (falls back to dense GEMM) | ~1x (dense is optimal) |

**Rule of thumb:** If you never call `.to("mps")` on your activations, use `strategy="scipy"`. If you mix sparse and dense ops on MPS, use `strategy="auto"`.

**Why all-sparse networks benefit more from scipy:** One MPS context dispatch costs ~0.2 ms minimum — equal to running an entire sparse layer on CPU. With no dense layers, there is no reason to ever touch the GPU, and scipy CSR via Apple's Accelerate framework is faster for the sparse work itself.

---

## Installation

```bash
pip install scipy          # for strategy="scipy"
pip install -e .           # install mps_sparse in editable mode
```

---

## Quick start

### strategy="auto" — MPS (embedding_bag)

Best when your network mixes sparse and dense layers.

```python
import torch
import mps_sparse as ms

# Build once at startup
weight = torch.randn(4096, 4096) * (torch.rand(4096, 4096) < 0.001)
layer = ms.SparseLinear(weight, bias=False)           # strategy="auto" by default
layer = layer  # stays on CPU until you call .to("mps")

x = torch.randn(batch, 4096, device="mps")
out = layer(x)   # sparse matmul runs on MPS via embedding_bag
```

Or using `StaticSparseMatrix` directly:

```python
ssm = ms.StaticSparseMatrix.from_dense(weight, strategy="auto")
out = ssm @ x    # sparse(M,K) @ dense(K,N) → (M,N)
```

### strategy="scipy" — CPU (Apple Accelerate)

Best for all-sparse networks. Keeps everything on CPU — no MPS dispatch overhead.

```python
import torch
import mps_sparse as ms

# Build adjacency matrix at startup (one-time cost)
adj = ms.StaticSparseMatrix.from_adjacency(
    edge_index,          # shape (2, E), PyG convention
    num_nodes,
    num_nodes,
    edge_weights=weights,
    strategy="scipy",    # use scipy CSR on CPU via Apple Accelerate
    reorder=False,       # set True for +43% cache locality (see note below)
)

# Forward pass — all CPU, no MPS dispatch
x = torch.randn(num_nodes, D)   # keep on CPU
for _ in range(num_layers):
    x = adj @ x                 # scipy CSR @ CPU tensor → CPU tensor
    x = torch.relu(x)

# Move to MPS only if a final dense head needs it
# logits = dense_head(x.to("mps"))
```

### reorder=True — Reverse Cuthill-McKee

Adds ~43% additional speedup by reordering non-zeros for better CPU cache locality. **Important:** reordering permutes the matrix's row and column indices, so results are numerically different from the original matrix. This is a global preprocessing step — you must apply the same permutation to all node feature tensors.

```python
adj = ms.StaticSparseMatrix.from_adjacency(
    edge_index, N, N,
    edge_weights=weights,
    strategy="scipy",
    reorder=True,          # stores permutation in adj._rcm_perm
)

# Permute node features consistently
perm = adj._rcm_perm                          # RCM permutation tensor (N,)
x = x[perm]                                   # reorder input features
for _ in range(num_layers):
    x = adj @ x
    x = torch.relu(x)
x = x[torch.argsort(perm)]                   # restore original order if needed
```

If you train the whole network with reordering applied from the start, you never need to un-permute.

---

## Benchmarked speedups (5000×5000, D=64, Apple M-series)

| Approach | Per layer | 10-layer net |
|---|---|---|
| **scipy CPU, plain** (`strategy="scipy"`) | 0.24 ms | 2.4 ms |
| **scipy CPU + RCM** (`strategy="scipy", reorder=True`) | 0.16 ms | 1.6 ms |
| embedding_bag MPS (`strategy="auto"`) | 0.28 ms | 2.8 ms |
| Dense matmul MPS (baseline) | 5.9 ms | 59 ms |

At 0.1% density:
- `strategy="auto"` → **~21x** vs dense MPS
- `strategy="scipy"` → **~25x** vs dense MPS  
- `strategy="scipy", reorder=True` → **~37x** vs dense MPS

---

## Full API

### StaticSparseMatrix

```python
# From a dense tensor (zeros below threshold are dropped)
ssm = ms.StaticSparseMatrix.from_dense(
    dense,               # (M, K) float tensor
    threshold=0.0,       # |w| <= threshold treated as zero
    learnable=False,     # True → values are nn.Parameters
    strategy="auto",     # "auto" | "scipy"
    reorder=False,       # RCM reordering (scipy only)
)

# From a PyTorch sparse tensor
ssm = ms.StaticSparseMatrix.from_sparse_tensor(sparse, strategy="auto")

# From a graph edge list (PyG convention)
ssm = ms.StaticSparseMatrix.from_adjacency(
    edge_index,          # (2, E) long tensor
    num_nodes_src,
    num_nodes_dst,
    edge_weights=None,   # (E,) float, defaults to ones
    strategy="auto",
    reorder=False,
)

# Direct constructor
ssm = ms.StaticSparseMatrix(values, row_indices, col_indices, (M, K),
                             strategy="scipy", reorder=False)

# Matmul
out = ssm @ x            # sparse(M,K) @ dense(K,N) → (M,N)
out = x @ ssm            # dense(B,M) @ sparse(M,K) → (B,K)

# Properties
ssm.nnz                  # number of non-zeros
ssm.density              # float, nnz / (M*K)
ssm.shape                # (M, K)
ssm.t()                  # transposed view (cached)
ssm.to_dense()           # materialize as dense tensor
ssm.to("mps")            # move indices/values to device
```

### SparseLinear (nn.Module)

Drop-in replacement for `nn.Linear` with a sparse weight.

```python
layer = ms.SparseLinear(
    weight,              # (out, in) dense tensor or StaticSparseMatrix
    bias=True,
    threshold=0.0,
    learnable_values=True,
    strategy="auto",     # "auto" | "scipy"
    reorder=False,
)
out = layer(x)           # (..., in) → (..., out)
```

### SparseMatMul (nn.Module)

Fixed sparse projection (adjacency, Laplacian, etc.).

```python
module = ms.SparseMatMul(
    sparse,              # (M, K) dense tensor or StaticSparseMatrix
    mode="left",         # "left": sparse @ x, "right": x @ sparse
    strategy="auto",
    reorder=False,
)
out = module(x)
```

### Functional API

```python
# sparse(M,K) @ dense(K,N)
out = ms.sparse_mm(values, row_indices, col_indices, dense, M)

# dense(M,K) @ sparse(K,N)
out = ms.dense_sparse_mm(dense, values, row_indices, col_indices, N)
```

---

## Running the benchmark

```bash
cd mps_sparse
source ../.venv/bin/activate

# MPS vs scipy comparison across densities
python benchmark.py --sizes 1024 4096 5000 --densities 0.001 0.005 0.01 --device mps

# CPU-only (useful on machines without MPS)
python benchmark.py --sizes 1024 4096 --densities 0.001 0.01 --device cpu
```

The benchmark prints per-row timing for dense MPS baseline, embedding_bag MPS, and scipy CPU side by side, with speedups.

---

## Notes

- Gradients flow through both strategies — both are differentiable via `torch.autograd.Function`.
- `strategy="scipy"` always returns **CPU tensors**. Do not mix with MPS tensors in the same operation.
- Above ~1.5% density, `strategy="auto"` automatically falls back to a dense matmul (faster in that regime).
- MLX was evaluated and eliminated: no sparse ops in mlx 0.31.1, and dense MLX is slower than PyTorch MPS dense.
