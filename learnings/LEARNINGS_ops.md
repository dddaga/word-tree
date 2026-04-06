# Training Stability, MPS, and Performance

## Training Stability

### Early stopping monitors train_loss, not val_loss
**Reason:** Validation set is only 3,925 samples across 10 classes. During GA search (15% subset),
val set is ~580 samples — too small for stable val_loss signal. Train_loss is more reliable.

### ReduceLROnPlateau: patience=10, factor=0.5
Halves LR when train_loss doesn't improve for 10 epochs. `min_lr=1e-7` prevents full stop.

### FP16 vs FP32: no meaningful difference for this task
**Date:** 2026-03-26
FP32 12.00%, FP16 11.97%. Use FP16 for speed, not accuracy.

---

## Failed Experiments

### N_hidden=2048 crash: backward shape mismatch
**Date:** 2026-03-26
**Error:** `RuntimeError: shape mismatch: value tensor of shape [4233306] cannot be broadcast to indexing result of shape [5306017]`
**Root cause 1:** `strength.fill_diagonal_(0)` in `wave_routing.py` — in-place mutation on a tensor in the computation graph.
**Root cause 2:** `dists[mask]` boolean indexing — creates a flat 1D tensor whose backward shape diverges from cdist [N,N] at large N on MPS.
**Fix 1:** `strength = strength * (1.0 - torch.eye(N_hidden, device=...))` (out-of-place)
**Fix 2:** Element-wise mask multiply instead of boolean indexing.

### N_hidden=1024 loss divergence (16720 → 125501)
**Date:** 2026-03-26
**Root cause:** The `fill_diagonal_` in-place mutation corrupted gradients silently for ~25 epochs before accumulation caused visible explosion.
**Lesson:** In-place operations on tracked tensors are silent bugs in autograd — they don't error immediately but accumulate graph corruption over time.

### ProximityWave N=512 explosion at epoch 25
**Date:** 2026-03-26
**Root cause:** Topology rebuild at epoch 10 selected near-zero distance edges → `1/d → ∞`.
**Fix:** Min-distance guard in `build_knn_conn` + gradient clipping.

### metalcompute pip install failed (Python 3.14)
**Workaround:** Write a standalone Swift script using `MTLDevice.makeLibrary(source:)` directly.
**File:** `src/metal/run_sparse_bench.swift`

---

## MPS / Apple Silicon Specifics

### PYTORCH_ENABLE_MPS_FALLBACK=1 is required
Some ops (cdist backward, segment_reduce) are not implemented on MPS and silently fall back to CPU.
Without it they crash. Always set in experiment scripts.

### CSR sparse tensors: not supported on MPS
`torch.sparse_csr_tensor` operations fail on MPS. Use COO (scatter_add) or fixed fan-in (conn_idx gather) instead.

### Boolean indexing backward: crashes at large N
`tensor[bool_mask]` creates a flat 1D graph node. At large N on MPS, its backward shape diverges.
Use element-wise `tensor * float_mask` instead.

### Fixed fan-in gather is not bandwidth-limited by precision
FP16 gives no speedup over FP32 for `Z[:, conn_idx, :].sum(dim=2)`. The operation is
random-access memory (cache-miss bound), not sequential bandwidth.

### MPS sparse algorithm benchmarks (N_in=25088, N_h=256, B=64)
| Method | Time | Notes |
|---|---|---|
| Dense einsum | 3.92 ms | Baseline |
| Fixed fan-in K=100 | 2.33 ms | ~1.7× faster |
| Grouped block-sparse | 0.49 ms | **8× faster** |
| Metal naive_fanin | 0.50 ms | Custom shader |
| COO scatter_add | 124 ms | 37× slower — avoid |
| CSR | N/A | Not supported on MPS |

---

## Performance Benchmarks

### Forward pass scaling (C_hh routing, K=6, B=64, D=4, 3 iterations)
| N | FP32 ms | Notes |
|---|---|---|
| 256 | 0.37 | — |
| 1024 | 1.54 | — |
| 4096 | 5.20 | — |
| 10000 | 12.71 | ~1.3 ms per 1000 neurons |

### Precomputed 2-hop adjacency: 1.65× faster
Replacing 3 sequential K=6 gather steps with 1 step of K=12 (2-hop precomputed neighbours) is 1.65× faster.
Only valid for inference — training needs sequential steps for gradient flow through iterations.

### Mac Studio (M2 Ultra, 256GB RAM) training speed
| Config | Time/epoch | Notes |
|---|---|---|
| N=512 D=4  120ep | ~0.7s/ep | ~87s total |
| N=512 D=16 120ep | ~3-5s/ep | ~360-600s total (depends on routing mode) |
| N=1024 D=16 120ep | ~8-12s/ep | est. 1000-1500s total |
| N=2048 D=16 120ep | ~30-40s/ep | est. 3600-5000s total |

### Parallel training capacity
Mac Studio 256GB RAM: 22.8GB active, 244GB free (2026-03-29).
Single model at N=512: ~2-3GB RAM. Can safely run 5-6 parallel tmux sessions.
Data (store.h5): 1.34GB cached in RAM per process. Shared by OS page cache if same file.
