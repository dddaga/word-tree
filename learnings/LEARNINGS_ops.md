# Training Stability, MPS, and Performance

## Training Stability

### Early stopping monitors train_loss, not val_loss
**Reason:** Val set only 3,925 samples across 10 classes. During GA search (15% subset),
val set ~580 samples — too small for stable val_loss signal. Train_loss more reliable.

### ReduceLROnPlateau: patience=10, factor=0.5
Halves LR when train_loss no improve for 10 epochs. `min_lr=1e-7` prevents full stop.

### FP16 vs FP32: no meaningful difference for this task
**Date:** 2026-03-26
FP32 12.00%, FP16 11.97%. Use FP16 for speed, not accuracy.

---

## Failed Experiments

### N_hidden=2048 crash: backward shape mismatch
**Date:** 2026-03-26
**Error:** `RuntimeError: shape mismatch: value tensor of shape [4233306] cannot be broadcast to indexing result of shape [5306017]`
**Root cause 1:** `strength.fill_diagonal_(0)` in `wave_routing.py` — in-place mutation on tensor in computation graph.
**Root cause 2:** `dists[mask]` boolean indexing — creates flat 1D tensor whose backward shape diverges from cdist [N,N] at large N on MPS.
**Fix 1:** `strength = strength * (1.0 - torch.eye(N_hidden, device=...))` (out-of-place)
**Fix 2:** Element-wise mask multiply instead of boolean indexing.

### N_hidden=1024 loss divergence (16720 → 125501)
**Date:** 2026-03-26
**Root cause:** `fill_diagonal_` in-place mutation corrupted gradients silently ~25 epochs before accumulation caused visible explosion.
**Lesson:** In-place ops on tracked tensors = silent autograd bugs. No immediate error but graph corruption accumulates.

### ProximityWave N=512 explosion at epoch 25
**Date:** 2026-03-26
**Root cause:** Topology rebuild at epoch 10 selected near-zero distance edges → `1/d → ∞`.
**Fix:** Min-distance guard in `build_knn_conn` + gradient clipping.

### metalcompute pip install failed (Python 3.14)
**Workaround:** Standalone Swift script using `MTLDevice.makeLibrary(source:)` directly.
**File:** `src/metal/run_sparse_bench.swift`

---

## MPS / Apple Silicon Specifics

### PYTORCH_ENABLE_MPS_FALLBACK=1 required
Some ops (cdist backward, segment_reduce) not implemented on MPS, silently fall back to CPU.
Without it they crash. Always set in experiment scripts.

### CSR sparse tensors: not supported on MPS
`torch.sparse_csr_tensor` ops fail on MPS. Use COO (scatter_add) or fixed fan-in (conn_idx gather).

### Boolean indexing backward: crashes at large N
`tensor[bool_mask]` creates flat 1D graph node. At large N on MPS, backward shape diverges.
Use element-wise `tensor * float_mask` instead.

### Fixed fan-in gather not bandwidth-limited by precision
FP16 gives no speedup over FP32 for `Z[:, conn_idx, :].sum(dim=2)`. Operation is
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
Replacing 3 sequential K=6 gather steps with 1 step K=12 (2-hop precomputed neighbours) = 1.65× faster.
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
Single model N=512: ~2-3GB RAM. Can run 5-6 parallel tmux sessions safely.
Data (store.h5): 1.34GB cached per process. Shared by OS page cache if same file.

---

## Known Failure Modes from GSD archive

### step222: MLP baseline trainer incompatibility
**Symptom:** MLP baselines broken — trainer not compatible with standard nn.Sequential MLP.
**Status:** Bug confirmed; CIFAR-10/MLP baselines blocked until fixed.
**Impact:** paper/baselines_needed.md gap — MLP comparison pending.

### step232: Broken gradient flow
**Symptom:** ΔW mechanism initial impl had broken gradient flow; mechanism appeared to fail.
**Fix:** Gradient flow repaired in step234 — ΔW proj (no AH) = 95.44% (+3.77pp). May replace AH.
**Lesson:** When novel mechanism underperforms, verify gradient flow before killing it.

### step66: kwarg bug
**Symptom:** Script used incorrect kwarg name; silently ran with wrong hyperparameter.
**Lesson:** Always print config dict at epoch 0 to verify all kwargs land correctly.

### step306 line 204: sum-on-tensors bug
**Symptom:** `sum()` called on list of tensors — produces scalar sum not tensor stack.
**Fix:** Use `torch.stack(...).sum(0)` or explicit loop with tensor accumulator.
**Lesson:** Python `sum()` on tensor lists silently does scalar accumulation.

---

## Environment Setup

### Python environments
- Mac Mini: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/d_env/bin/python3`
- Mac Studio: `/Users/admin/ml/dhiraj/qwen2_omni/testing/d_env/bin/python3`

### Required env vars (set at top of all experiment scripts)
```python
import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

### Device detection pattern
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

### MPS DataLoader requirements
- `num_workers=0` — spawn incompatible with MPS on macOS
- `pin_memory=False` — auto-detect or set explicitly

### PyTorch version gates
- GradScaler on MPS: requires PyTorch ≥2.3 (MPS inf-detection bug before 2.3)
- `torch.autocast('mps', ...)`: PyTorch ≥2.0
- CSR sparse tensors: NOT supported on MPS — use COO or fixed fan-in gather

---

## Script Smoke-Testing Protocol

Before dispatching any new script to Mac Studio:
1. Run locally 2 epochs: `d_env/bin/python3 scripts/SCRIPT.py --device cpu`
2. Verify config dict prints at epoch 0 (all kwargs correct)
3. Verify loss finite (not NaN/inf) after epoch 1
4. Verify checkpoint saves without error
5. Check no in-place tensor ops on tracked tensors (use `tensor * mask`, not `tensor[mask]`)
6. Confirm tmux session name unique: `tmux ls` before launch

## 2026-04-15 — Seed gather optimization: spatial precomputation

**Finding:** SGNNET `_seed()` creates [B, N_in, D] intermediate then gathers [B, N, K_in, D]. Since spatial_coords fixed, spatial sum per neuron precomputable at `__init__`. **Mathematical identity** — zero accuracy change, no hyperparameters.

**Implementation:** `model_smallworld.py` — added `self.spatial_sum = spatial[conn_in].sum(dim=1)` buffer. Rewrote `_seed()` to gather x[:, conn_in].sum() only then cat with precomputed spatial_sum.

**Benchmarks:**
- Seed FLOPs: 1.64M → 0.05M (16× reduction, N×K_in×D → N×K_in MACs)
- Memory: [B, N_in, D] + [B, N, K_in, D] eliminated → [B, N, K_in] only (14× less)
- Speed: CPU=8.6×, MPS=10×, CUDA=5.3× (all at B=128)
- T0 stability: 91.77% at 20ep — identical to pre-optimization

**Scripts:** `bench_step831_seed_opt_cuda.py` (CUDA validation), `train_step631_kin_sweep.py` (K_in sweep T1)

### 2026-04-15 — step631 K_in sweep T1 results
**Config:** N=2048 D=16 K_hh=2 K_iter=5, 75ep 50% data, Trainer (production). Tests minimum viable K_in after spatial precomputation.

| Config | K_in | best T1 | Δ_vs_ref | seed MACs | Advance? |
|--------|------|---------|----------|-----------|---------|
| Ref    | 25   | 94.01%  | —        | 51,200    | ref     |
| A_k15  | 15   | 93.66%  | −0.36pp  | 30,720    | **YES** |
| B_k10  | 10   | 92.84%  | −1.17pp  | 20,480    | NO      |
| C_k5   | 5    | 91.26%  | −2.75pp  | 10,240    | NO      |

**K_in=15 advances.** Compound seed reduction: 16× (precomputation) × 1.67× (K_in 25→15) = **26.7× total seed FLOP reduction** vs original. Routing (655K MACs) now dominates — K_iter reduction (step630) next lever.
Next: step632 K_in=15 T2 validation (150ep, 100% data).

### step406: soundfile missing dep crash
**Symptom:** `ERROR: pip install soundfile` — script crashed at extraction phase; slot freed immediately.
**Fix:** `d_env/bin/pip install soundfile` before launch.
**Lesson:** Audio cross-modal scripts need `soundfile`. Add to deps check: `transformers datasets h5py soundfile`.

## step522 Muon optimizer — BROKEN (2026-04-16)
Root cause: Muon crashes with 'lerp_() got NoneType' — some 2D params in matrix_params receive no gradient (not in computation path for SGNNET_AntiHebbian). zero_grad(set_to_none=False) doesn't fix. Need filter matrix_params to params that actually receive gradients after first backward, or use Nesterov-based Muon variant handling None grads.
AdamW baseline captured: Ref_adamw=95.67%, reaches 95% @ep21.
Fix needed: run one backward, check which params have non-None grad, use those as matrix_params.


## 2026-04-16 session 10 — Param count reconciliation
**Finding:** step199 actual trainable params = 34,976 (W_pos=32,928 + theta=2,048).
"67K" figure in CLAUDE.md / EXPERIMENT_QUEUE was STALE — measured before spatial precomputation refactoring removed learned seed projection weights.
Post-refactoring: seed connections structural (non-learnable indexing). Only W_pos and theta trainable.
Paper correction: params = 34,976 = 0.029% of VGG_FC (not 0.05%). Stronger claim.
Step851 MLP crossover also stronger: h=6 (150K params) = 4.29× SGNNET (not 2.22× if 67K used).


## 2026-04-17 — model_resonant.py W_phase=None crash (step860)
**Bug:** `SGNNET_Resonant` forward() calls `F.normalize(self.W_phase, dim=-1)` unconditionally at line ~185, but `W_phase=None` when `alpha_turing=0.0` (by design — phase inhibition disabled).
**Root cause:** Guard exists at init (`if alpha_turing != 0.0`) but not at use site.
**Fix:** `W_ph_norm = F.normalize(self.W_phase, dim=-1) if self.W_phase is not None else None`, then guard `_phase_inhibit` call.
**Also:** `make_model_k1` in step860 used `SGNNET_Resonant` (non-CUDA) instead of `SGNNET_Resonant_CUDA` + `SGNNET_AntiHebbian_CUDA` for CUDA device. Any model factory must branch on `DEVICE.type == "cuda"`.