# Phase 3: SGNNET Core Architecture - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the SGNNET nn.Module and all supporting components as standalone, tested Python modules. This phase is architecture-only — no training loop, no data loading from HDF5. Goal: forward pass produces valid output tensor, backward pass produces valid gradients, neuron positions (W) receive gradients during a toy training loop. All components are modular and independently testable.

**Note:** This architecture diverges from `sparse_geometric_network_report.md` in several key ways (see decisions below). The report is a starting point, not the final spec.

</domain>

<decisions>
## Implementation Decisions

### D-01: Geometric Dimensionality
**D=4 globally.** All neurons (hidden, output) live in [0,1]⁴. The four dimensions encode:
1. Value (activation signal)
2. Height (h_norm, 0→1)
3. Width (w_norm, 0→1)
4. Channel depth (channel_norm = channel_idx / 511, 0→1)

This is derived from the spatial structure of VGG16's pool5 output [512, 7, 7].

### D-02: Input Encoding — N_in=25088, No Adapter
**No adapter.** N_in = 25088 input neurons, one per element of the flattened [512, 7, 7] feature map.

For input neuron k (in the flattened [512, 7, 7] map):
```python
channel    = k // (7 * 7)          # 0..511
row        = (k % (7 * 7)) // 7   # 0..6
col        = (k % (7 * 7)) % 7    # 0..6
h_norm     = row / 6.0             # 0.0..1.0
w_norm     = col / 6.0             # 0.0..1.0
channel_norm = channel / 511.0    # 0.0..1.0

A_input[k] = [feature_val, h_norm, w_norm, channel_norm]  # dynamic, per sample
W_input[k] — NOT STORED: input neurons have no learnable W positions (see D-03)
```

This encodes both the VGG16 feature value AND spatial position (where in the 7×7 grid) AND channel depth. Neurons at the same spatial position but different channels are close in channel dimension but distinct.

### D-03: W Tensor Covers Hidden+Output Only
**W shape: [N_hidden + N_out, D=4].** Input neurons have no learnable positions — their spatial coordinates are precomputed constants used only to construct A_input. W is not defined for input neurons.

Input neurons are excluded from:
- Dynamic connectivity (they don't send or receive via cdist)
- Safety valve loss (only hidden+output neuron positions matter)
- Self-projection readout (only output neurons)

### D-04: N_hidden Is a Sweep Parameter
**Default N_hidden=256 for Phase 3 architecture.** Phase 4 sweeps N_hidden ∈ {64, 128, 256, 512} to find the minimum value that shows convergence trends. The architecture must accept N_hidden as a constructor parameter.

Total neurons for default config: N = 25088 (in) + 256 (hidden) + 10 (out) = 25354.

### D-05: Three C Matrices (Split Design)
**Replace the single C matrix from the report with three separate sparse matrices:**

| Matrix | Shape | Density | When Applied | Purpose |
|--------|-------|---------|-------------|---------|
| `C_input` | [N_in, N_hidden] | 10% | Once, before K iterations | Input→hidden seeding. Every input neuron guaranteed ≥1 connection. |
| `C_hh` | [N_hidden, N_hidden] | 10% | Iterations 1..K-1 | Hidden-hidden static connectivity. |
| `C_ho` | [N_hidden, N_out] | 10% | Final iteration K only | Hidden→output injection. |

All three use `nn.Parameter` values with registered `mask` buffers (pattern fixed at init, values learned).

**Parameter budget (N_hidden=256, 10% density):**
- C_input: 25088 × 256 × 10% ≈ 641K params
- C_hh: 256 × 256 × 10% ≈ 6.5K params
- C_ho: 256 × 10 × 10% ≈ 256 params
- W (hidden+output): 266 × 4 = 1,064 params
- LayerNorm: 2 × 4 = 8 params
- **Total: ~649K params = 0.52% of VGG16 FC** ✓

### D-06: Forward Pass Structure
**Three-phase forward pass:**

```
Phase 1 — Seeding (once):
  A_hidden_0 = relu(norm(A_input @ C_input))
  # A_input: [batch, N_in, D=4]
  # C_input: [N_in, N_hidden]
  # A_hidden_0: [batch, N_hidden, D=4]

Phase 2 — Hidden iterations (K-1 steps):
  for k in range(K - 1):
      static = A_hidden @ C_hh                        # [batch, N_hidden, D]
      dyn_contribution, gate = dynamic_connectivity(
          A_hidden, W_hidden, N_hidden, D, box_size
      )                                                # contribution: [batch, N_hidden, D]
      A_hidden = relu(norm(static + dyn_contribution))

Phase 3 — Output injection (step K):
  static_out = A_hidden @ C_ho                         # [batch, N_out, D]
  dyn_to_output, _ = dynamic_connectivity_to_output(
      A_hidden, W_out, N_hidden, D, box_size
  )                                                     # [batch, N_out, D]
  A_out = relu(static_out + dyn_to_output)             # NO norm at final step (optional)

Readout:
  W_norm = F.normalize(W_out, dim=-1)                  # [N_out, D]
  scores = (A_out * W_norm.unsqueeze(0)).sum(dim=-1)   # [batch, N_out]
```

### D-07: Output Neurons Are Pure Sinks
**Output neurons are inactive for iterations 1..K-1.** They activate ONLY at step K via:
1. Static signal from C_ho (hidden→output)
2. Dynamic signal: `cdist(A_hidden, W_out)` — hidden activations that fall within r* of output W positions

Output neurons never send — they don't appear as sources in C_hh or dynamic connectivity.

If an output neuron receives zero signal at step K, its activation = 0 and self-projection score = 0. Softmax over scores handles this safely (no NaN). Add `ε=1e-4` to logits before cross-entropy as numerical safety net.

**Phase 4 monitoring flag:** if >10% of training epochs show all output neurons dead (score ≈ 0 for most samples) → architecture problem, investigate N_hidden, K, or C_ho density.

### D-08: Load Balance Loss — Return Gate from dynamic_connectivity
**`dynamic_connectivity` returns `(activation_contribution, gate)` tuple:**
- `gate`: [N_hidden, N_hidden] Boolean tensor — `(dists < r*)` mask
- Training loop computes `gate.sum(dim=1)` for per-neuron selection counts
- `load_balance_loss(counts)` = `variance(counts.float() / counts.sum())`

### D-09: Initialization
- **W (hidden+output):** `torch.rand(N_hidden + N_out, D) * box_size` — random uniform in [0,1]⁴
- **C_input, C_hh, C_ho:** random pattern at init (fixed), values from `torch.randn * mask`
- **K-means init:** skipped for Phase 3. Can be added in Phase 4 if convergence is slow.

### D-10: Dynamic Connectivity Scope
- **Iterations 1..K-1:** `cdist(A_hidden, W_hidden)` — hidden-only geometry. W_hidden = W[:N_hidden].
- **Step K (output injection):** separate dynamic computation: `cdist(A_hidden, W_out)` — hidden activations routing to output neuron positions. W_out = W[N_hidden:].
- r* computation: uses only N_hidden (not total N) for the hidden-only iterations; recomputed for the hidden→output step if needed (same formula, same N_hidden).

### D-11: Normalization
**LayerNorm(D=4)** applied after relu at each hidden iteration step (as in report). Not applied at the final output injection step.

### Claude's Discretion
- C_input initialization: exact strategy for guaranteeing ≥1 connection per input neuron (e.g., one fixed random connection per input neuron + random sparsity on top)
- Whether to use `register_buffer` for C masks (same pattern as report: C_values is nn.Parameter, C_mask is registered buffer)
- Layer normalization implementation details (affine=True default)
- Exact ε value for logit safety net
- Test data shapes for unit tests (small N=10 synthetic inputs)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture Spec (Primary)
- `sparse_geometric_network_report.md` — Full SGNNET architecture spec. **READ CAREFULLY:** Sections 3.4–3.6 (recursive loop, dynamic connectivity, self-projection) and Section 7 (loss functions) are closest to the current design. BUT: this phase's architecture diverges significantly from the report — decisions above take precedence over the report where they conflict.

### Phase 3 Requirements
- `.planning/REQUIREMENTS.md` §SGNNET Architecture — ARCH-01 through ARCH-07 acceptance criteria
- `.planning/ROADMAP.md` §Phase 3 — Module structure and plan breakdown (Plans 3.1–3.4)

### Project Constraints
- `.planning/PROJECT.md` §Constraints — File size limit (250 lines/file), top-down code style, MPS hardware

### Prior Phase Artifacts (Reusable)
- `src/data/dataset.py` — IMAGENETTE_CLASSES, get_dataloader (useful for Phase 3 integration test)
- `src/data/extractor.py` — VGGExtractor (shows established MPS + num_workers=0 pattern)
- `src/data/store.py` — TensorStore (shows how to load feature vectors for toy training test)
- `src/utils/metrics.py` — compute_all_metrics, count_params (reused in Phase 4+)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/utils/metrics.py`: `count_params(model)` counts `model.classifier` params. For SGNNET, will need to sum `C_input`, `C_hh`, `C_ho`, and `W` parameters separately.
- `src/data/store.py`: `TensorStore.get_split("train")` returns (features, soft_labels, labels). Can use a small subset (e.g., 100 samples) as toy training data for Phase 3 backward pass test.
- `d_env/` virtual environment: sklearn, torch, torchvision, thop, h5py all installed.

### Established Patterns
- **MPS device:** `device = "mps" if torch.backends.mps.is_available() else "cpu"`. Always pass device explicitly; don't hardcode.
- **num_workers=0:** Required on macOS MPS for DataLoader (established in Phase 1).
- **PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0:** Required for large tensor operations on MPS (established in Phase 1).
- **File size ≤250 lines:** Split into modules. With 4 modules (geometry.py, model.py, losses.py, init.py), each should be manageable.
- **Top-down code style:** High-level class/function structure first, then fill in implementation details.

### Integration Points
- `src/sgnnet/` — new package, does not exist yet. Create `src/sgnnet/__init__.py` as package marker.
- Phase 4 will import from `src/sgnnet/model.py` (SGNNET class), `src/sgnnet/losses.py` (total_loss), and potentially `src/sgnnet/geometry.py` (for monitoring).
- The input encoding function (computing h_norm, w_norm, channel_norm for each of the 25088 input neurons) should live in `src/sgnnet/model.py` or `src/sgnnet/encoding.py` — NOT in `src/data/`.

</code_context>

<specifics>
## Specific Ideas

### Spatial Encoding Precomputation
The [h_norm, w_norm, channel_norm] for each of the 25088 input neurons is deterministic (depends only on index k). It should be precomputed once and stored as a buffer (not recomputed each forward pass):
```python
# In SGNNET.__init__:
spatial_coords = compute_spatial_encoding(N_in=25088)  # [25088, 3]
self.register_buffer('spatial_coords', spatial_coords)

# In forward: A_input = torch.cat([x.unsqueeze(-1), spatial_coords.unsqueeze(0).expand(B,-1,-1)], dim=-1)
# x: [batch, N_in] → unsqueeze to [batch, N_in, 1]; spatial_coords: [N_in, 3] → [batch, N_in, 3]
# Result: [batch, N_in, D=4]
```

### Dynamic Connectivity — Two Call Patterns Needed
Phase 3 needs two variants:
1. `dynamic_connectivity_hh(A_hidden, W_hidden, ...)` → used in iterations 1..K-1, returns (contribution, gate) with gate [N_h, N_h]
2. `dynamic_connectivity_ho(A_hidden, W_out, ...)` → used at step K, returns (contribution_to_out,) with no gate needed (load balance only tracks hidden-hidden routing)

Or a single function with a `return_gate` parameter.

### Parameter Budget Must Be Verified in Plan 3.4
After implementing Plans 3.1–3.3, Plan 3.4 must verify:
- `sum(p.numel() for p in model.parameters())` ≤ 1,236,428 (1% of VGG16 FC 123,642,856)
- With N_hidden=256: expected ~649K params — well within budget
- With N_hidden=512: expected ~1.29M params — slightly over, Phase 4 may need to tune

</specifics>

<deferred>
## Deferred Ideas

- **Soft gate (differentiable routing):** Replace hard gate `(dists < r*)` with `σ(-(dist - r*)/τ)`. Mentioned as a dead neuron alternative. Deferred to v2 (per the report, Section 10.1).
- **Expanding radius warmup:** r*_effective = r* × warmup_factor in early Phase 4 epochs. Deferred to Phase 4 if dead neurons observed.
- **K-means initialization of hidden/output neuron positions.** Random init sufficient for Phase 3. Add in Phase 4 if convergence is slow.
- **Adaptive K (stop when activations converge).** v2 feature per the report.
- **Input neuron W positions.** Currently not used (input neurons excluded from geometry). Could add a precomputed W_input buffer for visualization/analysis purposes in Phase 6.

None — discussion stayed within Phase 3 scope.

</deferred>

---

*Phase: 03-sgnnet-core-architecture*
*Context gathered: 2026-03-24*
