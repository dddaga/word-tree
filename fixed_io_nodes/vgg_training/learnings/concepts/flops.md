# FLOPs & Parameter Counting for NativeNeurographLayer

Line-by-line derivation from `core/custom_functions.py` and `native/layer.py`. No profiling library can instrument this model (no nn.Linear/nn.Conv inside the GNN), so this manual formula is the only accurate method.

## Notation

| Symbol | Meaning | Typical values |
|--------|---------|---------------|
| B | batch size | 4 |
| N | total_nodes | 15454 (run10), 4146 (run11) |
| V | vector_dim | 8 |
| I | iterations | 5 |
| C | cardinality (edges per node) | 200 |
| n_in | input_nodes | 3136 |
| n_out | output_nodes | 256 (runs 1-9), 10 (run10+) |

---

# Part 1: Parameter Count

## Learnable parameters

| Component | Parameter | Shape | Formula | Source |
|-----------|-----------|-------|---------|--------|
| NativeNodeStore | phase_weight | (N, V) | N * V | `node_store.py` nn.Parameter |
| NativeNodeStore | mag_weight | (N, V) | N * V | `node_store.py` nn.Parameter |
| LayerNorm (if enabled) | weight (gamma) | (V,) | V | `layer.py` nn.LayerNorm |
| LayerNorm (if enabled) | bias (beta) | (V,) | V | `layer.py` nn.LayerNorm |
| FFN head (runs 1-9 only) | weight | (n_out, 10) | n_out * 10 | `training.py` nn.Linear |
| FFN head (runs 1-9 only) | bias | (10,) | 10 | `training.py` nn.Linear |

## Parameter formulas

**With FFN (runs 1-9):**
```
Params = 2*N*V + [2*V if LayerNorm else 0] + n_out*10 + 10
```

**Without FFN (run10+):**
```
Params = 2*N*V + [2*V if LayerNorm else 0]
```

## Worked examples

| Run | N | V | LayerNorm | FFN (n_out) | Calculation | Total Params |
|-----|-------|---|-----------|-------------|-------------|-------------|
| run1 | 15454 | 8 | No | 256 | 2*15454*8 + 256*10 + 10 | **249,842** |
| run2 | 30000 | 8 | No | 256 | 2*30000*8 + 256*10 + 10 | **482,570** |
| run3-6 | 15454 | 8 | Yes | 256 | 2*15454*8 + 16 + 256*10 + 10 | **249,858** |
| run7-9 | 15454 | 8 | Yes | 256 | 2*15454*8 + 16 + 256*10 + 10 | **249,858** |
| run10 | 15454 | 8 | Yes | None | 2*15454*8 + 16 | **247,280** |
| run11 | 4146 | 8 | Yes | None | 2*4146*8 + 16 | **66,352** |
| run12 | 15454 | 8 | Yes | None | 2*15454*8 + 16 | **247,280** |
| run13 | 15454 | 8 | Yes | None | 2*15454*8 + 16 | **247,280** |

Note: The FFN adds only ~1% of total parameters. The GNN node weights dominate.

---

# Part 2: FLOP Count

## Conventions

- 1 FLOP = 1 floating-point operation (add, mul, div, comparison)
- Transcendentals (exp, log, sin, cos, atan2, tanh): 1 FLOP each
- scatter_reduce/scatter_add: 1 FLOP per element
- Indexing/gathering/reshaping: 0 FLOPs
- We count forward pass only; backward ≈ 3x forward (with grad checkpointing)

## The forward pass has 4 phases

### Phase 1: Initialization

| Operation | Code location | FLOPs |
|-----------|--------------|-------|
| `tanh(x) * pi` | layer.py:111 | B * n_in * V * 2 |
| Weight replication (clone+repeat) | layer.py:126-127 | 0 |
| LayerNorm on mag_act | layer.py:133 | B*N * 5*V |
| `activation_strength_forward` | layer.py:134 | B*N * 3*V |
| w_real = mag * cos(phase) | layer.py:175 | B*N * 2*V |
| w_imag = mag * sin(phase) | layer.py:176 | B*N * 2*V |

`activation_strength_forward` detail (custom_functions.py:57-59):
- `exp(mags)`: M*V
- `exp(mags) * cos(phases)`: M*V (cos) + M*V (mul)
- `.sum(dim=-1)`: M*(V-1) ≈ M*V
- **Total: M * 3V**

**Phase 1 total: B*n_in*V*2 + B*N*12*V** (with LayerNorm)

### Phase 2: Input Injection

One `update_activations` call with E_inj = B*n_in, M_inj = 2*B*n_in.
Plus `activation_strength_forward` for virtual nodes: B*n_in * 3V.

### Phase 3: Iterative Message Passing — (I-1) iterations

This is the computational bottleneck (~99% of total FLOPs).

#### `update_activations(E_b edges, M nodes, V dims)` — one call

**Step 1: Softmax routing** (custom_functions.py:103-106)
```python
max_act = scatter_reduce_(amax)        # E_b comparisons
exp((src - max[dest]) / T)             # E_b * 3 (sub + div + exp)
sum_exp = scatter_add_(exp)            # E_b adds
routing = exp / (sum[dest] + eps)      # E_b * 2 (add + div)
```
→ **7 * E_b FLOPs** (scalar per edge, no V dimension)

**Step 2: Complex superposition** (custom_functions.py:117-122)
```python
exp(source_mag)                        # E_b * V
routing * exp_mag                      # E_b * V
cos(source_phase)                      # E_b * V
weighted * cos  →  source_real         # E_b * V
sin(source_phase)                      # E_b * V
weighted * sin  →  source_imag         # E_b * V
scatter_add real to destinations       # E_b * V
scatter_add imag to destinations       # E_b * V
```
→ **8 * E_b * V FLOPs**

**Step 3: Complex multiply with weights** (custom_functions.py:133-134)
```python
real_out = real_in*real_w - imag_in*imag_w   # M*V * 3
imag_out = real_in*imag_w + imag_in*real_w   # M*V * 3
```
→ **6 * M * V FLOPs**

**Step 4: Polar decomposition + new activation strength** (custom_functions.py:137-139)
```python
atan2(imag, real+eps)                  # M*V
real^2 + imag^2 + eps → log → *0.5    # M*V * 5
sum(dim=-1)                            # M*V
```
→ **7 * M * V FLOPs**

Note: prior to the 2026-04-11 refactor, Step 4 also did an in-function
mean-subtraction + geom_mean division (3*M*V + M extra). That normalisation
now lives in `native/layer.py` as post-update LayerNorm — see
`concepts/mag_normalization.md`.

**Step 5: Selective update** (custom_functions.py:145-152)
```python
where(mask, new, old) × 3             # M*V + M*V + M
```
→ **M * (2V + 1) FLOPs**

#### `update_activations` total per call

```
F_update = 7*E_b + 8*E_b*V + 15*M*V + M
```

#### Per iteration (with post-update LayerNorm + temporal decay)

With E_b = B*N*C (batched edges), M = B*N.  For the LN path, the
`_normed_update` closure adds LayerNorm (≈5*M*V) and recomputes act_strength
from the normalised mag via `activation_strength_forward` (≈4*M*V: exp + cos +
multiply + sum).

```
F_iter = F_update
       + B*N*5*V       [LayerNorm post-update]
       + B*N*4*V       [act_strength recompute from LN'd mag]
       + B*N*V         [temporal decay]
       = B*N * (8*C*V + 7*C + 25*V + 1)
       ≈ B*N * (8CV + 7C + 25V)
```

Net change vs pre-refactor: +1*V per node per iteration (≈+4% of `24V` term).
Negligible for the grand total approximation.

### Phase 4: Output Extraction

`act_strength[idx] / sqrt(V)` → B * n_out. Negligible.

## Grand Total Formula

```
FLOPs_forward = B*n_in*V*2                           # Phase 1: input tanh
              + B*N*12*V                              # Phase 1: init
              + F_update(B*n_in, 2*B*n_in, V)         # Phase 2: injection
              + (I-1) * B*N*(8*C*V + 7*C + 25*V)     # Phase 3: message passing
              + B*n_out                                # Phase 4: output
```

**Simplified (Phase 3 dominates by 99%+):**

```
FLOPs_forward ≈ (I-1) × B × N × (8CV + 7C + 25V)
```

**Per training step** (forward + backward with grad checkpointing):
```
FLOPs_step ≈ 3 × FLOPs_forward
```

**Per epoch:**
```
FLOPs_epoch = steps_per_epoch × FLOPs_step
            = ceil(num_train_samples / batch_size) × 3 × FLOPs_forward
```

---

# Part 3: How to Calculate for a New Run

## Step-by-step

1. Read the config: get N, V, C, I, B, n_in from `config.yaml`
2. **Parameters:** `2*N*V + (16 if layernorm else 0) + (n_out*10 + 10 if FFN else 0)`
3. **FLOPs per forward pass:** `(I-1) * B * N * (8*C*V + 7*C + 24*V)`
4. **FLOPs per step:** `3 * FLOPs_forward`
5. **Steps per epoch:** `ceil(num_train_samples / B)` — adjust for data_fraction if <1.0
6. **FLOPs per epoch:** `steps_per_epoch * FLOPs_step`
7. **Total training FLOPs:** `epochs * FLOPs_epoch`

## Quick Python snippet

```python
def calc_flops_and_params(N, V, C, I, B, n_in, n_out,
                          layernorm=True, ffn=False,
                          num_train=9469, data_fraction=1.0, epochs=40):
    # Parameters
    params = 2 * N * V
    if layernorm:
        params += 2 * V
    if ffn:
        params += n_out * 10 + 10

    # FLOPs
    flops_fwd = (I - 1) * B * N * (8*C*V + 7*C + 25*V)
    flops_step = 3 * flops_fwd
    effective_samples = int(num_train * data_fraction)
    steps_epoch = -(-effective_samples // B)  # ceil division
    flops_epoch = steps_epoch * flops_step
    flops_total = epochs * flops_epoch

    return {
        "params": params,
        "flops_forward": flops_fwd,
        "flops_step": flops_step,
        "steps_per_epoch": steps_epoch,
        "flops_per_epoch": flops_epoch,
        "flops_total_training": flops_total,
    }
```

## Caveats & adjustments

- **Edge dropout** (p=0.2): Saves ~20% of edge-related FLOPs during training. Formula gives worst-case (all edges). Multiply edge terms by (1-p) for average-case.
- **Active mask growth**: Early iterations have fewer active edges than N*C. Formula uses worst-case. First 1-2 iterations may be 30-50% cheaper.
- **Radiation** (if `radiation_targets > 0`): Adds cosine search O(num_active * N * 2V) per iteration. Not included since radiation=0 for run10+.
- **Backward pass multiplier**: 3x with grad checkpointing (2x recompute + 1x grads). Without checkpointing: 2x forward for backward.
- **FFN head FLOPs** (runs 1-9): `B * (n_out * 10 + 10)` per forward — negligible compared to GNN.
- **Validation FLOPs**: Forward-only (no backward). Multiply forward FLOPs by val_steps for total.
