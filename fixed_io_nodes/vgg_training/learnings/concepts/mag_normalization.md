# Concept: Magnitude Normalisation

## What it is

How the per-node log-magnitude vector `mag_act` (shape `N × V`) is kept bounded
across propagation iterations so that `exp(mag_act)` in the aggregation step
does not explode or collapse.

Two regimes exist in the history of this codebase:

| Regime | Runs | Where it lives |
|---|---|---|
| **Mean-subtraction (legacy)** | 1–16 | inside `update_activations` |
| **Post-update LayerNorm (current)** | 17+ | inside `native/layer.py` |

---

## Notation

Each node holds a complex vector across `V` dims:
`z[j] = exp(mag[j]) · exp(i · phase[j])`

After message-passing + weight multiplication, each destination node `d`
receives Cartesian outputs per dim `j`:
- `r[d, j]` — real part
- `im[d, j]` — imaginary part

From these:
- `new_phase[d, j] = atan2(im[d, j], r[d, j])`
- `new_mag[d, j] = 0.5 · log(r[d, j]² + im[d, j]²)`

---

## Legacy: in-function mean-subtraction (runs 1–16)

`update_activations` did the following immediately after computing `new_mag`:

```python
μ[d]             = (1/V) · Σⱼ new_mag[d, j]
new_mag[d, j]    ← new_mag[d, j] − μ[d]
act_strength[d]  = Σⱼ ( r[d, j] / exp(μ[d]) )
```

**What it did:** mean-subtraction across the V dims of each node's log-magnitudes.
Equivalently, division of each node's linear amplitudes by the geometric mean of
its V per-dim amplitudes. The `act_strength` formula was divided by `exp(μ)` to
stay consistent with the shifted mag.

**Why `act_strength` was coupled:** without the `/ exp(μ)` division, `act_strength`
would not equal `Σⱼ exp(new_mag[j]) · cos(new_phase[j])` after the shift — the
coupling is an accounting fix, not an independent design decision.

**Why it was wrong:** this is LayerNorm with `γ = 1`, `β = 0` hardcoded. No learned
parameters. The network could never express that a particular node (or dimension)
should operate at a consistently higher or lower amplitude scale — mean was
always locked to zero in log-space, i.e. geometric mean amplitude always 1.

The `layernorm: true` config flag in runs 3–16 applied `nn.LayerNorm` to the
*source* mag of the previous iteration *before* aggregation (pre-update). That
normalised what a node *sent*, not what it *received*, and stacked on top of the
hardcoded in-function subtraction. The learned `γ, β` of LN only affected what
went into the next aggregation — the stored state immediately had them stripped
by the in-function mean-subtraction, so their effect was limited.

---

## Current: post-update LayerNorm (run 17+)

`update_activations` no longer touches `new_mag`:

```python
new_phase[d, j]  = atan2(im[d, j], r[d, j])
new_mag[d, j]    = 0.5 · log(r[d, j]² + im[d, j]²)
act_strength[d]  = Σⱼ r[d, j]
```

The `act_strength = Σ r` form is correct because
`exp(new_mag) · cos(new_phase) = sqrt(r² + im²) · r/sqrt(r² + im²) = r`.

Then `native/layer.py` runs the external LayerNorm inside the `grad_checkpoint`
closure, immediately after the update:

```python
μ[d]             = mean( new_mag[d, :] )
σ[d]             = std ( new_mag[d, :] )
new_mag[d, j]    ← γ[j] · (new_mag[d, j] − μ[d]) / σ[d]  +  β[j]
act_strength[d]  = Σⱼ exp(new_mag[d, j]) · cos(new_phase[d, j])
```

`γ, β` are `nn.Parameters`. The network can now express a non-zero target mean
log-amplitude (`β ≠ 0` → geometric mean amplitude `exp(β) ≠ 1`) and a learned
spread (`γ`).

`act_strength` is **recomputed from the normalised mag** so the next iteration's
softmax routing sees the node's true current state, not a stale pre-LN value.

The same post-update normalise-then-recompute flow is applied inside
`_inject_inputs_batched` for the updated input-node slice.

---

## Scope of the change

- `core/custom_functions.py` — `update_activations` no longer mutates `new_mag`
  or couples `act_strength` to a mean. Purely the math, no flag.
- `native/layer.py` — `_normed_update` closure applies LN post-update and
  recomputes `act_strength`; `_inject_inputs_batched` does the same for the
  Bn input-node slice.
- All 16 legacy configs carry a documentary `subtract_mean: true` annotation
  pointing at this file. The code **does not read it** — it exists to tell
  future readers what those runs used.
- No backward-compat flag in code. Use git to reproduce legacy runs.

---

## Reproducing legacy runs

The last commit before this refactor is `7f072c4` on branch `fixed_io_nodes`:

```bash
git checkout 7f072c4    # detached head at last legacy commit
# or:
git checkout -b legacy-mean-subtract 7f072c4
```

All runs 1–16 were trained under that code. Their weights in
`training_runs/runN/*.pt` are only meaningful when loaded against that version
of `core/custom_functions.py` / `native/layer.py`.

---

## What to watch in run 17+

After the switch to post-update LN:

- **Magnitude drift**: the in-function mean-subtraction was also (implicitly)
  preventing unbounded log-mag growth. LN's `/σ` should cover this, but worth
  checking `mag_act.abs().max()` across iterations in `diagnose.ipynb`.
- **`γ, β` distributions**: after some epochs, inspect `layer.mag_norm.weight`
  and `layer.mag_norm.bias`. If they collapse back to ~1, ~0 the network isn't
  using the new freedom and LN is acting as plain mean/variance normalisation.
- **act_strength distribution**: the shift from `Σ(r / exp(μ))` to
  `Σ exp(LN(new_mag)) · cos(new_phase)` changes the scale of act_strength.
  Routing temperature sweeps from runs 10–15 may need recalibration.
