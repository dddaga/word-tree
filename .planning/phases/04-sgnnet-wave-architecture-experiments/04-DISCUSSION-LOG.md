# Phase 4: SGNNET Wave Architecture & Experiments - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 04-sgnnet-wave-architecture-experiments
**Areas discussed:** N_in strategy, Spatial encoding for variable D, GA fitness scoring, model_wave.py code reuse

---

## N_in Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Keep N_in=25088 direct | Same as Phase 3. C_input_mask is [25088, N_hidden], binary. Spatial encoding carries over as-is. | ✓ |
| Add input adapter (Linear) | Linear(25088 → N_hidden) before SGNNET_Wave. Adds learned params. | |
| Drop input encoding | Treat inputs as 1D scalars, lose spatial structure. | |

**User's choice:** Keep N_in=25088 direct
**Notes:** Since C_input is binary/immutable in Phase 4 (no learnable values), there are zero params in the seeding matrix. No budget reason to add an adapter. Spatial encoding stays identical to Phase 3.

---

## Spatial Encoding for Variable D

| Option | Description | Selected |
|--------|-------------|----------|
| Lock D=4 for all experiments | GA sweeps only K and N_hidden; D stays 4. Phase 3 encoding works as-is. | ✓ |
| Truncate/pad the 4D encoding | D=2: [feature_val, h_norm]; D=8: zero-pad to 8D. | |
| Decouple geometry from VGG structure | Drop spatial encoding; input positions random in [0,1]^D. | |

**User's choice:** Lock D=4 for all experiments
**Notes:** Simplest approach — avoids encoding complexity. GA search space updated: D removed from sweep, only K ∈ {1,2,3,4} and N_hidden ∈ {64,128,256} are swept.

---

## GA Fitness Scoring

### Round 1: Base fitness approach

| Option | Description | Selected |
|--------|-------------|----------|
| Simple: −final_loss | Score = −mean_loss_last_3_epochs. NaN → −1e6. No monotonicity check. | ✓ (initial) |
| Trend-weighted | stability_bonus multiplier. | |
| Strict monotonic | Any uptick > 5% disqualifies. | |

**Initial choice:** Simple −final_loss

### Round 2: Complexity penalty (user-initiated)

User raised the valid concern: vanilla −final_loss will always prefer larger N_hidden since bigger models have more capacity. Asked about incorporating time/space complexity into scoring (analogy to NAS efficiency metrics).

| Option | Description | Selected |
|--------|-------------|----------|
| Efficiency ratio | score = −final_loss × (params_min/model_params)^0.2 | ✓ |
| Pareto-front selection | 2D (loss, params) Pareto-optimal frontier for parent selection. | |
| Log-params penalty | score = −final_loss − γ × log(model_params), needs γ calibration. | |

**Final choice:** Efficiency ratio with β=0.2
**Notes:** params_min = parameter count of smallest search config (N_hidden=64, K=1). With β=0.2, a 4× larger model needs ~24% lower loss to score the same. NaN/inf → −1e6 (disqualified). final_loss = mean of last 3 epochs on 15% partial data.

---

## model_wave.py Code Reuse

| Option | Description | Selected |
|--------|-------------|----------|
| Import shared helpers | model_wave.py imports from model.py, encoding.py, geometry.py. Phase 3 files unmodified. | ✓ |
| Fully standalone | model_wave.py duplicates all shared code. Clean isolation. | |
| Refactor into sgnnet/utils.py | Extract to utils.py; both model.py and model_wave.py import from there. Cleanest but touches Phase 3. | |

**User's choice:** Import shared helpers
**Notes:** model_wave.py is additive only — Phase 3 files never modified. Imports: _make_sparse_c (or binary variant), compute_spatial_encoding, personal_volume_radius.

---

## Claude's Discretion

- Binary C mask initialization strategy
- ε threshold for masked normalization
- Whether to add `binary=True` flag to _make_sparse_c or define _make_binary_c in model_wave.py
- GA mutation/crossover details
- Whether Stage B/C seed from Stage A best config

## Deferred Ideas

- Sweeping D in GA (locked to 4 for Phase 4, could be Phase 4.5/v2 experiment)
- Pareto-front GA selection (considered, user chose efficiency ratio)
- Time-per-epoch in fitness function (discussed, not added — implementation complexity)
