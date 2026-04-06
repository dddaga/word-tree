# Research Findings: Translation Invariance & Open Hypotheses

## Translation Invariance: Gap & Fixes

### SGNNET does not have translation invariance — and why
**Date:** 2026-03-26

The seeding step embeds spatial position directly into activations:
```
A_input[b, i] = [x[b,i], channel_norm[i], h_norm[i], w_norm[i]]
```
A golf ball at (row=2, col=3) and the same golf ball at (row=5, col=1) produce
activations in different D-space positions and activate different hidden neurons.
C_input is fixed, so there is no mechanism to recognise them as the same object.

**CNNs get translation invariance from weight sharing** — the same filter applied at
every position. SGNNET has no weight sharing in the connectivity.

### Why dynamic connectivity alone doesn't solve it
Proximity routing connects neurons based on their learned W_pos positions. If W_pos
is in a semantic space, routing *could* aggregate features regardless of position. But
by the time routing fires, spatial coordinates are already embedded in the activations.
Routing works on position-contaminated data.

**The root conflict:** D=4 is doing two jobs — encoding feature content AND spatial origin.

At D=16 Fourier encoding, this is partially addressed: 16 dimensions can encode
both positional structure (multiple frequency bands per axis) and feature content,
with routing learning to separate them. But true translation invariance still requires
either weight sharing or explicit position encoding separation.

### Fixes (ordered by invasiveness)

**Fix B — Channel-only C_input fan-in (recommended first step)**
Each hidden neuron samples from *all 49 spatial positions of one VGG channel*.
A_input drops h/w coordinates. Result: the same feature activates the same neurons
regardless of where it appears in the image.
Tested in `diagnose_ceiling.py` as `cinput_mode='channel'`.

**Fix C — Routing in semantic space (requires Fix B)**
With channel-based seeding, W_pos organises neurons by feature semantics.
Proximity routing then aggregates evidence from semantically similar neurons.

**Fix D — Learnable spatial attention (keeps spatial context when useful)**
Replace hard-coded h/w coordinates with a learnable attention mask over input
positions: `Z[h] = sum_i attn[h,i] * x[i]`.

**Avoid Fix A** (remove all spatial coords) — loses spatial relationships.

---

## Open Hypotheses Under Investigation

### Why was accuracy capped at ~11-16% at D=4?

**H1 — Random C_input destroys discriminative structure** (CONFIRMED)
Block-local C_input (channel-grouped) gives 17.96% vs random 10.17% at N=256/D=4.
Random projection of 25,088 VGG features averages out discriminative structure.
*Resolution:* `_build_fanin_conn` uses block-local bias — confirmed as important.

**H2 — Too few routing iterations** (PARTIALLY CONFIRMED)
K=2 (1 hidden iteration) reaches ~25 neurons per hop. 3 hops reach the 3-neighbourhood.
K_iter=3 confirmed as baseline. D=16 experiments with K_iter=1-12 (step13) will resolve
the "how much depth helps" question definitively.
*Current test:* `train_step13_beam_iter_scaling.py --row depth`

**H3 — K_in fan-in size too small/large**
K_in=50 may be too few or too many connections from the 25,088 input pool per hidden neuron.
*Status:* Not yet tested. Lower priority now that D=16 gives 29.22%.

### Why did D=16 produce such a large jump (+6.55% over D=4)?

**S³ overcrowding at D=4:** With N=512 neurons on S³ (4D sphere), average angular
separation between neurons is ~10°. Neurons are so close together that dot-product
similarity is high for almost ALL pairs — the routing signal is very noisy.

**S¹⁵ room at D=16:** Same 512 neurons on S¹⁵ have massive angular separation.
Each neuron has a genuinely distinct direction. Dot-product similarity between unrelated
neurons approaches zero — routing signal is clean and discriminative.

**Practical implication:** D is NOT about output dimension capacity (few readout params);
it is about the ROUTING space capacity. The readout only needs D × N_out = 16 × 10 = 160
params; the routing needs D >> log(N) to give meaningful angular separations.

### What limits accuracy at 29.22%?

Candidates:
1. **Graph topology bottleneck:** K_hh=6 structural connections may be too sparse.
   Test: K_local=8/K_random=8 sweep (step13b/connection density).
2. **Depth limitation:** K_iter=3 reaches a 3-hop neighbourhood of ~19 neurons.
   The full graph has diameter ~9 hops — routing only uses 1/3 of the graph.
   Test: K_iter=8 and K_iter=12 (step13 depth row).
3. **Routing mechanism conflict:** reflection and turing hurt at D=16 (step10a).
   Current best still uses alpha_reflect=0.3, alpha_turing=0.3 (from step9 which predates step10a).
   Test: step11 (routing ablation WITH geo mode).
4. **Data ceiling:** 9,469 training samples for 10-class VGG features.
   Resolution: multi-seed confirmation + potential data augmentation retest.
5. **Excitation/inhibition imbalance:** Radiation is only inhibitory.
   Adding excitatory radiation (step14) may unlock new capacity.

### Is W_pos learning meaningful structure?

W_pos starts at `torch.rand(N_hidden + N_out, D) * box_size`. It receives gradients
from: (a) the readout (projection onto W_out[c]) and (b) dynamic_z_geo routing
(geo_gamma * ||W_pos[beam] - W_pos[all]||² penalty enters the routing score).

Hypothesis: After training, W_pos clusters by class-related feature selectivity.
Neurons that become selective for the same class should drift together in W_pos space.
*How to verify:* Visualise W_pos t-SNE coloured by which class index activates each
neuron most strongly. No new experiment needed — can run post-hoc on any trained model.
