# Phase 5: Phase Mechanism Research & New Experiment Designs (2026-03-30)

Three parallel research agents investigated dynamic excitatory connections via phase anchors.
Key architectural finding that drove this work:

**W_phase is wired to nothing in dynamic_z_geo mode.** In the forward() loop,
`W_ph_norm = F.normalize(self.W_phase, dim=-1)` is computed and passed to
`_phase_inhibit()`, but `_phase_inhibit` in dynamic_z_geo mode calls `_inhibit_dynamic_z()`
which ignores W_ph_norm entirely. Every D=16 experiment (steps 8–16) has W_phase
completely disconnected from routing. The "resonant" mechanism has NEVER been tested
in the best configuration.

The research was structured around 3 questions:
  1. Phase-based neural dynamics (Kuramoto, STDP, binding-by-synchrony)
  2. Dynamic GNN routing (GAT, capsule networks, transformer self-attention)
  3. Fast in-context adaptation of W_phase (fast weights, Hopfield, Oja's rule)

---

## Core Finding: W_phase Must Be a Fast Weight, Not a Slow Parameter

**Why learned W_phase fails** (confirmed by step10b):
Gradient descent on W_phase across training examples converges it to an average direction
that discriminates no specific input. This is the fundamental failure mode.

**The fix**: W_phase should be adapted WITHIN each forward pass using a Hebbian rule —
making it input-specific without any gradient descent. The nn.Parameter W_phase becomes
a "slow prior" (initialization). A local variable `A = self.W_phase.clone()` is the
"fast weight" — discarded after each forward, never accumulated across examples.

This is the Ba et al. (2016) fast-weight architecture applied to routing.

---

## Candidate Fast-Weight Rules for A (batch-local clone of W_phase)

All rules update A inside the K_iter routing loop. A is shape [N, D] (batch-shared)
or [B, N, D] (per-sample, correct but 2× memory).

### Rule (b): Oja's rule — stable, biologically motivated
```
sim = (A * Z).sum(-1, keepdim=True)     # [B, N, 1] or [N, 1]
A = normalize(A + α * Z * sim)
```
Fixed point: A = principal eigenvector of Z's auto-covariance.
= the dominant routing direction for this input.
Risk: all A[h] collapse to the same direction if initial Z is uniform.
Cost: O(N·D) per step. **Best first experiment.**

### Rule (c): Hopfield attract — Z is pulled toward stored A directions
```
gate = sigmoid(einsum('nd,bnd->bn', A, Z).unsqueeze(-1))   # [B, N, 1]
Z = Z + α * A.unsqueeze(0) * gate                          # Z attracted to A[h]
Z = normalize(Z)
```
Fixed point: Z converges toward whichever A[h] direction it started closest to.
A acts as an attractor / "clean-up memory" for Z.
Risk: runaway excitation if l2-normalize is not applied after each step (it is).
Cost: O(N·D) per step. **Architecturally cleanest.**

### Rule (d): Attention update — A becomes cluster centroid of Z
```
scores = einsum('bnd,md->bnm', Z, A) / sqrt(D)   # [B, N, N]
A_new = einsum('bnm,bmd->bnd', softmax(scores, -1), Z)
A = normalize(A_new)                               # [B, N, D]
```
Fixed point: A[h] = centroid of the Z-cluster most similar to current A[h].
Fully input-specific (per-sample). O(N²·D) — expensive without beam sparsification.
Risk: all A collapse to one cluster (temperature τ=1/√D=0.25 needed to prevent this).
Cost: O(B·N²·D) — use sparse variant with beam.

---

## SGNNET as LISTA Sparse Encoder

Discovered connection: the architecture already implements Learned ISTA (LISTA, Gregor &
LeCun 2010) when viewed through the sparse coding lens:
  - W_phase = dictionary D
  - Z = sparse code s
  - θ-gating (relu(Z - theta)) = soft-thresholding = ISTA shrinkage
  - K_iter routing steps = K unrolled ISTA iterations

The fixed point of LISTA iterations: Z* = argmin_s 1/2||x - W_phase*s||² + λ||s||₁
This is a principled sparse representation, not ad-hoc routing.
Implication: more K_iter → sparser, more accurate representations (as long as no over-smoothing).

---

## Signed Coupling: The Unified Excitatory/Inhibitory Mechanism

Based on binding-by-synchrony (von der Malsburg 1981, Singer 1989):
  - Two neurons with cos(Z_h, Z_j) > 0: same "phase group" → EXCITE each other
  - Two neurons with cos(Z_h, Z_j) < 0: different "phase group" → INHIBIT each other

This replaces TWO separate pathways (inhibitory radiation + excitatory radiation) with ONE:
```
Z_h += alpha_signed * Σ_j cos(Z_h, Z_j) * Z_j     # positive = excite, negative = inhibit
Z_h = normalize(Z_h)
```
All-pairs: O(N²·D) = 4.2M ops at N=512, D=16 — feasible on MPS in batches.

**Key advantage**: no hyperparameter for excitation vs inhibition balance — the cosine
similarity naturally allocates both. No separate beam selection needed.

---

## STDP: Causal Cross-Step Excitation

Spike-Timing Dependent Plasticity analog: neurons active at step k causally excite
neurons that become active at step k+1. Implementation:
```
Z_prev = Z.clone()    # cache before routing step
# ... standard routing produces Z_new ...
# STDP: prior beam Z excites currently similar neurons
cos_sim = einsum('bmd,bnd->bmn', Z_prev[prev_beam], Z_new)
Z_new += alpha_stdp * (gate * Z_prev[prev_beam]).sum(dim=1)
```
Creates a "routing agenda": early-step activations guide later-step activations within
the same forward pass. No new learnable parameters.

---

## GAT Softmax Temperature Fix

Step14 excitatory radiation uses LINEAR normalisation for excitatory weights:
  `exc_weight = exc_vals / exc_vals.sum(-1, keepdim=True)`
Research confirms: SOFTMAX with temperature τ=0.25 (= 1/√D at D=16) is correct:
  `exc_weight = softmax(exc_vals / 0.25, dim=-1)`
This is the single highest-leverage fix to step14 — likely to improve results by
+0.5% even without other changes.

---

## Step 17: Fast W_phase Experiments (PLANNED → scripts/train_step17_fast_phase.py)

All configs: D=16 Fourier N=512 dynamic_z_geo 120ep. Reference: step9A 29.22%.

| Key | Config | Mechanism |
|---|---|---|
| Ref | current dynamic_z_geo (no fast phase) | baseline |
| A  | rule (b) Oja, α=0.1, batch-shared | stable Hebbian |
| B  | rule (b) Oja, α=0.3, batch-shared | stronger Oja |
| C  | rule (b) Oja, α=0.1, per-sample | input-specific |
| D  | rule (c) Hopfield attract, α=0.1 | Z pulled to A |
| E  | rule (c) Hopfield attract, α=0.3 | stronger attract |
| F  | rule (d) attention update, τ=0.25, beam=32 | sparse attention |

---

## Step 18: Signed Coupling + STDP (PLANNED → scripts/train_step18_signed_coupling.py)

| Key | Config | Mechanism |
|---|---|---|
| Ref | current dynamic_z_geo (baseline) | |
| A  | Signed coupling full, α=0.3 (all-pairs, O(N²)) | unified exc/inh |
| B  | Signed coupling sparse, α=0.3 (top-32 + bottom-32) | efficient |
| C  | Signed coupling α=0.1 (weaker) | sensitivity test |
| D  | STDP S2, α=0.2 (Z_prev beam → current) | causal chain |
| E  | Signed coupling B + STDP D combined | both |

---

## Wait-For Dependencies

- **Step 14 results**: tells us if linear-normalised excitatory radiation helps at all.
  If it does: step17 (fast phase) and softmax-fix variant of step14 are highest priority.
  If it doesn't: signed coupling (step18) becomes the primary direction.
- **Step 13 depth results**: tells us if more K_iter unlocks STDP-like causal chains.
  If K_iter=8 >> K_iter=3: run step17/18 at K_iter=8.
