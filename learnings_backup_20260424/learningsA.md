# Dynamic Routing — Failure Analysis and Next Experiments

*Written: 2026-04-14. This note captures why every dynamic routing attempt has
failed, what the loss curves and architecture reveal, and which paths remain
untried.*

---

## What We've Tried (and What Failed)

### step700 — Multi-hop Precompute (KILLED)
Replace K=5 serial iters with K=1-3 "wider" iters that aggregate K_hh_eff=6-32
neighbors in a single pass.

**Result:** All configs −3.5 to −13pp vs Ref (92.28%). Accuracy loss TRACKS
the number of lost iterations, not receptive field size:
- C_k1_kh32 (1 iter, 32-hop): −13pp
- A_k3_kh6  (3 iters, 6-hop): −3.5pp

**What this tells us:** Each iteration applies a nonlinear refinement stack
{ReLU-θ → gather-sum → AH-suppress → reflection → F.normalize}. Collapsing 5
passes into 1 loses 4 applications of this stack. The network needs depth, not
breadth. Expanding K_hh does not compensate for fewer iterations.

**Architecture insight:** K_iter is more like Transformer depth than like
multi-head width. You cannot parallelize it.

---

### step701 — Parallel Routing Branches (KILLED)
Replace the single K=5 sequential path with num_branches parallel paths
(K=5, 3, or 2), weighted by learnable α (softmax-normalized). Train α to
select which branch to use per-forward.

**Result:** All variants −7 to −34pp. Best: D_par3_k3=84.64% (−7pp).

**What the loss curves showed:** Branch weights α stayed nearly uniform
throughout training. All branches learned near-identical representations.
Learnable α never differentiated — entropy of α distribution stayed high.

**Why it failed (mechanistically):**
1. All branches start from the same input Z
2. No diversity pressure forces branches to specialize
3. The model finds it easier to average uniformly (safe, low-loss) than to
   route discretely (risky during early training)
4. Result = depth reduction disguised as branching

**What would need to change:** Explicit diversity regularization, or discrete
(hard) routing that prevents branch collapse.

---

### step610 — K_iter Annealing (KILLED)
Start training at K_iter=16, anneal to K_iter=5 over the first 50 epochs.
Hypothesis: the network would learn to "compress" high-K representations.

**Result:** ALL schedules −2 to −6pp at efficiency config.

**Why it failed:** Annealing doesn't preserve representations across K. The
network calibrated to K=16 during warm-up, then lost capacity when K decreased.
There is no transfer — K=16 weights are wrong for K=5 dynamics.

---

## Root Cause: The Per-Iter Nonlinear Stack Is Not Separable

All three failures trace to the same root cause. The per-iteration computation:

```
Z → ReLU(Z − θ) → gather(conn_hh) → AH_suppress(supp_w) → reflect → F.normalize
```

is a **nonlinear cascade** where each step's output conditions the NEXT step's
threshold (θ is applied to the normalized Z from the previous iter). This is
not like an unrolled linear recurrence that can be collapsed.

Specifically:
- **ReLU-θ gating**: threshold θ gates based on current Z magnitude; magnitude
  changes each iter due to F.normalize
- **F.normalize**: re-normalizes Z onto S^{D-1} each iter, changing the
  effective "distance" for the next gather-sum
- **AH suppression (wpos)**: static weights but applied to dynamically changing
  Z — the suppression pattern is different every iter

**The number of F.normalize applications equals K_iter.** These normalizations
are what prevent gradient explosion and maintain the hypersphere geometry.
Collapsing K iters into 1 = 1 normalization = geometry collapse.

---

## What Has NOT Been Tried

These are architecturally distinct from the failed approaches:

### 1. Hard Sparse Routing (Top-K Activation)
**Idea:** Instead of changing K_iter, change WHICH neurons activate per sample.
Only the top-K neurons (by input similarity score) participate in each forward
pass. Different inputs activate different subgraphs.

**Why this might work:** It doesn't change K_iter (routing depth is preserved).
It creates input-conditional sparsity, which is genuine dynamic routing.
Current AH suppression is static — same suppression every sample.

**Implementation:** Score each neuron by dot product with input x. Select
top-K per forward. Apply routing only through active neurons.

**Risk:** Breaks differentiability. Needs STE (Straight-Through Estimator) or
Gumbel-Softmax for gradient to flow through the top-K selection.

**Experiment design:** N=512 D=16 first (faster iteration). Add a learnable
1-layer scorer q = x @ W_score (N_in → N_hidden). Top-K selection with
STE. Compare vs Ref (same K_iter=5, same K_hh=2).

---

### 2. Input-Conditional K_iter (Per-Sample Adaptive Depth)
**Idea:** Simple inputs get K=2-3, hard inputs get K=5-8. The network learns
when to stop.

**Why this is different from step610:** We don't change training K_iter. We
add an early-exit mechanism that short-circuits the loop when Z has converged.
Convergence criterion: max(||Z_t - Z_{t-1}||) < ε.

**Expected behavior:** Easy samples (clear background, single object) converge
in 2-3 iters. Hard samples (cluttered, multiple classes) need 5+.

**Cost:** Near-zero params (just the convergence check). Could reduce average
FLOPs without accuracy loss.

**Experiment design:** Record convergence depth distribution over validation
set. Choose ε s.t. 80th percentile stops at K=3, 100th at K=5. No training
change needed — just an inference-time modification.

---

### 3. Input-Conditioned Suppression (Dynamic supp_w)
**Idea:** Current supp_w is static (wpos — position similarity, precomputed
once). Replace with a dynamic supp_w that depends on the current input batch.

**Why the static version works:** supp_w decorrelates positionally similar
neighbors. This is a prior about geometry.

**Why dynamic might help:** For some inputs, two neurons at nearby positions
might carry very different features. A dynamic version could suppress by
FEATURE similarity, not just position similarity.

**Implementation:** Compute supp_w per batch as Z@Z.T (neuron feature
correlation), then threshold. Cost: N×N matrix per iter — expensive at N=2048
(2048×2048 = 4M FLOPs). Need a sparse approximation.

**Experiment design (cheap):** First test concept at N=256 D=16 (fast). If
positive signal, design efficient approximation.

---

### 4. Attention-Weighted Gather (Soft Dynamic Connectivity)
**Idea:** Current gather: Z_nb = Z[conn_hh, :].sum(). Replace with:
Z_nb = (attn_weights * Z[conn_hh, :]).sum() where attn = softmax(Z @ Z[conn_hh].T).

**This IS dynamic routing:** different inputs produce different attention
patterns over the same static conn_hh graph.

**Critical constraint:** This ADDS FLOPs (attention computation). Must verify
efficiency still beats VGG baseline.

**Why previous attention-style mechanisms failed (step118: attention readout
−60pp):** That was at the READOUT (aggregating all neurons → class). This is
at the ROUTING (intra-layer message passing). Different operation, different
failure mode.

**Experiment design:** Tier-0 at N=512. Measure accuracy delta AND FLOPs delta.
Accept only if FLOPs increase < 2× AND accuracy gain > 0.5pp.

---

## Design Rules for Next Dynamic Routing Experiments

Based on failures, every new dynamic routing experiment must satisfy:

1. **Preserve K_iter=5 as training depth.** Never reduce the number of serial
   passes.

2. **One mechanism at a time.** Never compound routing changes with AH/reflect
   modifications.

3. **Diversity pressure if branching.** Any parallel branch design needs an
   explicit penalty forcing branch diversity (e.g., cosine similarity penalty
   between branch outputs > 0.9).

4. **Tier-0 at N=512 D=16 first.** Dynamic routing changes have most visible
   signal at small N where gradient paths are shorter.

5. **Check diagnostic metrics.** Add per-iter Z variance and branch entropy to
   TrainingDiagnostics. If these collapse during training, the routing is
   failing early.

6. **Sparsity as a constraint, not an outcome.** Don't let sparsity "emerge"
   from soft penalties — enforce it hard (top-K selection, STE gradients).

---

## Priority Queue (Untried Dynamic Routing Experiments)

| Priority | Mechanism | Key Risk | Effort |
|---|---|---|---|
| 1 | Per-sample adaptive K_iter (early exit) | ε tuning | Low (no new params) |
| 2 | Top-K hard neuron selection (STE) | Gradient instability | Medium |
| 3 | Attention-weighted gather (soft) | FLOPs budget | Medium |
| 4 | Dynamic supp_w (feature correlation) | N² cost | High |

Start with #1 — zero risk, zero params, inference-only change. If it shows
convergence speedup without accuracy loss, it's free FLOPs reduction.

---

*Related concepts: [[antihebbian]], [[architecture_dead_ends]], [[normalization]]*
*Evidence: step700 (KILLED, confirmed), step701 (KILLED, confirmed), step610 (KILLED, confirmed)*
