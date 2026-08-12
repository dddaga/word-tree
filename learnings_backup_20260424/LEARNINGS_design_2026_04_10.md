# Design Discussions — 2026-04-10

**Parent:** LEARNINGS_design.md (index)

---

## Core Hypothesis: Constraint Discovery (step152)

**Date:** 2026-04-10

Physical reality is constrained. Data from physical processes has compact sufficient representations that exploit those constraints. SGNNET's random graph + iterative routing is a search process for that representation.

**Question:** Can we give SGNNET architectural tools to actively find low-rank, objective-relevant representations?

### Mechanisms designed:

1. **Nuclear norm regularization** — penalizes rank of activation matrix Z via eigenvalues of Z^T Z. Forces low-rank representation.
2. **Mid-routing bottleneck** — D→D'→D projection at K_iter midpoint forces information compression through narrow channel.
3. **Dimensional gating** — learned per-step sigmoid gates over D dimensions. Network discovers which dims matter.
4. **L1 activation sparsity** — encourages sparse neuron activations.
5. **Contrastive routing loss (SupCon)** — shapes representation directly for class separation.

Script: `train_step152_constraint_discovery.py` — 7 configs (Ref + A-F), N=1024, 50%/75ep.

---

## Progressive Capacity Reduction (step153)

**Date:** 2026-04-10
**Inspired by:** Matformer (Google), GMP cubic schedule, Nested Dropout

Start with excess capacity (more N, more D), progressively prune by contribution during training.

### Modes designed:

1. **prune_dims** — GMP cubic schedule: D=32→16, prune lowest-importance dimensions
2. **nested_dropout** — enforce dimension ordering via cummin; unit j only active if j-1 active
3. **matformer** — multi-granularity loss at D={32,24,16} using truncated weights
4. **prune_neurons** — N=2048→1024, prune lowest-activation neurons
5. **prune_both** — simultaneous N and D pruning

Script: `train_step153_progressive_capacity.py` — 6 configs, multi-N/D.

---

## Safety Valve & Load Balance Removal

**Date:** 2026-04-10

**step154 result:** Ref(lambda=0.49)=64.08%, A(lambda=0)=73.83%(+9.75pp), B(lambda=0.98)=74.65%(+10.57pp).
VERDICT: Safety valve loss is NOT helping — actively hurting. AH handles positional diversity.

**Actions taken:**
- `experiment_config.py`: lambda_safety=0.0, lambda_lb=0.0
- `trainer.py`: Guards skip O(N^2) safety and O(N) load_balance computation entirely when lambda=0
- All future experiments inherit these defaults automatically

---

## Training Diagnostics System

**Date:** 2026-04-10
**Motivation:** "Just looking at the loss might be very limiting" — need to measure WHY training works or fails.

Implemented `src/training/diagnostics.py`:
- **Effective rank of Z** — SVD-free via eigenvalues of gram matrix Z^T Z → Shannon entropy → exp(entropy)
- **Neuron utilization %** — fraction with mean |activation| > 0.01
- **W_pos cosine similarity** — positional diversity (high = AH failing)
- **Separability ratio** — inter-class / intra-class cosine distance
- **Gradient norms** — per param group (W_pos, theta, fc)

Runs at epoch boundaries (every log_every epochs), not per-batch. One forward pass on 32 val samples.

**step155 diagnostics baseline:** Run on top-5 configs (A-E) to establish reference healthy ranges AND reveal gaps.

---

## Checkpoint System

**Date:** 2026-04-10

Implemented `src/training/checkpoint.py`:
- Full state: model, optimizer, scheduler, scaler, epoch, history, config, diagnostics_history
- CheckpointPolicy: save on new best val_top1, periodic (every 25ep), final epoch
- Best weights always at `{prefix}_best.pt`
- Integrated into Trainer.train() with resume_from parameter

---

## N x K Tradeoff (step140 — KILLED)

**Date:** 2026-04-10
**Result:** N dominates. More K (connectivity) HURTS accuracy (-8 to -44pp).
K_hh=4 confirmed optimal. Paper finding: N and K_iter are primary capacity knobs, not connectivity density.

---

## Dynamic Routing — Persistent Goal

**Date:** 2026-04-10

9 failed attempts analyzed (step58, 59, 60, 63, 65, 66, 73, 76, 83). Root causes:
1. **Gate death** — multiplicative gates compound g^K_iter → 0
2. **Co-adaptation** — router and AH fight over W_pos
3. **Temporal mismatch** — batch-level routing vs epoch-level AH
4. **Coarse routing** — S_g=mean(Z) loses per-neuron information

**What works:** Redistribution (step73/75, +1.78-3.98pp), static topology changes (step82, +3.01pp).
**Next-gen candidates must use:** additive injection, binary on/off, or sum-preserving redistribution.
See LEARNINGS_phase5_p15e_dynamic_routing_postmortem.md for full analysis.

---

## Paper Track Established

**Date:** 2026-04-10

Created `learnings/paper/` with:
- `claims.md` — 7 core claims (random graph sufficiency, N-scaling, AH as mechanism, gate-death theorem, etc.)
- `findings_log.md` — 11 chronological novel discoveries
- `baselines_needed.md` — gaps: MLP baseline, CIFAR-10, FLOPs demo
- `figures_planned.md` — 7 figures, 3 tables

Key paper-worthy finding: random additive aggregation over sparse O(N*K) graph crosses 90%+ on Imagenette — questions whether dense connectivity is necessary for representation learning.
