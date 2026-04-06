# SGNNET Experiment Queue (Live)

**GA rule (STRICT):** Every new experiment base = ALL confirmed winners from all prior generations.
**Calibration rule:** When base changes generation/scale, run a 40-50ep param sweep BEFORE full 150ep runs.
**Critical Findings:** See [EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md](EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md)

---

## Current Architecture Defaults (N=4096, patched arch)

| Param | Value | Source |
|-------|-------|--------|
| AH alpha | 1.0 | step88 confirmed |
| K_hh | 4 | step86 (+0.56pp, −18% FLOPs) |
| K_iter | 12 | step71/step89 confirmed |
| turing | 0.0 | step70 confirmed |
| reflect | 0.5 | step69 confirmed |
| n_groups | max(8, N//8) | step86/feedback |
| Project best | **97.86%** | step89-A (K_iter=12+K_hh=4), 150ep, best_ep=134 |

---

## Currently Running (cap: 4)

| Machine | Slot | Session | Step | Status | Note |
|---------|------|---------|------|--------|------|
| Mac Studio MPS | 1 | step115_scout (tmux ✓) | step115 | 20ep scout | Z-bias + redistribution at N=4096, Ref at ep10 |
| Mac Studio CPU | 1 | step113C_t1 (tmux ✓) | step113 | 75ep Tier-1 | Heading toward KILL. C at ep30, all below Ref |
| Mac Mini | both | — | — | IN USE | Sudarshan |

---

## Priority Queue (Reprioritized 2026-04-09 — Scored Assessment)

### Scoring: Impact(1-5) + Evidence(1-5) + Speed(1-5) + Script(0/2) + Vision(1-5) = Total

### P0 — Critical path (score ≥ 17, launch immediately)

| Step | Score | Description | Why P0 | Script |
|------|-------|-------------|--------|--------|
| **step115** | 17 | **Scale Z-bias + redistribution to N=4096** — 4 configs, 20ep scout RUNNING | Biggest N=1024 winners, never scaled. Direct path to 98%+. | ✅ RUNNING |
| **step129** | **18** | **Markov routing revival** — D=32: soft-norm (magnitude preservation), zact (dynamic AH), hybrid (wpos×zact), redistribution compound. 6 configs. | **Core vision test.** F.normalize kills magnitude history, static AH kills content-routing. Addresses the 3 structural constraints. D=32 = 2× faster. | ✅ YES |

### P1 — High value (score 13-16, launch when slots free)

| Step | Score | Description | Why P1 | Script |
|------|-------|-------------|--------|--------|
| **step128** | 14 | **ConcatReLU / activation ablation** — weighted negative, ConcatReLU+proj, SwiGLU, LeakyReLU. 6 configs, N=1024. | Sub-threshold info recovery within-step. Complements step129 (activation fn vs norm/AH). | ✅ YES |
| step102 | 14 | **Phase polarizer + alternating training** — Malus's law cos² filter + alternating BCD + Pauli exclusion. D=32. | step105 CONFIRMED Phase+AH synergistic (+4.21pp). Unblocked. | ✅ YES |
| step72 | 13 | **N-scaling patched arch** — N={256-4096-8192}, K_hh=4 | Gates entire efficiency track. Can't target 1% FLOPs without this curve. | ✅ YES |
| step117 | 13 | **Learned input projection** — W_proj [D,D], low-rank, per-neuron bias, GroupNorm. N=1024. | Bottleneck #2 (input compression 392×). Step55 showed +5pp potential. | NO |

### P1.5 — Moderate value (score 10-12)

| Step | Score | Description | Why P1.5 | Script |
|------|-------|-------------|----------|--------|
| step124 | 12 | **RigL topology refinement** — swap K_hh edges by gradient magnitude every 10ep | Sparse-to-sparse training. Novel for SGNNET. | NO |
| step103 | 11 | **Wave interference** — distance-based phase shift + decay, D=16 | step105 confirmed phase not dead with AH. | ✅ YES |
| step110 | 10 | **Muon optimizer ablation** — Muon, LION, schedule-free vs AdamW | Orthogonal to arch. Could unlock gains everywhere. | ✅ YES |
| step118 | 10 | **Attention-pooling readout** — learned neuron attention, top-k pool, multi-head. N=1024. | Bottleneck #3. Mean-pool discards all graph structure. | NO |

### P2 — Lower priority (score 8-9, or gated on other experiments)

| Step | Score | Description | Depends on | Script |
|------|-------|-------------|------------|--------|
| step120 | 9 | **K_iter=16-24 + Z-bias + gradient checkpoint** at N=4096 | step115 (does Z-bias scale?) | NO |
| step108 | 9 | **Hierarchical polar routing** — W_pos polar decomposition | Independent but low expected impact | ✅ YES |
| step130 | 9 | **Beam-as-global-broadcast** — top-M active neurons broadcast to all via soft attention. M={4,8,16}, λ={0.1,0.3}, every step vs every 3rd. D=32. | Long-range info at O(M×N) cost. Beam code exists but was dead (turing=0.0). | ✅ YES |
| step125 | 8 | **AH alpha fine-sweep** — α={1.05, 1.1, 1.2, 1.3} at N=4096, 40ep | Quick but low expected delta | NO |
| step100 | 8 | **K_in sweep** — K_in={5,10,15,25,50} at N=4096 | FLOPs lever, never tested | NO |
| step119 | 8 | **Adaptive K_iter per sample** — confidence-based early exit | For efficiency track | NO |
| step123 | 8 | **Stochastic depth training** — skip random K_iter steps during training | Free regularization, trivial | NO |
| step104 | — | **Compound wave+polar** | step102 + step103 results | ✅ YES |
| step126 | — | **µP initialization for N-scaling** | step72 (need scaling curve) | NO |
| step127 | — | **Progressive K_iter distillation** | Efficiency track | NO |

### P3 — Speculative / deprioritized

| Step | Description | Reason |
|------|-------------|--------|
| step92 | ReLU group routing | Group routing: 0 wins in 3 attempts (step83, step107, step90). Track record too poor. |
| step121 | Spectral norm on W_pos | Theoretical, low evidence |
| step122 | DiffPool hierarchical readout | Adds significant params |
| step93 | GRAND implicit diffusion | Speculative, no script |
| step94 | Hamiltonian message passing | Speculative, no script |
| step97 | Beltrami flow | Speculative, no script |
| step85 | Stacked SGNNET parallel | Low priority, old design |
| step74 | Safety valve re-validation | step79 showed aux losses marginal |
| step95/101 | Graph init comparison | Low priority |

### Removed from queue

| Step | Why |
|------|-----|
| step116 | **SUBSUMED by step129-A** (soft-norm = magnitude preservation test). Not needed as standalone. |
| step114-C | **REMOVE** — Scout was −2.14pp. Not worth a Tier-1 slot. |
| step113 | **Heading toward KILL** — Tier-1 running, all configs below Ref (89.25%). Will be marked KILLED when complete. |
| step99 | DONE — W_phase marginal at N=4096 (+0.18pp) |
| step96 | KILLED — DropMessage catastrophic at K_hh=4 |
| step109 | Blocked permanently — step107 KILLED |
| step84 | Gated on step83 which showed inter-group routing hurts |
| step98 | step90 killed group topology at N=4096 |
| G4 | Subsumed by step115-B |

---

## Launch Order (when slots free)

Given 2 Mac Studio slots (1 MPS, 1 CPU) and Mac Mini intermittently:

1. **step129** → Mac Studio MPS (after step115 scout completes) or Mac Mini — 20ep scout at D=32, fast
2. **step128** → Mac Studio CPU (after step113 completes) — 20ep scout at N=1024
3. **step102** → next free slot — 20ep scout at D=32
4. **step72** → when a long slot opens — N-scaling sweep, multiple N values
5. **step117** → needs scripting first, then 20ep scout

---

## Gen4+ Completed Experiments (Reference)

| Step | Result | Key finding |
|------|--------|-------------|
| step69 ✓ | Ref=83.36%, A(turing=0.3)=85.04% | NEW REF_BASELINE_v2 established |
| step70 ✓ | **97.38%** (Config B, turing=0.0) | PROJECT BEST (later surpassed). turing=0.0 wins at N=4096 |
| step71 ✓ | Ref=95.87%, **C=96.66% (K_iter=12)** | K_iter=12 optimal at N=4096 |
| step73 ✓ | Ref=84.56%, **D=86.34%** | Softmax redistribution routing wins (+1.78pp) |
| step75 ✓ | Ref=~84%, **D=87.24%** | Temperature routing wins (+3.98pp) — best N=1024 routing |
| step76 ✓ | Ref=83.67%, **A=86.55%** | turing=0.0 + W_phase trained wins (+2.88pp) |
| step77 ✓ | Ref=85.10%, all others below | Learnable theta KILLED. Fixed θ=0.1 optimal |
| step79 ✓ | Ref=96.10%, D/F=96.31% | Aux losses borderline (+0.21pp), not adopted |
| step80 ✓ | N512=72.79%, N2048=92.74% | Patched arch gains grow with N |
| step81 ✓ | **A=85.55%** | Hebbian rewiring +1.12pp |
| step82 ✓ | **A(n_g=8)=85.63%** | Group topology +3.01pp; n_groups=8 optimal |
| step83 ✗ | A=78.60% KILLED | Group routing killed; temporal mismatch + coarse S_g |
| step86 ✓ | **A=96.59%, K_hh=4**; I=95.95% (D=32) | K_hh=4 new default; −18% FLOPs. D=32 viable at N=4096 |
| step87 ✓ | Ref=84.28%, **C=86.80%** | Proximity routing C wins (+2.52pp), most configs killed |
| step88 ✓ | **α=1.0 → 95.52%** | α=1.0 confirmed optimal at N=4096 |
| step89 ✓ | **A=97.86%**, Ref=97.58% | **NEW PROJECT BEST**. K_iter=12+K_hh=4 at N=4096, 150ep |
| step90 ✓ | All ~96.6% (Ref=96.66%) | Group topology **NULL at N=4096**. +3.01pp at N=1024 doesn't scale |
| step91 (partial) | Ref=84.13%, **A=86.80% (a=0.05)** | GCNII +3.44pp. B-E still running |
| step96 ✗ | All KILLED | DropMessage catastrophic at K_hh=4 |
| step99 ✓ | Ref=96.56%, B=96.74% | W_phase trained **marginal at N=4096** (+0.18pp) |
| step72 ✓ (partial) | N=256→57.9%, 512→74.5%, 1024→87.0%, 2048→93.7% | Steep monotone N-scaling |
| step105 ✓ | **A(PT+AH)=86.68%**, C(coh)=85.50%, B(PT only)=35.9% | **H1 CONFIRMED: Phase+AH synergistic (+4.21pp)** |
| step106 ✓ | **A=90.78%** (Z-bias init=0), E=89.35% (Z-bias+GCNII) | **NEW N=1024 RECORD +7.42pp**. 768 params. |
| step107 ✗ | All top-k<8 KILLED; E(delay=30)=84.97% marginal | Group MoE routing KILLED again |
| step112 ✗ | Ref=87.77%; A/B/C ~28-33% KILLED; D=73.76% | RNN sequential injection KILLED |
| step113 (Tier-1) | Ref=89.25%; A~87.87%; B=88.61%; C running ~80% | Intermediate supervision: all below Ref. Heading toward KILL. |
| step114 (scout) | Ref=84.46%; C=82.32% (-2.14pp); A/B/D KILLED | Parallel chunks: mostly KILLED |
