# Vision Review — sparse_geometric_network_report.md vs Reality (2026-06-10)

Original brief: March 2026. Review date: 2026-06-10. ~988 experiment steps elapsed.
Three-agent audit: mechanism fate trace, pivot timeline, architecture diff.

## 1. Vision Scorecard — Three Core Ideas

| Core idea | Verdict | Evidence |
|---|---|---|
| 1. Sparsity (most neurons don't talk) | **EXCEEDED** | O(N·K_hh=2) vs proposed O(N²) masked dense. 34,976 params = 0.029% of VGG FC. |
| 2. Dynamic connectivity (input-dependent routing) | **FALSIFIED in proposed form; SURVIVES mutated** | 16+ kills: gate-death theorem (steps 58–66), dynamic conn_hh 8/8 negative (step852), co-activation rewire (511–514), PhaseGate all forms (985/987/988). What survived: ΔW-projection (step234, +3.77pp) — input-dependent *weighting* on static topology. The diffraction spirit (geometry of medium governs flow) lives in W_pos + ΔW-proj, not in proximity thresholds. |
| 3. Recursive computation (K iterations) | **CONFIRMED load-bearing** | K_iter=5 default; K_iter primary capacity knob on Imagenette; K=1 only viable via KD from K=5 teacher (step605). |

Quantitative outcome vs brief's ambition: 95.95% @ 0.20M FLOPs (0.16% of VGG FC), 5.26×
wall-time speedup — exceeds anything the brief dared to number.

## 2. Mechanism Fate (11 design elements)

| Mechanism | Verdict |
|---|---|
| r*-threshold dynamic connectivity | ABANDONED (evidence: steps 31, 50, 511–514, 852) — but kills predate modern base; brief's *additive* form never tested at N=2048/D=16/ΔW-proj base → **STALE** |
| Static sparse C [N,N] learned values | EVOLVED → index tables, no learned values (conn_hh topology-only) |
| Self-projection readout | **KEPT verbatim** — attention readout killed (step118 −60pp) |
| Dead-zone Coulomb safety valve | PRESENT BUT HARMFUL — disabling = +9.75pp (step154); collisions never occur (AH organizes geometry) |
| Hebbian prune-and-grow (v2 roadmap) | **NEVER-TESTED at spec** — co-activation variants killed (511–514), GA killed (740–742), Gumbel killed (230); exact epoch-boundary prune-grow on |c_ij| stats untested |
| K-means init | **ABANDONED WITHOUT TEST** — violates try-and-test rule |
| Load-balancing loss | REDUNDANT — AH substitutes; 100% neuron utilization (step155) |
| Adaptive K stopping | ABANDONED — MoD variants killed (34, 119); inference-time convergence early-exit untested |
| Hypercube [0,1]^D | EVOLVED → S^{D-1} via L2-norm (Fourier breakthrough, 2026-03-26) |
| Per-step normalization | KEPT — load-bearing (step958: all alternatives kill) |
| Distillation training | KEPT — but target mutated (see Pivot 1) |

## 3. Pivot Audit (10 major)

7/10 evidence-driven with CONFIRMED tags (sphere, gate-death, ΔW-proj, static+geometry,
scale ladder, KD-for-K=1, AH-as-prerequisite). 3/10 pragmatic (FFN→VGG target, cross-dataset
scope, vision-only paper).

**Pivot 1 is the debt.** Transformer-FFN replacement was the brief's *problem statement* —
"We are targeting the FFN sub-layers of transformers first" — and is the single pivot with
**no recorded rationale and zero experiments**. VGG-FC was a testbed convenience that became
the paper. Legitimate (paper milestone served), but the original vision is untested, not failed.

## 4. Faithfulness Assessment

**Strong:**
- Every architectural abandonment except two backed by ablations with step numbers.
- Honest negatives preserved (audio −14pp, text −1pp) instead of buried.
- Evidence-tag discipline (CONFIRMED/HYPOTHESIS/STALE) held under pressure.
- Self-projection + per-step norm + recursive loop — the brief's *identity* — survived
  intact because experiments said keep them, not from sentiment.

**Weak:**
1. **Roadmap starvation** — "future work" items (Hebbian v2, adaptive K, K-means init,
   transformer target) never got even one T0 each at the mature base. Never-tested ≠ killed.
2. **K-means init abandoned without experiment** — direct try-and-test violation.
3. **Dead-direction re-attack** — 16+ gating experiments after gate-death theorem
   (873–916, then 985/987/988 today). Meditation 003 flagged it; behavior repeated.
4. **Pivot 1 undocumented** — convenience pivots skip the paper trail that failure
   pivots get.

## 5. Forward Plan — Increase P(vision seen through)

Principle: keep what demonstrably worked (tier ladder, evidence tags, kill-fast,
autonomous slot-filling); patch the two failure modes (roadmap starvation,
dead-direction re-attack).

### Track A — Ship Paper 1 (protects milestone; in motion)
step982 T2 (CIFAR aug claim), step986 T1 (N=16384) → paper claims locked.
Nothing in vision-debt preempts a paper slot.

### Track B — Vision-debt T0 batch (after paper claims locked)
| Step | Test | Why |
|---|---|---|
| step989 | **Transformer FFN distillation T0** — GPT-2-small layer-6 FFN, x_ffn→y_ffn MSE, brief §5 verbatim. N_out=768 self-projection readout, existing MSE path from TS work | Brief's problem statement. Never attempted. Architecture now mature enough to answer it. If positive → Paper 2 spine; if negative → honest close of original target |
| step990 | **Additive r*-threshold dynamic connectivity** on modern base (N=2048, D=16, ΔW-proj). One STALE retest. Additive path `normalize(static + dynamic)` — gate-death applies to multiplicative gates, never tested the brief's additive form post-ΔW | Either resurrects core idea 2 in original form or closes it CONFIRMED at current arch — ends the re-attack cycle permanently |
| step991 | **Hebbian prune-grow on conn_hh** via \|c_ij\| (ΔW-proj alignment) statistics, epoch-boundary outer loop. AH precedent: outer-loop W mutation works where in-forward gating dies | Brief §9 exact mechanism, adapted to surviving signal (\|c_ij\|), placed on the one signal path proven safe (epoch boundary) |
| step992 | **K-means init of W_pos** (hidden from X, output from class means) | Cheapest debt item; repairs try-and-test violation |

Rule: each gets ONE T0. Kill → CONFIRMED close + entry in dead-ends concept.
No re-attack without a new variable (enforced by VISION_DEBT register check).

### Track C — Process patches
1. **`learnings/VISION_DEBT.md` register** — every brief promise + status
   (KEPT/EVOLVED/KILLED-CONFIRMED/NEVER-TESTED). Meditation protocol gains step:
   review register, any NEVER-TESTED item older than 2 meditations must get a T0
   or an explicit user-approved WONTFIX.
2. **Convenience pivots get paper trail** — same rigor as failure pivots: one
   dated entry with rationale at decision time.
3. **Re-attack guard** — before launching any gating/routing-weight experiment:
   grep dead-ends concept + gate_death.md; require named new variable vs all
   prior kills, else don't launch. (Today's 985/987/988 triple-kill = 3 slots
   spent confirming a theorem from April.)

### Sequencing
1. Now: Track A to completion (days).
2. Paper 1 submitted → Track B batch (4 T0s ≈ 2 days of slot time).
3. step989 outcome decides Paper 2 spine: FFN-replacement (vision completed)
   vs dynamic-routing+hypercomplex (existing Paper 2 scope).

## 6. One-line verdict

Claude Code falsified 1 of 3 core ideas rigorously, exceeded the efficiency ambition,
kept the architecture's geometric identity, and documented nearly every deviation —
but let the original problem statement (transformer FFN) and the v2 roadmap starve
unexamined. The plan above converts each unexamined promise into exactly one
evidence-generating experiment.
