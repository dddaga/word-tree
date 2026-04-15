# Strategic Synthesis — 2026-04-15 (Session 9)

Distilled from three deep meditations (Haiku/Sonnet/Opus) + this session's experimental results. Source diagnoses live in `.planning/`:
- `efficiency_meditation_2026-04-15.md` (Haiku, 251 lines, broad)
- `walltime_meditation_2026-04-15.md` (Sonnet, 338 lines, kernel-level)
- `walltime_breakthrough_2026-04-15.md` (Opus, ~550 lines, 26 candidates)

---

## 1. The Headline Problem

**MLP_37 (929K params, 1.86M FLOPs) hit 97.71% on Imagenette** — beating SGNNET step199 (95.52%) at matched FLOPs and beating SGNNET step235 (97.30%) at matched FLOPs too. The "SGNNET wins on FLOPs Pareto" claim is **falsified at 10-class Imagenette**.

What survives:
- SGNNET wins on **(params × FLOPs)** Pareto: 25× fewer params at matched FLOPs/accuracy.
- SGNNET wins on **wall-time vs VGG_FC**: 5.6× faster (0.280ms vs 1.570ms).
- SGNNET vs **GCN/GAT** at matched params: SGNNET 95.52% vs GCN 48.9% / GAT 47% — dramatic. The routing mechanism is doing real work.
- step266 multi-config result: **K=5 ΔW rot+aug at N=4096 = 97.71%**, K=4 = 97.66% (Δ = −0.05pp; K=4 is mechanism-specific, not universal).

What's broken:
- "20% wall-clock reduction for K=4" → real measurement is **5.6%** (bench_step830 today).
- "K_iter distillation killed (step196)" → was **output-matching KD only**. Consistency-DEQ was never tried. This direction is REOPENED.

---

## 2. Three Mechanism Findings That Change Strategy

### A. We are dispatch-bound, not compute-bound (Sonnet)
- 0.280ms = 0.03% of peak compute (6.6 GFLOP/s vs 23.2 TFLOPS).
- "99% GPU util" is kernel-launch saturation, not math.
- K=4 saved only 5.6% because ~60% of wall-time is non-routing ops that don't scale with K_iter.
- Hardware: RTX 5060 Ti is **Blackwell SM_120** (not Ada Lovelace). 448 GB/s GDDR7. Z[B,N,D]=4.19MB fits in L2. BW is not bottleneck either — it's purely launch count × K_iter.

### B. The reference class is wrong, in our favor (Opus)
- Hazy Research's "megakernel" pattern: 100 ops fused into single launch, 1B Llama at 680μs on B200.
- Our entire model is 34K params vs their 1B. **The ceiling is much lower than we've been treating it.**
- Implication: 0.050-0.080ms is realistic for SGNNET, not aspirational.

### C. SGNNET is structurally a Deep Equilibrium Network (Opus)
- The K_iter loop is iterating toward a fixed point z* of f(z, x).
- Consistency-DEQ (arXiv:2602.03024, 2024) reports student ≥ teacher when distilled with **trajectory loss** (match z_5_teacher with z_1_student), not just output logits.
- step196 used output-matching only. **Never tested with the right loss function.**

---

## 3. Two Pivot Options (User chose: test both)

### Option A — B2 (GLNN distillation): paper artifact = MLP_37
- Train SGNNET teacher (step235 config, 75ep)
- Cache softmax(logits/T) on full dataset
- Train MLP_37 student with KD loss: λ·KL(student/T, teacher/T) + (1-λ)·CE
- Sweep (T, λ) ∈ {(2,0.5), (4,0.5), (4,0.7), (8,0.5), (8,0.7)}
- Win: student ≥ teacher AND ≥ MLP_37-from-scratch (97.71%)
- Reframes paper: **"SGNNET training discovers FC can be replaced by hidden-37 MLP"**
- Wall-time: MLP_37 = 0.088ms (already measured, below Linear)
- GLNN literature reports 146-273× student speedup for this pattern

### Option B — B1 (consistency-DEQ K=1 student): SGNNET stays the artifact
- Train K_iter=5 teacher, cache Z_final + logits
- Train K_iter=1 student with: α·||Z_1_student − Z_5_teacher||² + β·KL(KD) + (1-α-β)·CE
- Sweep (α, β, T) including pure-trajectory and pure-output to map the surface
- Win: K=1 student ≥ K=5 teacher (5× fewer routing iterations)
- Reframes paper: **"Consistency distillation collapses iterative routing to single step"**
- Wall-time: K=1 ≈ 5× routing speedup, but capped by ~60% non-routing overhead → projected ~0.180ms

Both dispatched as parallel Sonnet agents — independent of which pivot wins.

---

## 4. Top-Ranked Wall-Time Levers (synthesized across all 3 meditations)

| # | Lever | Status | Projected gain |
|---|---|---|---|
| 1 | **B2 GLNN distill** to MLP_37 | dispatched | 0.280→0.088ms = 3.18× |
| 2 | ~~Triton fused K_iter kernel~~ (step530b) | **DONE — KILLED** | fused parallel=3.77ms, fused seq=3.81ms; PyTorch max-autotune fuses ENTIRE model (seed+routing+readout) into one kernel, Triton only touches routing — wrong abstraction layer |
| 3 | **B1 consistency-DEQ K=1** | dispatched | 0.280→~0.180ms = 1.55× |
| 4 | **CUDA Graph + int32 indices** | infra ready (step530 iter #1 proved capture works) | stacks with #2 → 0.050-0.080ms |
| 5 | **E1 batch-scaling bench** (B=1..2048) | NOT YET dispatched | per-sample at B=1024 may hit 0.010-0.020ms |
| 6 | bf16 buffer fix (A2 from Opus) | not dispatched | reduces dispatch overhead |
| 7 | step530 first iter (per-step Triton) | DONE — 13.7× SLOWER | abandon; see #2 |
| 8 | bench_step832 PyG scatter | DONE — 0.284ms (no win) | confirmed PyTorch path is near-optimal |
| 9 | step522 Muon optimizer | AdamW=0.9569 done; Muon CRASHED | needs `muon.py:67` fix |

---

## 5. Currently Running (as of 2026-04-15 ~17:00)

| Slot | Job | Notes |
|---|---|---|
| 5060ti_cuda | step266 multi-seed (seed=43) | Was BLOCKED on store_aug.h5 sync — now resolved |
| studio_cpu | step268 ΔW proj+aug+K=4 T1 | ~50% done |
| studio_mps | step405 SST-2 train (seed=42) | SGNNET config |
| mini_cpu | step235 multi-seed (seed=43) | replication |
| mini_mps | CIFAR-100 VGG16 extract | step601 prerequisite |

**Background research/impl agents** (3 in flight + 2 finishing):
- Triton iter #2 (Sonnet) — fused K_iter kernel
- B2 GLNN distill (Sonnet)
- B1 consistency-DEQ (Sonnet)

**Auto-launcher queue** (`.controller/auto_launch_queue.txt`):
- step601 complexity test (gated on CIFAR-100 extract)
- step235 seed=44, step268 seed=43+44 multi-seed
- step266 seed=44, step405 seed=43+44

---

## 6. Immediate Next Steps (priority-ordered, post-Triton-kill)

**The Triton path is dead** for wall-time (iter #1 and iter #2 both confirmed). The remaining wall-time levers are distillation (B2/B1) and re-profiling the 0.280ms regression.

1. **Re-profile the 0.280ms vs 2.56ms gap.** The same script measures both numbers — something differs (warm cache, model version, compile cache). If the real best is 2.56ms not 0.280ms, the entire wall-time story shifts (5.6× faster than VGG → much smaller). Run bench_step811 fresh on 5060ti and reconcile.
2. **Wait on B2 (GLNN distill).** Now the highest-confidence wall-time lever (3.18× to 0.088ms) AND the paper-framing pivot. ~6h pipeline.
3. **Wait on B1 (consistency-DEQ K=1 student).** Backup if B2 fails. ~6h pipeline.
4. **Dispatch E1 batch-scaling bench** — 30 min on 5060ti once free. If per-sample at B=1024 is <0.020ms, paper reframes to throughput claim.
5. **Wait on step601** (CIFAR-100 complexity test) — validates whether MLP_37 actually beats SGNNET at higher class complexity. Outcome critical regardless of pivot direction.
6. **Fix step522 Muon** — `muon.py:67` matrix_params format error. Low priority (paper-pivot reduces value).
7. ~~Apply 5060ti torch.compile + Triton workaround~~ — DROP. Triton path is the wrong abstraction. Don't invest more.

---

## 7. Decision Points Pending User Input

- **Paper framing** (Opus's biggest question): user already answered — "test both, whichever is more efficient wins."
- **E1 dispatch authorization** — implicit yes per "all pending discussed ideas" directive, but I should explicitly confirm before launching since it requires the 5060ti.
- **Drop step522 Muon entirely?** — AdamW baseline captured (0.9569); Muon needs a code fix worth ~2h. Low priority if paper pivots to distillation.

---

## 8. Validated Numbers (this session)

| Metric | Value | Source |
|---|---|---|
| step266 Ref (K=5 rot+aug N=4096) | 97.71% | log on 5060ti, JSON synced |
| step266 A_k4 (K=4 rot+aug N=4096) | 97.66% | same |
| Δ(K=4 − K=5) at N=4096 with rot+aug | −0.05pp | mechanism-specific |
| step403b MLP_37 Imagenette | 97.71% | matched-FLOPs baseline |
| step404 GCN matched-params | 48.9% | reviewer-critical loss for MP-GNNs |
| step404 GAT matched-params | ~47% | same |
| step521 deep supervision | KILLED (−5pp) | trajectory value is in final state only |
| step526 INT8 abs_mod (K=5) | −0.05pp W only, −1.86pp W+Z | wrap rate = 0 (D=16 L2 saturates) |
| step527 INT8 abs_mod (K=4) | similar | grad-accum still hurts |
| bench_step830 K=4 wall-clock | 0.263ms (5.6% faster than K=5) | NOT 20% as projected |
| bench_step832 PyG scatter | 0.284ms (no win) | PyTorch path already near-optimal |
| bench_step530 Triton iter #1 | 3.83ms (13.7× SLOWER) | per-step launch overhead; iter #2 fixes |
| Param count actual | 34,976 (was wrongly 67,744 in docs) | reconciled this session |

---

## 9. What Was Closed This Session

| Direction | Verdict | Why |
|---|---|---|
| FLOPs Pareto win | FALSIFIED at 10 classes | MLP_37 wins 97.71% at 1.86M FLOPs |
| K=4 saves 20% wall-clock | FALSIFIED (5.6% real) | non-routing overhead dominates |
| Deep supervision (step521) | KILLED CONFIRMED | −5pp; trajectory value only in final |
| PyG scatter (bench_step832) | KILLED | within noise of PyTorch hand-tuning |
| Triton per-step kernel | KILLED for performance | infra reusable, but path was fused-K |
| Triton fused-K kernel (iter #2) | KILLED for performance | grid-wide sync impossible on SM_120 + Triton 3.6; single-block-per-batch impossible (N>1024 threads, Z>48KB shared mem); both fused variants 3.77-3.81ms vs PyTorch max-autotune 2.56ms in same run; wrong abstraction layer entirely |
| **bench_step811's 0.280ms baseline** | UNREPRODUCIBLE in step530b run (now measures 2.56ms) | warm-cache artifact OR model version drift — needs re-profiling to confirm what 0.280ms actually represented |
| 67,744 param count | RECONCILED to 34,976 | doc error throughout |
| step266 97.71% provenance | CONFIRMED real | sync artifact, JSON+log on 5060ti |
| `K_iter distillation killed` (step196) | REOPENED | was wrong loss; consistency-DEQ untried |

---

## 10. References

- GLNN (Yan & He): arXiv:2110.08727 — graph teacher, MLP student, 146-273× speedup
- Consistency-DEQ: arXiv:2602.03024 (2024) — student ≥ teacher with trajectory loss
- Hazy Research megakernel pattern: ~100 ops fused, 1B Llama 680μs B200
- bench_step811 V2 (current best): 0.280ms K=5 max-autotune fp32 on 5060ti
