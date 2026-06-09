# AntiHebbian Learning

## What It Is

Lateral inhibition in SGNNET — structurally similar neurons suppress each other during routing. Per-neighbor suppression weight applied before gather step:

```
suppress(h, k) = 1 - alpha * cosine_sim(W_pos[h], W_pos[k]).clamp(0)
Z_struct = (Z_nb * suppress).sum(dim=2)
```

`wpos` variant (static, spatial surround) — cosine similarity in learned position space `W_pos`. Pre-computed once outside K_iter loop. `zact` variant (dynamic, Z-space decorrelation) — similarity in current activation space per step. `wpos` dominates all experiments.

Biological analog: Mexican-hat surround suppression from cortical V1. Neurons with nearby W_pos directions on S^{D-1} contribute less, forcing spatial diversity in learned representations.

Source: `src/sgnnet/mechanisms_inhibitory.py` class `SGNNET_AntiHebbian`.

## Key Parameters

- **alpha (alpha_ahebb)**: Suppression strength. alpha=1.0 → neurons with identical W_pos directions contribute zero (complete co-directional suppression). alpha=0.0 → no suppression (standard sum aggregation).
  - D=16: alpha=0.5 optimal (step16: +8.00pp)
  - D=64 uncalibrated base: alpha=0.5 optimal (step29: 70.14%)
  - D=64 calibrated base: alpha=1.0 optimal (step29c: 80.08%) — monotonic scaling confirmed across {0.1, 0.3, 0.5, 0.7, 1.0}
  - D=64 patched arch, N=4096: alpha=1.0 used in project best (step70: 97.32%)
- **variant**: `wpos` (static spatial surround) vs `zact` (dynamic Z-space). `wpos` wins consistently (step16: 37.22% vs 31.41% at alpha=0.5).

## Confirmed Findings

| Step | Config | Result | Delta | Finding |
|------|--------|--------|-------|---------|
| step70 B | AH=1.0, N=4096, patched arch, 150ep full | **97.32%** | +13.08pp vs old best | **Project best.** turing=0.0, reflect=0.5 |
| step70 Ref | AH=1.0, N=4096, patched arch, turing=0.3 | 97.20% | +12.84pp vs old best | turing slightly harmful at N=4096 |
| step69 Ref | AH=1.0, N=1024, patched arch, 50%/75ep | 83.36% | +9.83pp vs buggy REF | Patch gain: input_coverage (+6.29pp) + alpha_reflect fix (+3.54pp) |
| step29c A | AH=1.0, N=1024, calibrated base, 150ep | 80.08% | +21.40pp vs calibrated Ref | AH alone = Gen4. Compound ceiling (all mechanisms) = 67.72% |
| step29 A | AH=0.5, N=1024, uncalibrated, 150ep | 70.14% | +13.86pp vs D=64 ceiling | First AH result at D=64 |
| step16 H | AH=0.5 wpos, D=16, N=512 | 37.22% | +8.00pp vs Ref | Discovery experiment; second-largest individual gain |
| step16 G | AH=0.3 wpos, D=16, N=512 | 32.87% | +3.65pp vs Ref | Lower alpha = weaker gain |
| step16 I | AH=0.3 zact, D=16, N=512 | 31.41% | +2.19pp vs Ref | `zact` variant inferior to `wpos` |

### Alpha Sweep at D=64 Calibrated Base (step29c Phase 1, 40ep)

| alpha | top1@40ep | Trend |
|-------|-----------|-------|
| 0.1 | lowest | monotonic |
| 0.3 | -- | monotonic |
| 0.5 | -- | monotonic |
| 0.7 | 70.98% | (step29 full-run anchor) |
| 1.0 | highest | monotonic — adopted as Gen4 |

Physical interpretation: at D=64, Fourier encoding creates near-orthogonal neuron directions on S^63. alpha=1.0 → maximal diversity pressure. Calibrated routing params (step22b) unlock stronger AH than uncalibrated base.

## Scale Interactions

### N=1024 vs N=4096

| Property | N=1024 | N=4096 |
|----------|--------|--------|
| AH=1.0, patched, 50%/75ep | 83.36% (step69) | 95.87% (step71 Ref) |
| AH=1.0, patched, 100%/150ep | ~80.08% (step29c, buggy arch) | **97.32%** (step70 B) |
| Optimal K_iter with AH=1.0 | 16 (step68: +0.61pp vs K_iter=8) | 12 (step71: +0.79pp vs K_iter=8) |
| turing interaction | turing=0.3 helps +1.68pp (step69 A) | turing=0.0 wins by +0.12pp (step70) |

K_iter optimal shifts with N: N=1024 peak K_iter=16; N=4096 drops to K_iter=12. Both show non-monotone K_iter curves with sharp cliffs (K_iter=24 costs -3.57pp at N=1024).

### N=10000 Regression (step56)

N=10000 regresses to 82.37% vs N=4096 at 84.36% (buggy arch). Hypotheses: (a) AH pressure saturates at large N with too many competing inhibitory signals, (b) W_pos space too sparse for K_local neighborhoods. N-scaling NOT monotonic above N=4096 on buggy arch. Patched arch N-scaling curve unknown.

### D=64 Specifics

- AH gain at D=64 2x stronger than D=16 (+13.86pp vs +8.00pp at alpha=0.5). Higher-D Fourier encoding gives neurons more distinguishable directions → suppression of nearby directions has more routing alternatives.
- `wpos` dominates because W_pos similarity meaningful on S^63 — enough directions for suppression to be selective rather than uniform.

## Compound Behavior

AH alone optimal. Every compound tested hurt:

| Compound | Step | Best result | Delta vs AH alone | Mechanism |
|----------|------|-------------|--------------------|----|
| AH + phase_exc (alpha=0.3) | step29c E | 66.96% | -13.12pp | Phase_exc re-introduces proximity coupling AH suppresses |
| AH + phase_exc + interneurons | step29c F | 66.93% | -13.15pp | Same as above, interneurons add nothing |
| AH + all mechanisms | step29c G (step32) | 67.72% (73.71%) | -12.36pp | Compound ceiling far below AH alone |
| AH + hub interneurons (fan_in=512) | step61 C | 64.05% | -9pp vs Ref | AH suppresses hub->hidden paths |
| AH + phase-target plasticity | step66 D | 40.33% | -42.85pp vs Ref | AH and phase-target antagonistic |
| AH + distance-phase routing | step60 C, B_anchor | 14-19% | -55pp+ vs Ref | Phase routing incompatible with AH geometry |
| AH + activation-gated routing | step63 B | 55.41% | -18pp vs Ref | Any routing modification destabilizes AH equilibrium |
| AH + low-rank mixing | step53 E | ~69.78% | below Ref | Mixing disrupts Fourier layout AH depends on |

**Root cause**: at alpha=1.0, AH enforces stable fixed point where structurally similar neurons contribute zero. Any additive mechanism re-introducing coupling (excitatory, gating, phase-based) opposes diversity pressure, destabilizes training. AH routing not "improvable" by attention — it IS optimal fixed point for current architecture.

**Exception — alpha_reflect**: reflection accumulator (leaky memory of threshold-suppressed signal, decay=0.5) compounds with AH for +3.54pp (step69 B vs Ref). Reflection NOT additive routing — within-step residual helping AH by giving suppressed activations second chance. Orthogonal axis to AH.

## Signal Conservation Property

AH survives K_iter iterations because suppression applied **before** gather step, and `F.normalize()` at end of each routing step restores magnitude to unit vectors. Net per-step signal conserved. Why AH succeeds where every multiplicative gating mechanism fails ([[gate-death]]).

Gate-death theorem: any per-step multiplicative gate g in [0,1] produces signal proportional to g^K after K steps. At g=0.7, K=8: 0.06x original signal. AH avoids this by modifying **weights** (which neighbor contributes how much) rather than **activations** (how much signal passes through).

## Architectural Bugs (Fixed)

Two bugs silently degraded AH performance before step69:

1. **Input coverage gap** (step69, +6.29pp fix): `_build_fanin_conn` used random per-neuron sampling, leaving ~13% of VGG16 features unreachable. Round-robin coverage guarantee fixed.
2. **alpha_reflect silenced** (step69, +3.54pp fix): `SGNNET_AntiHebbian.forward()` omitted `Z_reflected` accumulator. alpha_reflect=0.5 stored but never applied across K_iter.

All experiments steps 57-68 on buggy code. Relative verdicts (mechanism A vs B) hold, but absolute accuracy values 9-10pp below patched arch potential.

## Scale Transfer Pattern (2026-04-09 analysis)

Mechanisms adding to AH at N=1024 often fail at N=4096. Three confirmed cases:
- Group topology: +3.01pp→0pp (step82→step90)
- W_phase trained: +2.88pp→+0.18pp (step76→step99)
- Turing: +1.68pp→−0.12pp (step69→step70)

Hypothesis: at N=4096 with K_hh=4, connectivity 0.1%. Structure-dependent mechanisms can't differentiate in extreme sparsity. AH fixed-point routing already near-optimal. Signal-operating mechanisms (Z-bias, redistribution) may transfer because topology-density-independent. step115 tests this.

## Diagnostics Findings (step155) [CONFIRMED]

| Config | N | D | AH | eff_rank | trend | grad_theta | grad_wpos |
|--------|---|---|----|----------|-------|------------|-----------|
| A (best arch) | 4096 | 64 | 1.0 | — | — | 0.997 | 0.07 (14:1 ratio) |
| B | 1024 | 16 | 1.0 | 4.6/16 | rising | — | — |
| E (AH=0 control) | 1024 | 16 | 0.0 | 5.2/16 | collapse | — | — |

**AH as anti-collapse mechanism [CONFIRMED clean ablation E vs B]:** Without AH, eff_rank collapses during training. With AH, eff_rank rises. First direct mechanistic evidence for *why* AH works — enforces dimensional diversity by suppressing co-directional neurons.

**W_pos barely learns at N=4096 [CONFIRMED]:** grad_theta/grad_wpos = 14:1. wpos_norm_mean=4.609 static throughout 20ep. W_pos converges early and freezes. Implication: W_pos initialization matters most; fine-tuning W_pos LR unlikely to help; Fourier initialization provides sufficient prior that AH locks in quickly.

**Neuron utilization = 100% across all configs:** No dead neurons. Every neuron load-bearing — confirms why pruning methods (step153) fail catastrophically.

## Open Questions

1. **N-scaling on patched arch**: step72 partial results show steep monotone N-scaling (256→57.9%, 512→74.5%, 1024→87.0%, 2048→93.7%). N=4096 pending.
2. **AH saturation at large N**: does AH pressure saturate when N>>K_local? step90 (group topology NULL at N=4096) suggestive. N=10000 on patched arch untested.
3. **Z-bias + redistribution at N=4096**: step106 Z-bias (+7.42pp at N=1024) and step75 redistribution (+3.98pp at N=1024) most likely to compound with AH at scale. step115 P0.
4. **alpha > 1.0**: never tested. Monotonic alpha curve at D=64 suggests alpha=1.0 may not be true optimum.
5. **F.normalize() vs RMSNorm**: current unit-sphere projection destroys magnitude info every step. If RMSNorm works better, magnitude carries signal AH currently discards (step116).

## See Also

- [[gate-death]] — why every multiplicative routing mechanism fails; AH signal conservation key contrast
- [[k_iter]] — K_iter optimal N-dependent and non-monotone; AH prevents over-smoothing at higher K_iter
- [[n-scaling]] — N-scaling law, AH saturation hypothesis at large N
- [[reflection]] — alpha_reflect accumulator, only mechanism compounding with AH (+3.54pp)
- [[input-coverage]] — bug fix contributing +6.29pp; AH performance silently degraded without it