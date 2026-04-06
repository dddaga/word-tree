# SGNNET Experiment Queue

**GA rule (STRICT):** Every new experiment base = ALL confirmed winners from all prior generations. Ablate only the variable being tested.
**Calibration rule:** When base changes generation/scale, run a 40-50ep param sweep BEFORE full 150ep runs. Params (alpha, tau, K, beam_size) are NOT scale-independent.
**Logging:** Results → `learnings/LEARNINGS_phase5_p*.md`. Step details in scripts.

---

## ⚠️ Critical Findings

| Finding | Detail |
|---------|--------|
| **Signed coupling DEAD at D=64** | 5 experiments confirm ≤32% at D=64 any α/K_iter. cos-sim on S^63 ≈ noise. **Do not test again.** |
| Signed coupling + K_iter≥8 collapses | Power iteration → dominant eigenvector. K_iter=3 is stability limit — but K_iter=3 D=64 also fails vs ref |
| **alpha_reflect=0.5 wins at D=64** | step22b calibration confirmed: alpha_reflect=0.5 → 52.94% at 40ep vs 0.3 default |
| **D=64 is the encoding ceiling** | D=128 non-viable: all LR/schedule combos stuck at ~10%. Fourier encoding on S^127 collapses. |
| **AntiHebb α=0.5 at D=64 = 70.14%** | step29 Config A FINAL: **+13.86pp** over D=64 ceiling. Superseded — see below. |
| **⭐ AntiHebb α=1.0 calibrated base = 80.08%** | step29c Config A Phase 2: **NEW ALL-TIME BEST**. +21.40pp vs calibrated Ref. On step22b routing params. Adding phase_exc or interneurons KILLS gain (→67%). AntiHebb α=1.0 ALONE is Gen4. Gen4 compound ceiling (all mechanisms) = 67.72% (config G). |
| **step57 REF_BASELINE = 73.53%** | AntiHebb α=1.0, 50% data, 75ep. Comparison point for steps 58-61. Full-scale = 80.08%. |
| **⭐ step56 N=4096 = 84.36% — PROJECT BEST** | Full data 150ep, Gen4 params, best_ep=146/150, params=529K. N-scaling law: 512→69.58%, 1024→80.92%, 2048→81.10%, 4096→84.36% (PEAK), 10000→82.37% (REGRESSES). N-scaling is NOT monotonic above N=4096. N=10000 worse by 2pp. Do not pursue N>4096 without understanding reversal mechanism. |
| **Wave-1 VERDICT: ALL mechanisms KILLED** | Steps 58-63 complete. Gate death on every new mechanism. Static AH routing remains best. AGR (step63): best B/D/E=55%, A=31%, C=39% — all -18pp+ vs 73.48% Ref. |
| **step64 stacked SGNNET queued** | 7 configs (Ref, A-F) ablating series depth, skip connections, linear bridge, parallel fusion. N=1024 per layer, 50%/75ep. |
| **step65 KILLED: distance-phase routing** | exp(-γ·d_norm) on conn_hh, γ∈{0.5,1.0,2.0}×{raw,softmax}. All A-F at ~48-49% vs REF=73.53% (−25pp). Distance geometry adds nothing AH doesn't already provide. |
| **step60 KILLED: phase-distance routing (all 13 configs)** | Ref=73.55%, all 12 other configs gate-dead 10-19%. Magnitude-weighted, coherent, absolute, AH variants — all killed. Comprehensive closure on phase-routing modifications. |
| **step54 KILLED: LR schedule sweep** | Plateau=70.78% (Ref), CosineWarmRestart T_0={10,25,50}=66-68%. Cosine warm-restarts hurt. Plateau is optimal. Closed — step56=84.36% with plateau confirms. |
| **step48 SLOT-KILLED: K_iter at D=64** | Only Ref(58.24%) and A(K_iter=12,55.08%) ran — B-G never launched (slot pressure). K_iter>8 WITHOUT AH hurts. K_iter>8 WITH AH=1.0 untested → step68 rewrites at Gen4. |
| **Signed coupling permanently dead** | step49 closed: best 42.14% vs 46.85% Ref across all K_iter. Architecturally incompatible with D=64. Do not re-propose. |
| **step55 grouped input: Config A = 89.43%** | 7-row separate projection (overlap=0%) = 89.43% at 3.8M params. Config D (positions_separate, 552K params) = 84.10% — best param efficiency. Grouped projection is a massive gain but at high param cost. |
| **step53 low-rank mixing: KILLED** | rank=4: 69.15%, rank=8: 70.17% — both below REF 73.53%. Cross-dim mixing at any rank hurts Fourier geometry. Do not re-propose. |
| **step50 spatial dynamic W_pos: neutral/negative** | All dynamic K-NN configs ≤55.64% vs static 57.48%. Dynamic W_pos connectivity does not improve over static small-world. |
| K_iter=3→8 gap = ~15pp at D=64 | Confirmed ×3. All 8 routing iterations are necessary; MoD early exit kills performance. |
| MoD adaptive depth fails | step34 KILLED: 19-20% through e110. All K_iter steps contribute; early exit destroys iterative refinement. |
| Oja's rule routing update fails | step41 KILLED: 24% at e60 vs 56.79% Ref. PCA compression destroys directional diversity across K_iter. |
| Dynamic Z-KNN underperforms static | step31 provisional (~87% done): ~44% vs 56.28% static ceiling. Unstable K-NN on S^63 per step. |

---

## New Architecture Experiments (Steps 57-61)
**Ablation setup: 50% data, 75ep. All compare against step57 fast benchmark.**
**Final winners re-run at 100% data, 150ep vs 80.08% baseline.**

| Step | Script | Description | Status | Priority |
|------|---------|-------------|--------|----------|
| step57 | train_step57_benchmark_ablation | Fast benchmark anchor: AntiHebb α=1.0 calibrated base, 50% data 75ep | **COMPLETE** — REF_BASELINE = **73.53%** | **CRITICAL — runs first** |
| step58 | train_step58_resonance_excitatory | Resonance-gated phase-excitatory: per-batch K-NN + activation gate vs static | **COMPLETE — KILLED** (~55% best) | HIGH |
| step59 | train_step59_active_beam_unified | Beam unification: beam gates both structural + phase channels | **COMPLETE — KILLED** (~55% best) | HIGH |
| step60 | train_step60_phase_routing_magnitude | Phase-distance routing: 13 configs, all variants | **COMPLETE — KILLED** (Ref=73.55%, all 12 others gate-dead 10-19%) | HIGH |
| step61 | train_step61_hub_interneurons | Hub interneurons: high fan-in mixing layer (N_mix=256, fan_in=512) | **COMPLETE — KILLED** (best C=64.05%, Ref=73.30%) | HIGH |
| step63 | train_step63_act_gated_routing | Activation-Gated Routing (AGR): soft-attn over expanded candidate set | **COMPLETE — KILLED** (best B=55.41%, Ref=73.48%) | HIGH |
| step64 | train_step64_stacked_sgnnet | Stacked SGNNET: 7 configs (Ref, A-F) — series depth, skip, bridge, parallel fusion | **RUNNING** (Mac Mini CPU) — relaunched after W_phase fix | HIGH |
| step65 | train_step65_dist_phase_routing | Distance phase routing: exp(-γ·d) on conn_hh. No AH. | **COMPLETE — KILLED** (A-F all 48-49% vs REF=73.53%) | HIGH |
| step66 | train_step66_phase_target | Phase-target routing: W_pos=Key/Value (gradient), phase_target=Query (local plasticity). Per-batch K/2 attract/repel. Diversity penalty (β). AH compound. | **QUEUED** — launch on Mac Mini MPS when step60 frees | HIGH |
| step67 | train_step67_safety_ablation | Safety valve λ ablation with AH=1.0: λ∈{0.489,0.0,0.05,0.1} | **PARTIAL** — Ref=73.68%, A(λ=0)=70.78% (−2.9pp); B(λ=0.05) running, C queued | MEDIUM |
| step68 | train_step68_kiter_gen4 | K_iter scaling at Gen4: K_iter={8,10,12,16,24}+AH=1.0. 50%/75ep. Does AH prevent over-smoothing at depth? | **RUNNING** (Mac Studio MPS — launched 2026-04-06 when step56 freed) | HIGH |

**REF_BASELINE (steps 58-61):** 73.53% (AntiHebb α=1.0, 50% data, 75ep). Full-scale equivalent: 80.08%.
**Wave 2**: compound winners from 58-61, full data 150ep, vs 80.08%.
**AntiHebb ablation for new arch**: after Wave 2 concludes.

---

## Currently Running (cap: 4)

| Machine | Step | Script | Status | Note |
|---------|------|--------|--------|------|
| Mac Studio MPS | step68 | train_step68_kiter_gen4 | Ref=73.63%, A(K_iter=10)=73.58% DONE (−0.05pp, flat); B(K_iter=12) running | K_iter=10 no gain vs K_iter=8 |
| Mac Studio CPU | step67 | train_step67_safety_ablation | Ref=73.68%, A(λ=0)=70.78% DONE (−2.90pp); B(λ=0.05) running | Safety valve IS needed; AH doesn't substitute |
| Mac Mini MPS | step60 | train_step60_phase_routing_magnitude | Still running (E config ~e50, gate-dead pattern) — step66 queued behind | All configs gate-dead, finishing run |
| Mac Mini CPU | step64 | train_step64_stacked_sgnnet | Ref=73.38%, A(no-skip)=71.62% DONE (−1.76pp); B(skip) at ~e20 | Skip conn is key test; A loss not catastrophic vs wave-1 |

---

## ARM 0 — Base Calibration (PREREQUISITE)

Must complete before Gen4 compound is meaningful.

| Step | Description | Status | Priority |
|------|-------------|--------|----------|
| step22b | Calibrate base routing params at D=64 N=1024 K_iter=8 | RUNNING (PID 91944) | **CRITICAL** |

**step22b sweeps (40ep each):**
```
alpha_turing:  {0.0, 0.1, 0.3*, 0.5, 1.0}   (* = D=16 default)
alpha_reflect: {0.0, 0.1, 0.3*, 0.5}
K_phase:       {4, 8*, 16, 32}
beam_size:     {16, 32*, 64, 128}   (32* = 6% N=512, now 3% N=1024)
geo_gamma:     {0.0, 0.5, 1.0*, 2.0}
Phase 2 (150ep): Ref + best of each individually + ALL best combined → true D=64 ceiling
```

---

## ARM 1 — Generational Compounding (GA)

| Step | Description | Depends on | Priority |
|------|-------------|-----------|----------|
| step29 ✓ | AntiHebb α sweep at D=64 K_iter=8 (uncalibrated base) | RUNNING | — |
| step29b ✓ | phase_exc + interneurons + fast_W_phase calib+full (uncalibrated base) | RUNNING | — |
| step22b ✓ | Calibrate base routing params at D=64 | RUNNING (PID 91944) | **CRITICAL** |
| step29c ✓ | Re-validate ALL mechanisms on calibrated base from step22b | COMPLETE — Config A (AntiHebb α=1.0) = **80.08%** best; Config G (all) = 67.72%; AntiHebb alone wins | DONE |
| step32 | Gen4 compound: step29c winners stacked | after step29c | NEXT |

**step29c design (script: `train_step29c_mechanisms_calibrated.py`):**
```
Loads best_params from results/train_step22b_routing_calib_d64.json automatically.
Phase 1 (40ep): calibrate alpha for each mechanism at calibrated base
  - AntiHebb: α ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
  - phase_exc: α ∈ {0.1, 0.3, 0.5, 1.0}
  - fast_W_phase: (α, τ) ∈ {0.1×0.25, 0.3×0.25, 0.1×0.5, 0.1×1.0}
  - interneurons: frac ∈ {25%, 50%, 75%}
Phase 2 (150ep): Ref + A(AH) + B(exc) + C(int) + D(fp) + E(AH+exc) + F(AH+exc+int) + G(ALL)
Output: results/train_step29c_mechanisms_calibrated.json
```

**step32 design (after step29c complete):**
```
Ref     calibrated D=64 base (from step22b best)   [step29c Ref result]
A–G     step29c winners stacked progressively
```

---

## ARM 2 — New Theory / New Mechanisms

All run at D=64 K_iter=8 base (unless noted). Results feed into step32 if they win.

| Step | Description | Status | Priority |
|------|-------------|--------|----------|
| step30 ✓ | Cross-dim W_mix [D×D]: shared/per-neuron/once | DONE — neutral D=16; hurts D=64 | — |
| step31 ✓ | Dynamic Z-KNN per routing step | RUNNING e140 (~44%) — underperforming static; see ARM 3 | **CRITICAL** |
| step33 ✓ | D=128 extension | KILLED — all configs ~10%, architecturally non-viable | DONE |
| step34 ✓ | MoD adaptive K_iter | KILLED e110 — stuck 19-20%, all 8 steps necessary | DONE |
| step36 ✓ | Input-gated adjacency W_gate[N,D] | RUNNING; config 1 done 58.62% ← above ceiling | HIGH |
| step37 ✓ | Phase-queried D×D matrix bank | QUEUED (was e140, 51.2% — killed; still failing vs baseline) | LOW |
| **step48** | **K_iter scaling sweep** | ❌ **SLOT-KILLED** (Ref=58.24% only, B-G never ran) → superseded by **step68** at Gen4 | CLOSED |
| **step52** | **High-D routing** | ❌ **COMPLETE — KILLED** (gate-dead ~45%) | CLOSED |
| **step53** | **Low-rank + frequency-group cross-dim mixing** | **COMPLETE — A=69.15%, B=70.17%, KILLED** | **MEDIUM** |
| **step54** | **LR schedule sweep** | ❌ **COMPLETE — KILLED** (plateau=70.78% wins; cosine WR=66-68%, all below Ref) | CLOSED |
| step35 | MatFormer nested N training: D=64 N=2048 | PLANNED | LOW |
| **step57** | **Beam routing speed/accuracy trade-off at D=64 + AntiHebb** | **PLANNED** | **MEDIUM** |

**step48 design — K_iter scaling at D=64:**
```
Motivation: D=64 K_iter=3→8 gain = +16pp (3× D=16 gain). Over-smoothing threshold
likely shifts higher on S^63. AntiHebb may prevent collapse at deep K_iter.

Base: D=64 N=1024 alpha_reflect=0.5 (step22b winner)
Ref    K_iter=8   no AntiHebb  [~56.28%]
A      K_iter=12  no AntiHebb
B      K_iter=16  no AntiHebb
C      K_iter=24  no AntiHebb
D      K_iter=32  no AntiHebb
E      K_iter=16  + AntiHebb α=0.5  [depth × inhibition compound]
F      K_iter=8   + AntiHebb α=0.5  [anchor: replicates step29A ~70.14%]
G      K_iter=32  + AntiHebb α=0.5  [extreme depth + inhibition]

Compute cost: scales linearly with K_iter. K_iter=32 ≈ 4× longer than K_iter=8.
```

**step52 design — High-D geometry routing:**
```
Hypothesis: Exploit near-orthogonality as capacity rather than fighting it.
Base: D=64 N=1024 K_iter=8 + AntiHebb α=0.5 (70.14%)

Ref   AntiHebb α=0.5 uniform routing  [~70.14%]
A     + Z-subspace routing split=16  (gate by cos(Z[0:16]) in routing subspace)
B     + Z-subspace routing split=32  (gate by cos(Z[0:32]))
C     + W_pos-subspace routing split=16  (gate by cos(W_pos[0:16]))
D     + W_pos-subspace routing split=32  (gate by cos(W_pos[0:32]))
E     + projection routing |Z_i · normalize(W_pos_j)|  (diffraction gate)
F     + centering diversity λ=0.05  (subtract mean activation each step)
G     W_pos-subspace split=16  NO AntiHebb  [ablation: gate vs subtract?]

Key questions:
  C/D > Ref → structural subspace gate compounds with AntiHebb
  E > Ref   → diffraction (cross-modal Z·W_pos) gate works
  G ≈ Ref   → structural gate REPLACES AntiHebb (more efficient)
```

**step53 design — Low-rank + frequency-group cross-dim mixing:**
```
Hypothesis: Full D×D cross-dim mixing hurts (steps 30, 37). Does LOW-RANK mixing
work? And does mixing within Fourier frequency PAIRS (2×2 mixing) preserve structure?

Background: step30 (shared D×D): neutral D=16, -15pp D=64.
            step37 (phase-queried D×D): -5pp D=64. All full D×D fails.
Why it fails: D×D mixes all 64 dimensions indiscriminately, scrambling Fourier structure.
Why low-rank might work: R=4 mixing uses only D×R + R×D = 128+128 params (vs D²=4096).
Preserves most frequency structure while allowing limited cross-band communication.

Base: D=64 N=1024 K_iter=8 + AntiHebb α=0.5 (70.14%)

Ref   AntiHebb α=0.5 uniform  [~70.14%]
A     + low-rank mixing U·V^T where U,V ∈ R^{D×4}  (rank-4)
B     + low-rank mixing rank-8
C     + frequency-pair mixing: 2×2 within each (dim_2k, dim_2k+1)  (D/2=32 pairs)
D     + group mixing: 8 groups of 8 dims, 8×8 within each group
E     + frequency-pair mixing + AntiHebb only  (C + AH anchor)
F     low-rank rank-4 WITHOUT AntiHebb  [ablation: mixing vs inhibition]
```

**step57 design — Beam routing speed/accuracy trade-off at D=64 + AntiHebb:**
```
Context: step25 tested beam routing at D=16 K_iter=3 and found route=64 gave +2.88pp
over K_iter=3 full routing (53× speedup) — but was never tested against the real
K_iter=8 baseline (36.69%). At D=16 it lost ~7pp vs K_iter=8. Never tested at D=64.

Hypothesis: At D=64, the Fourier geometry on S^63 concentrates routing signals more
sharply — AntiHebb + high-D may tolerate sparse beam routing with smaller accuracy loss
than D=16. If route=128 (12.5% of N) loses <2pp vs 70.14%, that's an enormous
compute saving for deployment.

Base: D=64 N=1024 K_iter=8 + AntiHebb α=0.5 (70.14%)
Phase 1 (40ep calibration): find accuracy cliff
  Ref    route=1024 (full N)         [~70.14% anchor]
  A      route=512  (50% of N)
  B      route=256  (25% of N)
  C      route=128  (12.5% of N)
  D      route=64   (6.25% of N)     [step25 winner at D=16]
  E      route=32   (3% of N)

Phase 2 (150ep full): Ref + top-2 configs from Phase 1 + ablation without AntiHebb
  F      best_route  NO AntiHebb     [does AH matter more/less at sparse routing?]
  G      best_route  K_iter=12       [can more depth compensate for sparse routing?]

Key metrics:
  - Accuracy vs full N: target <2pp loss at best route size
  - Speed: routing ops ∝ route_size × K × D (linear)
  - Decision rule: any route < 256 with < 3pp loss → viable for deployment path
```

---

## ARM 3 — Dynamic Connectivity Research

**Goal:** Prove sparse O(N×K) input-dependent topology matches O(N²) signed coupling.

| Mechanism | Cost | Input-dependent? | Gain (D=16 ref) |
|-----------|------|-----------------|---------|
| Static conn_hh | O(N×K×D) | No | base |
| conn_phase K-NN on W_phase | O(N×K×D) | No | neutral alone |
| Fast W_phase attention | O(N×K×D) | Partial | +1.84pp |
| Signed coupling (Z Z^T) | O(N²×D) | Yes (fully) | +10.93pp ⭐ |
| Sparse K-NN on W_phase (step24) | O(N×K×D) | No (static) | +4.05pp (37% N²) |
| **Dynamic Z-KNN (step31)** | O(N×K×D + sim) | **Yes** | **??? ← running** |
| Input-gated (step36) | O(N×K×D) | Yes (per recv Z) | **??? ← running** |

**Key metric for step31:** `vs_N² recovery = (gain / 10.93pp) × 100%`. Target >80% at K=8.

| Step | Description | Status | Priority |
|------|-------------|--------|----------|
| step26 ✓ | FAISS conn_phase rebuild frequency | DONE — all neutral ±0.5pp; per-epoch rebuild is fine | DONE |
| step31 ✓ | Dynamic Z-KNN per routing step | RUNNING e140 ~44% vs 56.28% static — underperforming | **CRITICAL** |
| step36 ✓ | Input-gated adjacency | RUNNING; config 1 done 58.62% (+2.34pp vs ceiling!) | HIGH |
| **step49** | **Signed coupling × K_iter threshold: K_iter={3,4,5,6,7,8} — find sweet spot** | ❌ **COMPLETE — KILLED** (best 42.14% vs 46.85% Ref; all configs below Ref; ARM closed permanently) | **CLOSED** |
| **step50** | **Spatial dynamic W_pos K-NN connectivity (original vision)** | **COMPLETE** (Ref=57.48%; all dynamic configs ≤55.64%, neutral-to-negative; dynamic W_pos K-NN does not improve over static) | DONE |
| **step51** | **Spatial W_pos K-NN + W_phase gating** | ❌ **CLOSED** — wave-1 verdict: routing gate modifications = gate death | CLOSED |
| step38 | Distill dynamic Z-KNN → static | ❌ **CLOSED** — step31 killed at ~44%, nothing to distill | CLOSED |

**step49 design — Signed coupling × K_iter threshold sweep:**
```
Question: All prior signed coupling failures at D=64 used K_iter=3 (too shallow) or
K_iter=8 (power-iteration collapse). Is there a viable K_iter=4-7 where:
  - Routing has built up non-trivial cosine similarities on S^63 (from ~0.016 at init)
  - Not yet at the collapse threshold

Design: 40-epoch cosine-LR calibration runs (plateau scheduler caused LR collapse in step42)
Ref    K_iter=8  no signed          [~42% at 40ep]
A      K_iter=3  signed α=0.3       [step42 replication: ~25%, confirms fail]
B      K_iter=4  signed α=0.3
C      K_iter=5  signed α=0.3
D      K_iter=6  signed α=0.3
E      K_iter=7  signed α=0.3
F      K_iter=8  signed α=0.3       [step23 replication: collapse <15%]
G-I    α sweep at best K_iter (conditional on Phase 1 finding)

Decision rule: any config > Ref_40ep (~42%) → viable, deserves 150ep full run.
Collapse signature: top1 < 15% at config X → X is the threshold.
```

---

## Completed Results Reference

Full details: `learnings/LEARNINGS_phase5_p6_gen2_gen3.md`

| Step | What | Best result | vs ref |
|------|------|------------|--------|
| step9  | D=16 baseline | 29.22% | — |
| step11 | Routing modes | theta-only best | simpler wins |
| step12 | N scale | N=1024→34.32% | +5.10pp |
| step13 | K_iter depth | K_iter=8→36.69% | +7.47pp |
| step14 | Conduction+excrad | neutral | 0pp |
| step15 | Hard beam | 19.36% | -9.86pp FAIL |
| step16 | Inhibition | AntiHebb α=0.5→37.22% | +8.00pp ⭐ |
| step17 | Fast W_phase | Attention→31.06% | +1.84pp |
| step18 | Signed coupling | α=0.3→40.15% | +10.93pp ⭐ |
| step19 | Phase excitatory | K=8 α=0.3→32.20% | +2.98pp |
| step20 | Interneurons | 50% readout=all→32.05% | +2.83pp |
| step22 | D extension | D=64 N=1024→56.28% | +19.54pp ⭐⭐ |
| step23 | Signed×scale | N=1024 K_iter=3→45.55% | K_iter≥8+signed=COLLAPSE |
| step24 | Sparse signed | K=8→33.27% | 37% N² recovery |
| step25 | Beam routing | route=64→29.73% | +2.88pp 53× speedup |
| step27 | Soft beam | all fail | P7 rejected |
| step28 | Gen3 compound | Ref=32.15% (RUNNING) | signed+D64 incompatible |

---

## ARM 4 — Gap-Bridging (InfraNodus-derived, 2026-03-31)

Source: `python infranodus_trial/v1_baseline/run.py` — structural holes in the research concept graph.
These experiments bridge disconnected knowledge clusters to find compounding gains.

| Step | Description | Bridges gap | Status | Priority |
|------|-------------|-------------|--------|----------|
| step39 ✓ | Mechanism-aware auxiliary loss: phase coherence + inhibition sparsity | loss/task ↔ mechanisms | QUEUED (Ref=56.79% done; A-D configs killed at e50, 45.3%) | HIGH |
| step40 | Attention distillation: ViT attention → SGNNET routing topology | architecture ↔ task | QUEUED (needs ViT teacher first) | LOW |
| step41 ✓ | Oja's rule routing update: PCA-like Z update replacing ad-hoc sum | hebbian/oja ↔ implementation | ❌ KILLED e60 — 24% vs 56.79% Ref; PCA destroys routing diversity | CLOSED |
| step42 ✓ | Signed coupling alpha calib at D=64 K_iter=3 (0.005→0.3 sweep) | G1: signed×D=64 | ❌ KILLED — max 25.43%; LR collapses; signed coupling architecturally dead at D=64 | CLOSED |
| step44 ✓ | Beam-restricted signed coupling O(beam²) vs O(N²) | G4: beam×signed sparse | ❌ KILLED e70 — 31.57% flat; confirms signed dead regardless of sparsity | CLOSED |
| step45 ✓ | Safety valve redesign for D=64 (variance/spectral reg) | G7: dead safety valve | QUEUED (killed at e100, 53.8%) | MEDIUM |
| step46 ✓ | Reconnect W_phase to routing in dynamic_z_geo | G11: W_phase disconnected | QUEUED (killed at e60, 52.9%) | HIGH |
| step47 ✓ | Comprehensive interneuron sweep D=64 + AntiHebb compound | G8 + interneuron sweep | QUEUED (killed at e100, 53.8%) | HIGH |

**step39 design — Mechanism-aware auxiliary loss:**
```
Gap: loss function cluster disconnected from mechanism experiments.
All experiments use the same loss = task_loss + safety_valve + load_balance.
Mechanisms modify the forward pass but never the loss.

Hypothesis: auxiliary losses that directly reward mechanism behavior can unlock
gains that forward-pass-only mechanisms miss.

Base: D=64 N=1024 K_iter=8 (calibrated from step22b)
Ref    standard loss (task + safety + LB)
A      + phase coherence loss: encourage W_phase neighbors to have correlated
         activations. L_phase = -mean(cos_sim(Z[j], Z[K_phase_neighbors(j)]))
         λ_phase ∈ {0.01, 0.1, 0.5}
B      + inhibition sparsity loss: reward routing steps that zero out neurons.
         L_sparse = -mean(fraction_below_theta per step)
         λ_sparse ∈ {0.01, 0.1, 0.5}
C      + routing diversity loss: penalise all neurons routing to same neighbors.
         L_div = -entropy(softmax(Z @ Z.T / sqrt(D))) averaged over neurons
         λ_div ∈ {0.01, 0.1}
D      A+B combined at best λ values
```

**step40 design — Attention distillation:**
```
Gap: architecture cluster disconnected from training objective.
SGNNET routing topology (conn_hh + dynamic routing) should learn to approximate
what a transformer attention head learns — but there's no explicit loss driving this.

Hypothesis: distilling a small pretrained ViT's attention patterns into SGNNET's
routing weights provides a strong inductive bias for the routing topology.

Base: D=64 N=1024 K_iter=8 (calibrated)
Teacher: ViT-Tiny or ViT-Small pretrained on same FashionMNIST task
Distillation: after each routing step, compute soft attention map from Z (cosine sim),
KL-diverge it toward teacher attention map.

Ref    no distillation [calibrated base]
A      + KL distillation λ=0.1 from ViT-Tiny last-layer attention
B      + KL distillation λ=0.5
C      + MSE on routing adjacency matrix (softer target)
D      + distill then fine-tune (two-phase: 50ep distill, 100ep task-only)

Note: requires training a small ViT first. If ViT accuracy < SGNNET, distillation
may hurt — run ViT baseline first as sanity check.
```

**step41 design — Oja's rule routing update:**
```
Gap: Hebbian/Hopfield/Oja learning rules isolated from practical implementation.
step16/step29 tested Anti-Hebbian, but Oja's rule (normalized Hebbian for
PCA-like principal component extraction) was never implemented.

Hypothesis: replacing the ad-hoc routing update Z = normalize(Z_struct + α * Z_inh)
with Oja's update rule produces a more principled feature extraction per routing step:
  ΔZ_j = η * (Z_j · Z_struct_j - (Z_j · Z_struct_j)² · Z_j)
This naturally learns the principal component of incoming signals.

Base: D=64 N=1024 K_iter=8 (calibrated)
Ref    standard routing update (Z_struct + alpha_turing * Z_inh)
A      Oja update η=0.1 (conservative)
B      Oja update η=0.3
C      Oja update η=1.0 (aggressive)
D      Hybrid: Oja on Z_struct, keep Z_inh additive (combine both rules)
E      Hopfield energy: Z_new = sign(W_eff @ Z_old) — associative memory step
         (requires binarization or soft-sign; may need different D)
```

---

## Open Questions (Priority Order)

**✅ CLOSED:**
- ~~Does D=128 continue the D trend?~~ → **NO.** Encoding non-viable at D=128 (10%, random baseline). (step33)
- ~~Does cross-dim W_mix help?~~ → **NO.** Neutral at D=16, hurts at D=64 (−15pp). (step30)
- ~~Does MoD adaptive depth match fixed K_iter=8?~~ → **NO.** 19-20% stuck; all 8 steps necessary. (step34)
- ~~Can Oja's rule replace the routing update?~~ → **NO.** −32pp; PCA compression destroys diversity. (step41)
- ~~Does signed coupling work at D=64 at any α?~~ → **NO.** 5 experiments, max 32%, cos-sim is noise on S^63. (steps 18, 23, 28, 42, 44)
- ~~Does beam-restricted signed coupling fix the D=64 issue?~~ → **NO.** Sparsity doesn't fix the mechanism. (step44)
- ~~Does AntiHebb compound with D=64 K_iter=8?~~ → **YES. 70.14% (+13.86pp). Largest gain ever.** (step29)

**🔴 ACTIVE — must resolve before Gen4:**
1. **What are the remaining calibrated base params at D=64?** (step22b — alpha_reflect=0.5 ✓; K_phase, beam_size, geo_gamma pending)
2. **Do Gen1 mechanism winners (phase_exc, interneurons, fast_W_phase) survive at D=64 K_iter=8?** (step29b — strong signal: 49.73% at 40ep)
3. **Does input-gated adjacency beat the D=64 static ceiling?** (step36 — 58.62% for config 1; gated configs running)
4. **Does dynamic Z-KNN recover N² coupling gain at D=64?** (step31 — provisional: ~44% vs 56.28%; likely NO)

**🟡 ACTIVE — informing Gen4 design:**
5. **Does phase matrix bank (step37) beat the D=64 ceiling?** (oscillating ~50-53%, still climbing)
6. **Does mechanism-aware auxiliary loss push past ceiling?** (step39 — Ref=56.79%; A-D configs running)
7. **Does reconnecting W_phase to routing help?** (step46 — just launched)
8. **Do interneurons compound with AntiHebb at D=64?** (step47 — Ref running, configs pending)
9. **Does safety valve redesign (variance/spectral reg) help?** (step45 — Ref running)

**🟢 Gen4 design questions (post-current-batch):**
10. **What does Gen4 compound stack look like with calibrated params + AntiHebb + surviving Gen1 mechanisms?** (step32, blocked on step22b+step29c)
11. **Can input-dependent dynamic topology (step36/step46 winners) replace O(N²) signed coupling?** (ARM 3 key question)
12. **Is the true D=64 ceiling (with calibrated params + AntiHebb) meaningfully higher than 70.14%?** (step29c Gen4 compound)
