# SGNNET Critical Findings

**Source:** Extracted from EXPERIMENT_QUEUE.md. All confirmed findings and dead ends.

---

## ⚠️ Critical Findings Table

| Finding | Detail |
|---------|--------|
| **Signed coupling DEAD at D=64** | 5 experiments confirm ≤32% at D=64 any α/K_iter. cos-sim on S^63 ≈ noise. **Do not test again.** |
| Signed coupling + K_iter≥8 collapses | Power iteration → dominant eigenvector. K_iter=3 stability limit — but K_iter=3 D=64 also fails vs ref |
| **alpha_reflect=0.5 wins at D=64** | step22b calibration confirmed: alpha_reflect=0.5 → 52.94% at 40ep vs 0.3 default |
| **D=64 is encoding ceiling** | D=128 non-viable: all LR/schedule combos stuck ~10%. Fourier encoding on S^127 collapses. |
| **AntiHebb α=0.5 at D=64 = 70.14%** | step29 Config A FINAL: **+13.86pp** over D=64 ceiling. Superseded — see below. |
| **⭐ AntiHebb α=1.0 calibrated base = 80.08%** | step29c Config A Phase 2: **NEW ALL-TIME BEST** at time. +21.40pp vs calibrated Ref. Adding phase_exc or interneurons KILLS gain (→67%). AntiHebb α=1.0 ALONE is Gen4. Gen4 compound ceiling (all mechanisms) = 67.72% (config G). |
| **step57 REF_BASELINE = 73.53%** | AntiHebb α=1.0, 50% data, 75ep. Comparison point for steps 58-61. Full-scale = 80.08%. |
| **⭐ step56 N=4096 = 84.36%** | Full data 150ep, Gen4 params, best_ep=146/150, params=529K. N-scaling: 512→69.58%, 1024→80.92%, 2048→81.10%, 4096→84.36% (PEAK), 10000→82.37% (REGRESSES). NOT monotonic above N=4096. N=10000 worse by 2pp. ⚠️ BUGGY ARCH — all numbers invalid, re-established by step71/step80. |
| **Wave-1 VERDICT: ALL mechanisms KILLED** | Steps 58-63 complete. Gate death on every new mechanism. Static AH routing remains best. AGR (step63): best B/D/E=55%, A=31%, C=39% — all -18pp+ vs 73.48% Ref. |
| **step65 KILLED: distance-phase routing** | exp(-γ·d_norm) on conn_hh, γ∈{0.5,1.0,2.0}×{raw,softmax}. All A-F ~48-49% vs REF=73.53% (−25pp). Distance geometry adds nothing AH doesn't provide. |
| **step60 KILLED: phase-distance routing (all 13 configs)** | Ref=73.55%, all 12 others gate-dead 10-19%. Comprehensive closure on phase-routing modifications. |
| **step54 KILLED: LR schedule sweep** | Plateau=70.78% (Ref), CosineWarmRestart T_0={10,25,50}=66-68%. Cosine warm-restarts hurt. Plateau optimal. |
| **⭐ K_iter=16 + AH=1.0 = 74.14% (+0.61pp vs K_iter=8)** | step68 COMPLETE: K_iter=16 Gen4 optimal at N=1024. Non-monotone: 8(73.63%)>10(73.58%)>12(72.94%)<16(74.14%)>24(70.06%). K_iter=24 cliffs −3.57pp. |
| **⭐⭐ step69 PATCH = +9.83pp — NEW REF_BASELINE_v2 = 83.36%** | step69 Ref (patched arch: input_coverage + alpha_reflect fix) = **83.36%** at 50%/75ep vs old REF_BASELINE 73.53%. All wave-1 mechanism comparisons (steps 58-68) against BUGGY baseline. Config A (turing=0.3) = 85.04% (+1.68pp vs Ref) at N=1024. |
| **⭐⭐⭐ step70 PROJECT BEST = 97.38%** | step70 Config B (turing=0.0, reflect=0.5, AH=1.0, N=4096, patched arch, 150ep): **97.38%** at best_ep=142. Config Ref (turing=0.3) = 97.20%. **turing=0.0 beats turing=0.3 by +0.18pp at N=4096.** Turing slightly harmful at full scale. Contrast N=1024 where turing=0.3 gave +1.68pp — contribution N-dependent. New Gen4+ optimal at N=4096: turing=0.0, reflect=0.5, AH=1.0. |
| **⭐ step86 K_hh=4 is NEW DEFAULT** | Mac Studio (Ref/A/B/C/D) done. Mac Mini (F/G/H/I) pending resume. **Key: K_hh=4 beats K_hh=6 Ref by +0.56pp AND saves −18% FLOPs (38.8M vs 47.2M).** AH suppression nullifies local K_local edges → all K_hh reductions free lunch. K_iter=6 costs −2.2pp (D=93.83% vs Ref=96.03%). K_hh lever >> K_iter lever at matched FLOPs. n_groups=512 (N//8) critical — n_groups=128 (N//32) causes −4.28pp regression. |
| **⭐ step88 COMPLETE — α=1.0 confirmed at N=4096** | Alpha sweep (40ep): α=0.5→87.69%, α=1.0→95.52% WINNER, α=1.5→89.48%, α=2.0→94.17%. α=1.0 confirmed optimal at N=4096 patched arch. |
| **FP16 AMP confirmed default** | `use_amp=True` already enabled in Trainer — all experiments run FP16 AMP. |
| **FLOPs goal extended** | Target: ≤1% params AND ≤1% FLOPs vs VGG16 FC (119.6M params, 119.6M FLOPs). Params: 529K/119.6M=0.44% ✓ achieved. FLOPs: 38.8M/119.6M=32.4% — 32× reduction needed. Path: N=512/256 + D=32 + K_hh=2 + K_in reduction. |
| **⭐ step71 K_iter=12 WINS at N=4096: 96.66%** | 5 configs complete (50%/75ep). Non-monotone: K_iter=4(92.82%) < 6(95.11%) < 8(95.87%) < 16(96.31%) < **12(96.66%)**. K_iter=12 +0.79pp over Ref. K_iter optimal N-dependent. Adopt K_iter=12 for future N=4096 experiments. |
| **step79 aux losses: sparsity/diversity borderline, phase coherence KILLED** | 6 configs (50%/75ep). Ref=96.10%. Phase coherence λ∈{0.01,0.001,0.0001}: all below Ref (−0.25 to −0.56pp) — KILLED. Sparsity reward D=96.31% (+0.21pp), routing diversity F=96.31% (+0.21pp). Marginal — not adopted. |
| **step80 N-scaling patched arch (partial)** | N512=72.79%, N2048=92.74% (both 50%/75ep). N=4096 from step71 Ref = 95.87%. Scaling law monotone so far: 512<2048<4096. Full curve pending. |
| **⭐ step82 group topology: A(n_groups=8)=85.63% (+3.01pp vs Ref=82.62%)** | Static topology wins — no new params. n_groups=8 > 16 > 32. D(input-align) KILLED (69.40%). Never tested at N=4096. |
| **step83 group routing KILLED** | Ref=84.61%, A(β=0.5 every-step)=78.60% (−6.01pp), B(β=0.5 final-only)=81.96% (−2.65pp). Root cause: S_g=mean(Z) too coarse, softmax collapses early; co-adaptation with W_pos. See `LEARNINGS_design_2026_04_08.md` for post-mortem. |
| **step48 SLOT-KILLED: K_iter at D=64** | Only Ref(58.24%) and A(K_iter=12,55.08%) ran — B-G never launched. K_iter>8 WITHOUT AH hurts. K_iter>8 WITH AH=1.0 untested → step68 rewrites at Gen4. |
| **Signed coupling permanently dead** | step49 closed: best 42.14% vs 46.85% Ref across all K_iter. Architecturally incompatible with D=64. Do not re-propose. |
| **step55 grouped input: Config A = 89.43%** | 7-row separate projection (overlap=0%) = 89.43% at 3.8M params. Config D (positions_separate, 552K params) = 84.10% — best param efficiency. |
| **step32 Gen4 compound: AH alone wins** | Ref=65.04%, A(AH alone)=73.76%. Every compound reduces accuracy. Confirms step29c. Buggy arch — relative verdict holds. |
| **step53 low-rank mixing: KILLED** | Ref=70.17%, A=69.15%, B=67.95%, C=71.06%, D=70.62%. None beat REF 73.53%. Cross-dim mixing adds nothing. CLOSED. |
| **step50 spatial dynamic W_pos: neutral/negative** | All dynamic K-NN configs ≤55.64% vs static 57.48%. Dynamic W_pos connectivity no improvement over static small-world. |
| **step46 W_phase reconnect: BORDERLINE/CLOSED** | Best C=59.62% (+3.19pp vs Ref 56.43%) — below 60% Gen4 re-test threshold. W_phase adds marginal value. CLOSED. |
| K_iter=3→8 gap = ~15pp at D=64 | Confirmed ×3. All 8 routing iterations necessary; MoD early exit kills performance. |
| MoD adaptive depth fails | step34 KILLED: 19-20% through e110. All K_iter steps contribute; early exit destroys iterative refinement. |
| Oja's rule routing update fails | step41 KILLED: 24% at e60 vs 56.79% Ref. PCA compression destroys directional diversity. |
| Dynamic Z-KNN underperforms static | step31 ~44% vs 56.28% static ceiling. Unstable K-NN on S^63 per step. |

---

## Dead Ends (Do Not Re-Propose)

| Mechanism | Why Dead | Final Step |
|-----------|----------|------------|
| Signed coupling at D=64 | cos-sim on S^63 = noise; 5 experiments confirm | step49 |
| D=128 encoding | All LR/schedule combos stuck ~10% | step33 |
| MoD adaptive depth | Early exit destroys iterative refinement | step34 |
| Oja's rule routing | PCA compression destroys directional diversity | step41 |
| Dynamic Z-KNN per step | Unstable on S^63 per step | step31 |
| Phase-excitatory (static W_phase) | -13pp vs AH alone at calibrated base | step29c |
| Hub interneurons | Fan-in too sparse at K_hh=6 | step61 |
| All wave-1 multiplicative gates | Gate-death theorem applies | steps 58-66 |
| Phase-target routing | -15 to -43pp vs Ref; static AH optimal | step66 |
| Distance-phase routing | All configs -25pp; distance adds nothing AH lacks | step65 |
| Low-rank cross-dim mixing | Best config +0.89pp — too marginal | step53 |
| Cosine warm restart LR | -4 to -8pp vs plateau schedule | step54 |

---

## Completed Results Reference

Full details: `learnings/LEARNINGS_phase5_p6_gen2_gen3.md`

| Step | What | Best result | vs ref |
|------|------|-------------|--------|
| step9 | D=16 baseline | 29.22% | — |
| step12 | N scale | N=1024→34.32% | +5.10pp |
| step13 | K_iter depth | K_iter=8→36.69% | +7.47pp |
| step16 | Inhibition | AntiHebb α=0.5→37.22% | +8.00pp ⭐ |
| step17 | Fast W_phase | Attention→31.06% | +1.84pp |
| step18 | Signed coupling | α=0.3→40.15% | +10.93pp ⭐ |
| step22 | D extension | D=64 N=1024→56.28% | +19.54pp ⭐⭐ |
| step29c | AntiHebb alone | **80.08%** | +21.40pp ⭐⭐⭐ |

---

## ARM Design Archive

Full ARM 1-4 experiment designs (step22b, step29c, step32, step36-53 etc.) — see complete EXPERIMENT_QUEUE.md content archived below line 160 in original (or refer to `LEARNINGS_phase5_p8_arm1_arm2.md` and `LEARNINGS_phase5_p9_arm3_arm5.md`).