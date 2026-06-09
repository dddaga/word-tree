# FFN Baseline Line — Queue

| Step | What | Tier | Slot | Status |
|---|---|---|---|---|
| ffn_step001 | Per-channel FFN, 2 budgets × 3 activation variants (6 configs), Imagenette | T0 | mini_mps | DONE 2026-06-10 |
| ffn_step001-T1 | Same 6 configs, 75ep | T1 | mini_cpu (scheduler) | DONE 2026-06-10 |

## LINE VERDICT (2026-06-10, T1 evidence)
Per-channel FFN plateaus at **93.0%** regardless of budget (1% = 5%, CONFIRMED
both tiers). SGNNET step605: 95.95% at 32× fewer params (35K vs 1.1M).
→ "small+sparse FFN" does NOT explain SGNNET win; architecture matters ≈+2.9pp.
RReLU-at-train hypothesis REJECTED (B worst at T0 and T1, −0.5pp vs plain ReLU).
norm→ReLU ≥50% sparsity CONFIRMED (≈0.50/layer all configs).
Line CONCLUDED. Optional T2 only if paper wants this baseline row.

### ffn_step001 T1 (75ep, 50% data, mini_cpu)
`results/ffn_baseline/ffn_step001_perchannel_t1_seed42__mini_cpu.json`
| Config | Best |
|---|---|
| b1_A | 0.9294 |
| b1_B | 0.9254 |
| b1_C | 0.9304 |
| b5_A | 0.9304 |
| b5_B | 0.9228 |
| b5_C | 0.9297 |

## Results

### ffn_step001 T0 (20ep, 50% data, seed 42, mini_mps)
`results/ffn_baseline/ffn_step001_perchannel_t0_seed42__mini_mps.json`

| Config | Best | Params (%FC) | Eff FLOPs |
|---|---|---|---|
| b1_A_norm_relu | 0.9279 | 1,112,074 (0.93%) | 1.79M |
| b1_B_norm_rrelu | 0.9246 | 1,112,074 (0.93%) | 1.80M |
| b1_C_norm_bias_rrelu | 0.9284 | 1,140,746 (0.95%) | 1.85M |
| b5_A_norm_relu | 0.9294 | 5,842,954 (4.89%) | 7.38M |
| b5_B_norm_rrelu | 0.9246 | 5,842,954 (4.89%) | 7.38M |
| b5_C_norm_bias_rrelu | 0.9292 | 5,937,162 (4.96%) | 7.99M |

Sparsity per layer ≈ [0.50, 0.50, 0.51, 0.52] — norm→ReLU ≥50% zeros CONFIRMED.

**Findings (T0, tag accordingly):**
- 1% budget ≈ 5% budget (+0.1pp) — budget SATURATED at 1%. CONFIRMED (clean ablation).
- RReLU-at-train (B) slightly WORSE than plain ReLU (A) at T0 (−0.3pp). HYPOTHESIS: noise hurts at 20ep; recheck at T1.
- Bias variant (C) ≈ A. Selectivity knob no effect at T0.
- Context: 92.8% T0 vs SGNNET full-pipeline refs 95.95% (T1 KD champion @35K params) / 97.30% (T2 ceiling). NOT directly comparable across tiers — T1 head-to-head queued.
