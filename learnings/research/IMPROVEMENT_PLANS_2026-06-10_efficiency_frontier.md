# Research Report: Efficiency Frontier — Improvement Plan
*Sub-agent fan-out 2026-06-10. Condensed from agent report.*

## Critical metric artifact (HIGH PRIORITY)
VGG_FC bench peak_mem (1.05 MB) measures **activation delta only** — ignores
456 MB resident weights. SGNNET already wins TOTAL memory footprint ~12×.
Current "loses on memory" claim in CLAUDE.md likely WRONG once audited.

## Ranked actions
1. **Memory-metric audit** + FP16/fused single-buffer inference for step605
   student. Target: 38.8 MB → ~5 MB.
2. **Energy benchmark harness** — Zeus/NVML on 5060ti, powermetrics on Mac.
   Metric: µJ/inference. Directly serves long-term goal (energy efficiency).
3. **PTQ eval grid** on trained step605 weights: W8A8 → W4A4. Post-training
   only — untouched by step968 QAT kill (that was training-time FP4).
4. **KD into ternary student** on full step605 recipe (soft-KD ΔW teacher) —
   removes step968 confounds (step968 lacked KD).
5. **Codebook/LUT routing** on K=1 student — step957 participation-ratio≈1.83
   means activations live near 2D manifold; small codebook plausible.

## Why this matters
Items 1–2 are paper-strengthening measurements (no training). Items 3–5 are
new efficiency wins on top of champion. All orthogonal to accuracy work.
