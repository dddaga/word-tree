# Parallel Line: Per-Channel Sparse FFN Baseline

**Status: ACTIVE. Isolated from main SGNNET line — own step numbers (ffn_stepNNN), own queue.**
Created 2026-06-10 per Dhiraj's directive.

## Hypothesis
Vision features highly compressible (spatial redundancy + classification task is easy).
Standard components (per-channel FFN + norm→ReLU sparsity) at tight param budgets may
get close to SGNNET ("fancy model"). This line quantifies how much of SGNNET's win is
architecture vs just "small + sparse works on vision features."

## Design
- VGG16 conv output (B, 25088) → view (B, 512, 49): **each channel gets its own MLP**
  (batched einsum, no cross-channel mixing until readout). Trickle-down depth-wise widths.
- Concat channel outputs (512×8=4096) → linear readout → 10 classes.
- Param budgets vs VGG16 FC block (119,586,826 params: 25088→4096→4096→10):
  - **1% = 1,195,868** — widths [49,28,20,8]/ch = 2092/ch → ~1.11M total
  - **5% = 5,979,341** — widths [49,84,60,32,8]/ch = 11,332/ch → ~5.84M total

## Activation variants (the sparsity mechanism)
| Variant | Train | Inference | Rationale |
|---|---|---|---|
| A_norm_relu | LN(no affine)→ReLU | same | zero-mean → ≥50% zeros guaranteed |
| B_norm_rrelu | LN→RReLU | LN→ReLU | gradient always flows in train; exact zeros at inference |
| C_norm_bias_rrelu | LN+learned bias→RReLU | →ReLU | bias = selectivity knob (% active neurons) |

Inference ReLU zeros → sparse multiplication leverage (effective-FLOPs metric).

## Comparison targets (main line, fixed)
- SGNNET step605 K=1 KD: 95.95% @ 34,976 params (0.029%), 0.20M FLOPs
- SGNNET D=16 ceiling: 97.30% T2 @ 0.98M FLOPs
- Both budgets are LARGER than SGNNET (1% ≈ 34× params) — if accuracy still trails,
  strengthens SGNNET claim; if matches, honest baseline for paper.

## Tier protocol — same as main line
T0 20ep/50% → T1 75ep/50% → T2 150ep/100%. Imagenette (data/store.h5) first.

## Metrics per config
accuracy + params + dense FLOPs + measured inference activation sparsity per layer
(→ effective FLOPs) + wall-time. Never accuracy alone.

## Queue
See [QUEUE.md](QUEUE.md).
