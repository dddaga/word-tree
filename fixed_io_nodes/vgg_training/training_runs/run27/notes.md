# run27 — Notes

## Config delta from run17
- `total_nodes: 2048` (was 4146)
- `input_nodes: 1568` (was 3136 — forced by 1568 × 16 = 25088)
- `vector_dim: 16` (was 8)
- `cardinality: 2` (was 200)
- All else identical to run17

## Result
- **val_best: 29.04% @ep36** (~2.9x random chance)
- FLOPs/fwd: ~5.36M per sample (44.5x cheaper than run17)
- vs run17: **-57.81pp** — catastrophic failure
- vs run22 (C=4, D=8): **+3.21pp** — slightly better than C=4, likely from D=16

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 11.21%  |
| ep5   | 19.16%  |
| ep10  | 24.46%  |
| ep15  | 26.22%  |
| ep20  | 27.11%  |
| ep25  | 27.97%  |
| ep30  | 28.48%  |
| ep35  | 28.20%  |
| ep36  | 29.04%  |
| ep40  | 28.84%  |

## Analysis
- **CONFIRMED: Our softmax routing cannot replicate team member's 95.52% at C=2.** Expected result — our routing causes gradient starvation at low cardinality.
- **ep15 val=26.22%** — above the 15% kill threshold but clearly non-viable. Same pattern as run22 (C=4): slow crawl to ~28%, never breaks out.
- Train acc only 26.91% at ep40 — extreme underfitting. The GNN cannot learn at this sparsity with softmax routing.
- **D=16 gave +3.21pp over run22's C=4/D=8 (25.83%)** — but this is a confounded comparison (C, D, N, input_nodes all changed). The D=16 benefit is minimal in the starvation regime.
- **Team member comparison:** They get 95.52% at identical (N, D, C, I). The 66pp gap confirms our architectures are fundamentally different — their gather-sum routing (no softmax, no complex numbers) avoids starvation entirely.
- Plateau reached by ep20 (~27%). Barely moves for last 20 epochs.

## Verdict
DONE. CONFIRMED: Softmax routing at C=2 fails on our architecture (29.04% ≈ 2.9x random).
The team member's 95.52% at same config is unreachable with softmax+complex routing.
Next: run29 — test if removing softmax (uniform routing) fixes starvation at C=200 first, then C=2.
