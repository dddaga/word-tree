# Research Report: Paper 2 (Dynamic Routing) — Improvement Plan
*Sub-agent fan-out 2026-06-10. Condensed from agent report.*

## State of pillar
Dynamic-topology pillar DEAD: 10 confirmed failure modes (gate-death theorem,
steps 873–916, 985–993). All multiplicative gating variants killed; additive
dynamic killed at T1 (step993: sign reversal from T0).

## Survivors (assets for Paper 2)
- **ΔW-proj** — halves seed variance (step760), CONFIRMED
- **Hypercomplex W_pos** — step960 quaternion design written, NEVER RUN
- **step989 FFN distillation** — GPT-2 extraction blocked on teammate GPU
- **Redistribution principle** — routing redistributes capacity, doesn't add it

## Three thesis options
| Option | Thesis | Dependency |
|---|---|---|
| A | Hypercomplex/algebraic routing (quaternion/sedenion W_pos; D=16=sedenion dim) | None — run step960 T0 now |
| B | Universal FFN replacement | TOTAL dependency on step989 |
| C | Anatomy of routing failure — taxonomy of 10 failure modes + expert-choice routing T0 (escapes FM7 2-way symmetry) | None |

## Recommendation
Run step960 quaternion T0 + expert-choice routing T0 now on Mac slots.
Hold paper spine decision until step989 resolves.
