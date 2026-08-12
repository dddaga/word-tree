# Phase 5 Part 15g — FLOPs Floor Mapping (2026-04-10)

Continues from: `LEARNINGS_phase5_p15f_warmstart_efficiency.md`

## Context

Phase exit confirmed at ≤6.18M FLOPs (step174 = 95.82%, step176-A = 96.18%).
This part tracks the FLOPs floor search: what is the *minimum* FLOPs to hit ≥95%?

Key reference points entering this part:
- step175 (N=1024 D=32 K_hh=4 Tier-2): 94.62% @ ~3.1M FLOPs — below 95%
- step177 (N=1024 D=48 K_hh=4 Tier-2, running ep110=93.96%): ~4.65M — likely below 95%
- step176-A (N=2048 D=32 K_hh=4 Tier-2): 96.18% @ ~6.1M FLOPs — phase exit ✓
- step174 (N=2048 D=16 K_hh=8 Tier-2): 95.82% @ ~6.2M FLOPs — phase exit ✓

Gap: N=1024 tops out ~94.6% even at full data 150ep. N=2048 is the right scale.
Question: at N=2048, what is the minimum D for ≥95%?

---

## step176 — N=2048 D=32 K_hh=4 scratch Tier-2 FINAL

| Config | α | top1_best | best_ep | top1_last | FLOPs | verdict |
|--------|---|-----------|---------|-----------|-------|---------|
| A (full data 150ep) | 1.0 | **96.18%** | 125 | 95.69% | ~6.1M | ✓ PHASE EXIT |
| B (full data 150ep) | 1.05 | running ep110=93.96% | — | — | ~6.1M | TBD |

**Config A FINAL: 96.18% best_ep=125** — strongest phase exit confirmation.
- step173-Ref (50%/75ep, same arch) = 94.93% → full data adds **+1.25pp**
- Confirms N=2048 scratch K=8 builds optimal routing topology (no warm-start mismatch)
- α=1.05 calibration (Config B): at ep110=93.96%, trending lower than A — warm Tier-1 precedent (step173-B<Ref by −0.23pp) may generalize

**Key finding (CONFIRMED):** At N=2048 D=32 scratch, α=1.0 (not 1.05) is optimal.
α=1.0 outperforms α=1.05 at this scale/data regime. (Previous α=1.05 win was at N=4096.)

---

## step177 — N=1024 D=48 K_hh=4 warm+W_proj Tier-2 FINAL

**Result: 95.13% best_ep=133, last=95.11%. ✓ PHASE EXIT @ ~4.72M FLOPs.**

New min-FLOPs efficiency record — 24% fewer FLOPs than previous record (step174/176-A at ~6.1-6.2M).

Reference chain: step172-B Tier-1 (50%/75ep) = 94.01% → full data 150ep adds **+1.12pp** → phase exit.
Prediction was ~94.0-94.5%. Actual was higher — warm+W_proj Tier-2 benefits from late-training convergence (best at ep133, not ep74).

**Key implication:** N=1024 D=48 warm+W_proj IS viable for phase exit at ~4.72M FLOPs.
Earlier ceiling (~94.0%) was a Tier-1 artifact, not a true capacity ceiling.

---

## step178 — N=2048 D=24 K_hh=4 scratch Tier-1 (launched 2026-04-10)

**Hypothesis:** N=2048 dominates D=1024 at same FLOPs. If N=2048 D=24 (~4.72M) 
can extrapolate from N=2048 D=32 (96.18%), the FLOPs floor for ≥95% may be ~4.7-5M.

Config: scratch α=1.0, 50%/75ep (Tier-1)
FLOPs: 3×2048×4×24×8 = 4,718,592 ≈ **4.72M** (24% below 6.18M budget)
Session: step178_mm (Mac Mini MPS)

Tier-1 precedent for N=2048 D=32 scratch: 94.93% at 50%/75ep (step173-Ref).
D=24 is ×0.75 of D=32 → expect ~1-2pp below → 93-94% at Tier-1.
If Tier-1 ≥93%: strong candidate for Tier-2 (likely ≥95% at full data 150ep).

---

## step179 — N=2048 D=28 K_hh=4 scratch Tier-1 (launched 2026-04-10)

**Hypothesis:** Midpoint between D=24 (4.72M) and D=32 (6.1M) at N=2048.
If D=24 fails 95% but D=28 passes → FLOPs floor is ~5.5M, not 4.7M.

Config: scratch α=1.0, 50%/75ep (Tier-1)
FLOPs: 3×2048×4×28×8 = 5,505,024 ≈ **5.51M** (11% below 6.18M budget)
Session: step179_mm (Mac Mini CPU)

Together steps 178+179 bracket-search the FLOPs floor:
- D=24 (4.72M): lower bound probe
- D=28 (5.51M): upper bound probe before D=32

---

## FLOPs Frontier Summary (2026-04-10)

| Config | N | D | K_hh | FLOPs | top1 | Method | Status |
|--------|---|---|------|-------|------|--------|--------|
| step175 | 1024 | 32 | 4 | ~3.1M | 94.62% | warm+W_proj | ✗ no exit |
| step178 | 2048 | 24 | 4 | ~4.72M | TBD | scratch | running |
| **step177** | **1024** | **48** | **4** | **~4.72M** | **95.13%** | warm+W_proj | **✓ exit NEW RECORD** |
| step179 | 2048 | 28 | 4 | ~5.51M | TBD | scratch | running |
| step174 | 2048 | 16 | 8 | ~6.2M | 95.82% | warm+W_proj | ✓ exit |
| step176-A | 2048 | 32 | 4 | ~6.1M | **96.18%** | scratch | ✓ exit BEST |

**Current floor: ~4.72M FLOPs** (step177). Steps 178/180 probe whether floor can be pushed below 4M FLOPs at N=2048.

## step180 — N=2048 D=20 K_hh=4 scratch Tier-1 FINAL

**Result: 94.14% best_ep=64, last=92.82% @ ~3.93M FLOPs. DONE.**

Tier-2 projection (+1.12-1.25pp based on prior N=2048 pattern): **~95.2-95.4%** → likely PHASE EXIT.
→ Launched step181 (N=2048 D=20 Tier-2, full data 150ep) immediately on Mac Studio CPU.

Also: step176-B hit **95.75% at ep120** (PHASE EXIT), then oscillated back to 94.60% at ep130. Running to ep150.

| Config | N | D | K_hh | FLOPs | top1 | Method | Status |
|--------|---|---|------|-------|------|--------|--------|
| step175 | 1024 | 32 | 4 | ~3.1M | 94.62% | warm+W_proj | ✗ no exit |
| step180 | 2048 | 20 | 4 | ~3.93M | 94.14% Tier-1 | scratch | done → step181 Tier-2 running |
| step178 | 2048 | 24 | 4 | ~4.72M | 94.57% Tier-1 | scratch | done → step184 Tier-2 running |
| **step177** | **1024** | **48** | **4** | **~4.72M** | **95.13%** | warm+W_proj | **✓ exit** |
| step179 | 2048 | 28 | 4 | ~5.51M | 94.24% Tier-1 | scratch | done → step183 Tier-2 running |
| step174 | 2048 | 16 | 8 | ~6.2M | 95.82% | warm+W_proj | ✓ exit |
| step176-A | 2048 | 32 | 4 | ~6.1M | **96.18%** | scratch | ✓ exit BEST |

---

## step181 — N=2048 D=20 K_hh=4 scratch Tier-2 FINAL

**Result: 96.03% best_ep=147, last=95.46% @ ~3.93M FLOPs. PHASE EXIT ✓. NEW MIN-FLOPs RECORD (beat step177 @ 4.72M).**

Tier-1 ref (step180): 94.14%@ep64 → Tier-2 gain: **+1.89pp**. Higher than typical +1.1-1.25pp range.
Training oscillated: peaked 94.80%@ep40 → dipped 93.37%@ep80 → recovered 95.11%@ep90 → final 96.03%@ep147.
High oscillation is characteristic of N=2048 training — always run to completion, never kill early.

---

## step182 — N=2048 D=16 K_hh=4 scratch Tier-1 FINAL

**Result: 93.96% best_ep=74, last=91.02% @ ~3.15M FLOPs. DONE → Tier-2 step185.**

Tier-2 projection (+1.1-1.25pp): ~95.1-95.2% → phase exit expected.
Actual Tier-2 (step185) gain was even higher (+1.91pp).

---

## step185 — N=2048 D=16 K_hh=4 scratch Tier-2 FINAL

**Result: 95.87% best_ep=140, last=95.49% @ ~3.15M FLOPs, params=67,744, elapsed=848s. PHASE EXIT ✓. NEW MIN-FLOPs RECORD.**

Beats step181 (96.03% @ 3.93M): floor dropped from 3.93M → 3.15M FLOPs.
Tier-1 ref: 93.96% → Tier-2 gain: **+1.91pp** (above typical range — N=2048 long-run benefit is larger than projected).

Key observation: accuracy only drops ~0.16pp from D=20→D=16 at Tier-2 (96.03% → 95.87%). Very flat curve in this region.

Mac Studio MPS freed → immediately launched step187 (D=8 Tier-1, ~1.57M).

---

## step186 — N=2048 D=12 K_hh=4 scratch Tier-1 (running)

Config: scratch α=1.0, 50%/75ep, FLOPs=2,359,296 (~2.36M)
Session: step186_sc (Mac Studio CPU), ep10=85.68%

Projection: if ~92%+ at ep75, Tier-2 may hit ≥95% at 2.36M FLOPs.

---

## step187 — N=2048 D=8 K_hh=4 scratch Tier-1 (running)

Config: scratch α=1.0, 50%/75ep, FLOPs=1,572,864 (~1.57M) — 75% below 6.18M budget
Session: step187_sc (Mac Studio MPS), just launched

Extreme probe: Fourier sphere S^7 is very low-dimensional.
Decision rule: if ≥92% → launch Tier-2 to check phase exit at 1.57M. If <90% → floor between D=8 and D=12.

---

## step186 — N=2048 D=12 K_hh=4 scratch Tier-1 FINAL

**Result: 93.10% best_ep=75 @ ~2.36M FLOPs. DONE → Tier-2 step188 (borderline).**

D=12 → D=16: drop of ~0.86pp at Tier-1. Following the accelerating Tier-2 lift trend (D=20→+1.89pp, D=16→+1.91pp), optimistic Tier-2 projection: 93.10% + 1.91pp = 95.01% — barely exits.
Launched step188 (D=12 Tier-2) immediately on Mac Studio MPS.

---

## step187 — N=2048 D=8 K_hh=4 scratch Tier-1 FINAL

**Result: 91.26% best_ep=72 @ ~1.57M FLOPs. D=8 RULED OUT for Tier-2.**

Even optimistic +1.91pp projection: 91.26% + 1.91pp = 93.17% — well short of 95%.
D=8 is the confirmed floor for N=2048 scratch at 50%/75ep.

---

## step188 — N=2048 D=12 K_hh=4 scratch Tier-2 FINAL

**Result: 94.62% best_ep=138, last=94.22% @ ~2.36M FLOPs. NO PHASE EXIT — 0.38pp short.**

D-reduction axis floor confirmed: D≥16 K_hh=4 @ 3.15M required for ≥95%.
Tier-1 to Tier-2 gain: 93.10% → 94.62% = +1.52pp (lower than D=16/D=20 pattern of +1.89-1.91pp).
The +1.9pp lift does NOT hold at D=12 — the lift itself diminishes with D.

→ Launched step191 (D=16 K_hh=3, same 2.36M FLOPs, Tier-1): tests if D=16 dimensionality with K_hh=3 beats D=12 K_hh=4 at same budget.

## step189/190/191 — New axis probes

- **step189** (D=10 Tier-1): 92.25%@ep73 @ ~1.97M. Tier-2 ruled out (max ~94.2% even with +2.0pp). D=10 is not viable.
- **step190** (D=16 K_hh=2 Tier-1): **93.86%@ep75 @ ~1.57M. CRITICAL RESULT.**
  - Beats D=8 K_hh=4 (91.26%) by +2.60pp at identical FLOPs. D=16 dimensionality >> K_hh.
  - Tier-2 projection: 94.96-95.76% → Tier-2 launched (step193).
- **step191** (D=16 K_hh=3 Tier-1): **94.68%@ep71 @ ~2.36M. OUTSTANDING.**
  - Already ABOVE step188 Tier-2 (94.62%) at same FLOPs. Tier-2 near-certain phase exit.
  - Tier-2 projection: 95.8-96.6% → Tier-2 launched (step192).

---

## CRITICAL FINDING: D=16 Dimensionality is the Binding Constraint

**Confirmed (2026-04-10):** At fixed FLOPs budget, higher D + lower K_hh always dominates lower D + higher K_hh.

Evidence:
| Config | FLOPs | Tier-1 | Verdict |
|--------|-------|--------|---------|
| D=8 K_hh=4 (step187) | 1.57M | 91.26% | — |
| D=16 K_hh=2 (step190) | 1.57M | **93.86%** | +2.60pp at same FLOPs |
| D=12 K_hh=4 (step186) | 2.36M | 93.10% | — |
| D=16 K_hh=3 (step191) | 2.36M | **94.68%** | +1.58pp at same FLOPs |

Implication: The Fourier hypersphere dimension (D) determines representational capacity. K_hh is a routing mechanism that can be reduced without destroying capacity. The architecture benefits from maintaining higher D over maintaining K_hh density.

## step192/193 — New Tier-2 Runs (running)

- **step192** (D=16 K_hh=3 Tier-2): **PHASE EXIT at ep50=95.06% @ ~2.36M FLOPs. NEW MIN-FLOPs RECORD.** Beats step185 (3.15M) by 25%. Still running, final likely ~96%+.
- **step193** (D=16 K_hh=2 Tier-2): ep20=93.04% @ ~1.57M — already near D=12 Tier-1 final (93.10%@ep75) with 130ep left. 1.57M phase exit is very possible.

---

## step189 — N=2048 D=10 K_hh=4 scratch Tier-1 (running)

Config: scratch α=1.0, 50%/75ep, FLOPs=1,966,080 (~1.97M)
Session: step189_sc (Mac Studio CPU)
Fills D=8→D=12 bracket. Decision: if ≥92% → launch Tier-2 at ~1.97M.

---

## FLOPs Frontier (Updated 2026-04-10)

| Config | N | D | K_hh | FLOPs | top1 | Tier | Status |
|--------|---|---|------|-------|------|------|--------|
| step187 | 2048 | 8 | 4 | ~1.57M | 91.26% | 1 | DONE — Tier-2 ruled out |
| step189 | 2048 | 10 | 4 | ~1.97M | TBD | 1 | running |
| step188 | 2048 | 12 | 4 | ~2.36M | TBD | 2 | running (borderline 95%) |
| step186 | 2048 | 12 | 4 | ~2.36M | 93.10% | 1 | DONE → step188 |
| **step185** | **2048** | **16** | **4** | **~3.15M** | **95.87%** | 2 | **✓ EXIT CURRENT FLOOR** |
| step181 | 2048 | 20 | 4 | ~3.93M | 96.03% | 2 | ✓ exit |
| step184 | 2048 | 24 | 4 | ~4.72M | TBD (ep30=93.86%) | 2 | running |
| step177 | 1024 | 48 | 4 | ~4.72M | 95.13% | 2 | ✓ exit |
| step183 | 2048 | 28 | 4 | ~5.51M | TBD (ep40=93.71%) | 2 | running |
| step176-A | 2048 | 32 | 4 | ~6.1M | 96.18% | 2 | ✓ exit BEST |
