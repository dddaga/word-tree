# Qwen Red-team Critique — Synthesis (2026-04-22)

## Source: critique_full_20260422_1631.md
## Reviewer: I (main brain) scoring qwen's attacks

---

## BLOCKERS — Must fix before submission

### B1: FLOPs scope not explicit enough (Attack 1)
**Status:** Claims + paper use "FC-head FLOPs" but the phrase isn't in abstract.
**Action:** Paper fix — add "FC head only, VGG16 conv features frozen and pre-extracted" to abstract + experiments section. No new experiment.
**Priority:** High — first thing a reviewer will ask.

### B2: JL claim is overreaching (Attack 2)
**Status:** claims.md line 11 says "Johnson-Lindenstrauss theory: random projections preserve distance structure." K_in=25 from 25088 features is a sparse random projection, not a JL embedding per se (JL guarantees d ≥ O(log n/ε²) for n data points, not input-dim scaling).
**Action:** Soften to: "motivated by random feature methods (Rahimi & Recht 2007); the full N=2048-neuron ensemble forms a wide sparse random projection, and formal JL-style concentration guarantees for sparse projections (Achlioptas 2003) apply across the ensemble."
**Priority:** High — a theory reviewer will flag this immediately.

---

## MAJORS — Address or explicitly disclaim

### M3: Neural gas / SOM not cited (Attack 5)
**Status:** Anti-Hebbian routing is functionally related to neural gas (Kohonen & Martinetz 1991) and Hebbian competitive learning. We claim novelty without citing repulsion-based routing ancestors.
**Action:** Related work addition: cite Kohonen (1990) SOM, Martinetz & Schulten (1991) neural gas, Lowe (1995) on competitive Hebbian learning. Distinguish: SGNNET uses W_pos geometric repulsion on graph topology, not feature-space repulsion without message-passing.
**Priority:** Medium — no experiment needed.

### M4: Attention readout failure may be param-count artifact (Attack 6)
**Status:** Step118 attention failure (−60pp) is used to justify mean-pool as architecturally load-bearing. But attention adds ~10× params vs mean-pool — failure may be a regularization mismatch, not geometry.
**Action:** Soften claim in paper from "mean-pool is best by design" to "mean-pool outperforms attention in our experiments; param-matched ablation is future work." 
**New experiment option (step978):** Train attention readout with reduced hidden dim to match mean-pool param count (one T0, low cost).
**Priority:** Medium.

### M5: Compound interference post-hoc (Attack 3)
**Status:** The "routing ceiling" hypothesis at N=1024 was never tested with extended epochs.
**Action:** Tag as HYPOTHESIS in claims.md. Already mostly framed this way. Low urgency.

---

## MINORS — Note and move on

### m6: W_pos isolation incomplete (Attack 7)
Step970 covers layerwise isolation. If θ is neutral (step886 result) but W_pos does heavy lifting, that confirms routing geometry matters but the "learning in dynamics" framing needs W_pos isolation. Tag as future work.

### m7: Hard-label at T=2 (Attack 8)
Step977 currently running covers T=1 KD vs CE. T=2 hard-label baseline is a minor gap — if step977 shows NEUTRAL, the KD story is already simplified and T=2 becomes moot.

---

## Actionable summary (ordered)

| # | Action | Type | Priority |
|---|--------|------|----------|
| 1 | Add "FC-head-only FLOPs" to abstract + experiments | Paper fix | BLOCKER |
| 2 | Soften JL claim → sparse random projection framing | Paper fix | BLOCKER |
| 3 | Add neural gas / SOM to related work | Paper fix | MAJOR |
| 4 | Soften attention readout claim | Paper fix | MAJOR |
| 5 | step978: param-matched attention T0 | Experiment | MAJOR |
| 6 | Tag compound interference as HYPOTHESIS | claims.md fix | MINOR |
