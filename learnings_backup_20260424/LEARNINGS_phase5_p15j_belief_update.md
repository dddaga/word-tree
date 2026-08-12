# LEARNINGS Phase 5 — Part 15j: Belief Update Framework (2026-04-11)

## Evidence-Based Belief System

Every claim carries a confidence level based on evidence quality:

| Level | Meaning | Evidence needed to CHANGE |
|-------|---------|--------------------------|
| **CONFIRMED** | Clean ablation, single variable | Contradicting ablation at same scale |
| **STRONG** | Multiple corroborating experiments | Single contradicting experiment + explanation |
| **HYPOTHESIS** | Post-hoc or indirect evidence | Any clean counter-evidence |
| **STALE** | Tested on old config/scale | Retest on current config |
| **KILLED** | Confirmed negative | Extraordinary evidence (new mechanism/scale) |

### Rule: Evidence > Theory. If data contradicts a belief, update the belief, don't explain away the data.

---

## Belief Updates from Step 216-219 (2026-04-11)

### UNLEARNED (beliefs we held that data now contradicts)

**1. "N=1024 winners might transfer to N=2048"** → **KILLED**
- Prior belief: HYPOTHESIS (untested at N=2048)
- Evidence: step216 — twopop_weight (-1.81pp), twopop_theta (-2.45pp), curriculum (-58pp)
- New belief: **CONFIRMED — capacity crutches do NOT transfer.** All N=1024 gains from step131/142/143 are invalid at step199 scale.
- Action: Remove twopop_weight, twopop_theta, curriculum from "promising" lists. They are capacity patches for N=1024 only.

**2. "α_ahebb=1.05 might help at N=2048"** → **NEUTRAL (not helpful)**
- Prior belief: STRONG (confirmed +0.79pp at N=4096)
- Evidence: step216-A — -0.33pp at N=2048 D=16 K_hh=2
- New belief: α=1.05 vs 1.0 is noise-level at N=2048 K_hh=2. The gain at N=4096 may be scale-specific.
- Action: Keep α=1.0 as default. Not worth pursuing further.

**3. "Topology design matters significantly"** → **MOSTLY WRONG**
- Prior belief: HYPOTHESIS (untested — we assumed dead-ends hurt)
- Evidence: step219 — anti-pref +0.10pp, coverage +0.15pp, uniform random -0.84pp
- New belief: **Topology barely matters at K_hh=2.** Eliminating all dead-ends gains only +0.15pp. Small-world is near-optimal. Uniform random (no group structure) is slightly worse.
- Action: Topology redesign is LOW priority. Group structure helps slightly. Dead-end elimination is not worth the complexity.

### LEARNED (new beliefs from data)

**4. "Full polarizer routing gives +1.27pp"** → **STRONG (needs Tier-1)**
- Evidence: step217-A — 92.89% vs 91.62% Ref (+1.27pp, clean ablation)
- Mechanism: project Z[neighbor] onto W_pos[receiver] before summing. Makes routing input-dependent.
- This is NOT a capacity patch — it changes routing DYNAMICS. Litmus test: does it survive at N=4096?
- Action: ADVANCE to Tier-1 immediately. Also test at N=4096.

**5. "Stronger polarization = better"** → **STRONG**
- Evidence: Full (α=1.0) > Partial (α=0.5) > Soft (α=0.3) > None
- The relationship is monotonic: more polarizer = better, at least up to α=1.0
- Action: Consider α > 1.0 (over-project?) in Tier-1

**6. "AH is the single most important component"** → **CONFIRMED (step218)**
- Evidence: Without AH, accuracy drops from 91.5% to 18.8% — a 73pp collapse
- This dwarfs all other effects (topology ±0.8pp, polarizer +1.27pp)
- AH is not a regularizer in the usual sense — it's a PREREQUISITE for learning
- Action: AH is a load-bearing wall. Never remove it. All future experiments must include AH.

**7. "Rotation routing barely helps"** → **HYPOTHESIS**
- Evidence: step217-D — +0.10pp (noise-level)
- The in-plane rotation with learned temperature didn't meaningfully improve over standard routing
- Possible explanation: temperature initialized too low (0.1), or 20 epochs insufficient
- Action: Park this. Polarizer (projection) is the simpler, stronger mechanism.

**8. "Heterogeneous K_hh helps"** → **KILLED**
- Evidence: step220 — ALL 4 variants worse than uniform K_hh=2 (best -0.28pp, worst -2.40pp)
- More variance in connectivity = worse. Extreme hubs (-2.40pp) worst of all.
- Action: Uniform K_hh is optimal. Do not pursue heterogeneous connectivity.

**9. "Output-assigned topology eliminates dead-ends and helps"** → **KILLED**
- Evidence: step221 — ALL 3 variants worse than input-assigned (best -1.30pp, worst -2.45pp)
- Zero dead-ends by construction, but still worse. Guaranteed input aggregation (in-degree=K_hh) matters more than zero dead-ends.
- Action: Input-assigned topology is correct. Dead-end elimination is not worth the tradeoff.

**10. "Static topology design is a lever"** → **KILLED (3 experiments)**
- Evidence: step219 (anti-pref +0.15pp), step220 (hetero all negative), step221 (output-assigned all negative)
- Three independent experiments all show: topology DESIGN barely matters. Uniform small-world is near-optimal.
- The opportunity is LEARNING topology dynamically (RigL, Gumbel-Softmax), not redesigning it statically.

### PRESERVED (beliefs that survived testing)

**11. "F.normalize is load-bearing"** → **CONFIRMED** (unchanged, prior evidence)
**12. "D > K_hh at fixed FLOPs"** → **CONFIRMED** (unchanged, step214/215)
**13. "D=16 ceiling at 97.17%"** → **CONFIRMED** (unchanged, step205/209)

---

## Research Direction: Dynamic Routing (User Priority)

**Status: PRIMARY RESEARCH GOAL** — Dhiraj is committed to finding parameter-efficient input-dependent routing despite repeated failures.

**What's failed and why:**
- Multiplicative gates → gate-death theorem (g^K_iter → 0)
- N=1024 capacity crutches (twopop, curriculum) → don't transfer to N=2048
- Static topology redesign (step219/220/221) → topology design barely matters

**What's worked:**
- **Polarizer (+1.27pp)** — projection-based, not multiplicative. First successful input-dependent routing.
- Survives because projection doesn't attenuate signal magnitude like gates do.

**Next approach: learn from failure patterns.**
- Analyze loss curves from catastrophic failures (curriculum -58pp) to understand failure modes
- Design routing mechanisms that avoid gate-death: projection, rotation, additive modulation
- Learn topology itself (RigL, Gumbel-Softmax) rather than designing it statically

---

## Meta: Learning vs Unlearning Protocol

### When to UPDATE a belief
- New experiment at same or higher scale directly contradicts it
- The delta is > 2× the noise floor (~0.5pp for Tier-0 scouts)
- The experiment is clean (single variable changed)

### When to PRESERVE a belief despite apparently contradicting data
- The contradicting experiment changed multiple variables (confounded)
- The scale is different (N=1024 result doesn't invalidate N=4096 finding)
- The delta is within noise floor

### When to KILL a research direction
- 3+ experiments show consistent negative results across scales
- The theoretical explanation for WHY it fails is confirmed
- No remaining untested variant could plausibly reverse the finding

### The Danger of Confirmation Bias
- Don't explain away surprising negative results ("the implementation must be wrong")
- Don't over-weight positive results from a single experiment
- When a mechanism shows +1pp at Tier-0, it's PROMISING not CONFIRMED
- Run the Tier-1 before changing defaults or writing conclusions
