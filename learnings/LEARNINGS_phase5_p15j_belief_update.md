# LEARNINGS Phase 5 — Part 15j: Belief Update Framework (2026-04-11)

## Evidence-Based Belief System

Every claim carries confidence level based on evidence quality:

| Level | Meaning | Evidence needed to CHANGE |
|-------|---------|--------------------------|
| **CONFIRMED** | Clean ablation, single variable | Contradicting ablation at same scale |
| **STRONG** | Multiple corroborating experiments | Single contradicting experiment + explanation |
| **HYPOTHESIS** | Post-hoc or indirect evidence | Any clean counter-evidence |
| **STALE** | Tested on old config/scale | Retest on current config |
| **KILLED** | Confirmed negative | Extraordinary evidence (new mechanism/scale) |

### Rule: Evidence > Theory. Data contradicts belief → update belief, don't explain away data.

---

## Belief Updates from Step 216-219 (2026-04-11)

### UNLEARNED (beliefs data now contradicts)

**1. "N=1024 winners might transfer to N=2048"** → **KILLED**
- Prior: HYPOTHESIS (untested at N=2048)
- Evidence: step216 — twopop_weight (-1.81pp), twopop_theta (-2.45pp), curriculum (-58pp)
- New: **CONFIRMED — capacity crutches do NOT transfer.** All N=1024 gains from step131/142/143 invalid at step199 scale.
- Action: Remove twopop_weight, twopop_theta, curriculum from "promising" lists. Capacity patches for N=1024 only.

**2. "α_ahebb=1.05 might help at N=2048"** → **NEUTRAL (not helpful)**
- Prior: STRONG (confirmed +0.79pp at N=4096)
- Evidence: step216-A — -0.33pp at N=2048 D=16 K_hh=2
- New: α=1.05 vs 1.0 noise-level at N=2048 K_hh=2. N=4096 gain may be scale-specific.
- Action: Keep α=1.0 default. Not worth pursuing.

**3. "Topology design matters significantly"** → **MOSTLY WRONG**
- Prior: HYPOTHESIS (untested — assumed dead-ends hurt)
- Evidence: step219 — anti-pref +0.10pp, coverage +0.15pp, uniform random -0.84pp
- New: **Topology barely matters at K_hh=2.** Eliminating all dead-ends gains only +0.15pp. Small-world near-optimal. Uniform random (no group structure) slightly worse.
- Action: Topology redesign LOW priority. Group structure helps slightly. Dead-end elimination not worth complexity.

### LEARNED (new beliefs from data)

**4. "Full polarizer routing gives +1.27pp"** → **STRONG (needs Tier-1)**
- Evidence: step217-A — 92.89% vs 91.62% Ref (+1.27pp, clean ablation)
- Mechanism: project Z[neighbor] onto W_pos[receiver] before summing. Makes routing input-dependent.
- NOT capacity patch — changes routing DYNAMICS. Litmus: survives at N=4096?
- Action: ADVANCE to Tier-1. Also test at N=4096.

**5. "Stronger polarization = better"** → **STRONG**
- Evidence: Full (α=1.0) > Partial (α=0.5) > Soft (α=0.3) > None
- Relationship monotonic: more polarizer = better, at least up to α=1.0
- Action: Consider α > 1.0 (over-project?) in Tier-1

**6. "AH is single most important component"** → **CONFIRMED (step218)**
- Evidence: Without AH, accuracy drops 91.5% → 18.8% — 73pp collapse
- Dwarfs all other effects (topology ±0.8pp, polarizer +1.27pp)
- AH not regularizer in usual sense — PREREQUISITE for learning
- Action: AH is load-bearing wall. Never remove. All future experiments must include AH.

**7. "Rotation routing barely helps"** → **HYPOTHESIS**
- Evidence: step217-D — +0.10pp (noise-level)
- In-plane rotation with learned temperature didn't meaningfully improve over standard routing
- Possible: temperature initialized too low (0.1), or 20 epochs insufficient
- Action: Park. Polarizer (projection) simpler, stronger mechanism.

**8. "Heterogeneous K_hh helps"** → **KILLED**
- Evidence: step220 — ALL 4 variants worse than uniform K_hh=2 (best -0.28pp, worst -2.40pp)
- More variance in connectivity = worse. Extreme hubs (-2.40pp) worst.
- Action: Uniform K_hh optimal. Do not pursue heterogeneous connectivity.

**9. "Output-assigned topology eliminates dead-ends and helps"** → **KILLED**
- Evidence: step221 — ALL 3 variants worse than input-assigned (best -1.30pp, worst -2.45pp)
- Zero dead-ends by construction, still worse. Guaranteed input aggregation (in-degree=K_hh) matters more than zero dead-ends.
- Action: Input-assigned topology correct. Dead-end elimination not worth tradeoff.

**10. "Static topology design is a lever"** → **KILLED (3 experiments)**
- Evidence: step219 (anti-pref +0.15pp), step220 (hetero all negative), step221 (output-assigned all negative)
- Three independent experiments show: topology DESIGN barely matters. Uniform small-world near-optimal.
- Opportunity is LEARNING topology dynamically (RigL, Gumbel-Softmax), not redesigning statically.

### PRESERVED (beliefs that survived testing)

**11. "F.normalize is load-bearing"** → **CONFIRMED** (unchanged, prior evidence)
**12. "D > K_hh at fixed FLOPs"** → **CONFIRMED** (unchanged, step214/215)
**13. "D=16 ceiling at 97.17%"** → **CONFIRMED** (unchanged, step205/209)

---

## Research Direction: Dynamic Routing (User Priority)

**Status: PRIMARY RESEARCH GOAL** — Dhiraj committed to finding parameter-efficient input-dependent routing despite repeated failures.

**What failed and why:**
- Multiplicative gates → gate-death theorem (g^K_iter → 0)
- N=1024 capacity crutches (twopop, curriculum) → don't transfer to N=2048
- Static topology redesign (step219/220/221) → topology design barely matters

**What worked:**
- **Polarizer (+1.27pp)** — projection-based, not multiplicative. First successful input-dependent routing.
- Survives because projection doesn't attenuate signal magnitude like gates do.

**Next: learn from failure patterns.**
- Analyze loss curves from catastrophic failures (curriculum -58pp) to understand failure modes
- Design routing mechanisms avoiding gate-death: projection, rotation, additive modulation
- Learn topology itself (RigL, Gumbel-Softmax) rather than designing statically

---

## Meta: Learning vs Unlearning Protocol

### When to UPDATE belief
- New experiment at same or higher scale directly contradicts it
- Delta > 2× noise floor (~0.5pp for Tier-0 scouts)
- Experiment clean (single variable changed)

### When to PRESERVE belief despite apparently contradicting data
- Contradicting experiment changed multiple variables (confounded)
- Scale different (N=1024 result doesn't invalidate N=4096 finding)
- Delta within noise floor

### When to KILL research direction
- 3+ experiments show consistent negative results across scales
- Theoretical explanation for WHY it fails confirmed
- No remaining untested variant could plausibly reverse finding

### Danger of Confirmation Bias
- Don't explain away surprising negative results ("implementation must be wrong")
- Don't over-weight positive results from single experiment
- Mechanism shows +1pp at Tier-0 → PROMISING not CONFIRMED
- Run Tier-1 before changing defaults or writing conclusions