# Best Practices — AI Collaboration Patterns

Seven patterns distilled from the SGNNET work. Each is in the project's living `CLAUDE.md` or its `MEMORY.md` index. Each comes with a concrete project example.

## 1. Tier protocol as a rejection filter, not a top-N selector

T0 = 20 epochs / 50% data. T1 = 75 epochs / 50% data. T2 = 150 epochs / 100% data. Every config that is **not clearly failing** advances from T0 to T1, not just the top performers. T0 over 46 early experiments predicts the T2 winner with Spearman ρ=0.80; 30 epochs raises it to 0.91. **Example:** when a "winner-take-top-3" filter was tried, it skipped a config that ranked 5th at T0 but ranked 1st at T2 in a later spot-check. Reverted to rejection-filter semantics permanently.

## 2. Evidence tagging — CONFIRMED / HYPOTHESIS / STALE

Every causal claim in the project gets one of three tags. CONFIRMED requires a clean ablation with one variable changed and a control present. Post-hoc failure explanations default to HYPOTHESIS. Base-architecture changes mark old conclusions STALE pending retest. "KILLED" requires CONFIRMED evidence; otherwise it's "KILLED (unvalidated)". **Example:** the gate-death theorem (compound multiplicative gates g^K → 0 at K_iter≥4) was held as HYPOTHESIS through eight failed configs, only promoted to CONFIRMED after step73 demonstrated a clean redistribution-based escape with +1.78pp delta. Retroactively cleaned up nine "KILLED" claims that were actually unvalidated.

## 3. Compounding rule — name the signal path before stacking mechanisms

Two mechanisms on the same signal path tend to cancel (gate-death, co-adaptation). Compounding is safe only when paths are **orthogonal** — different W_* matrices, or topology + signal modification. Before stacking, the path of each mechanism is named explicitly. **Example:** ΔW-projection + L2-norm were stacked safely because they touch different matrices. AH (anti-Hebbian) + multiplicative gate were stacked and produced -3pp; the post-hoc analysis showed both ran on the same suppression path, doubling sparsity catastrophically. Now any compounding proposal includes an isolation ablation first.

## 4. Gap-close attitude — never scope-narrow to avoid the gap

When SGNNET trailed a baseline on a dimension, the question was always *"what closes the gap so SGNNET wins on efficiency?"* — never *"drop that dimension from scope."* **Example:** SGNNET was 4× slower than VGG FC on MPS at batch 128 despite 116× fewer FLOPs. Root cause: gather-scatter is memory-bandwidth bound; cuBLAS dense matmul is not. Response: write `model_resonant_cuda.py` with `torch.compile(mode="reduce-overhead")` to fuse the K_iter loop, then benchmark on RTX 5060 Ti. Result: 5.26× faster than VGG FC at B=32. Gap closed by attacking it, not by dropping wall-time from the Pareto table.

## 5. Multi-machine autonomy with hard gates

Six training slots across three machines (Mac Mini MPS/CPU, Mac Studio MPS/CPU, RTX 5060 Ti CUDA/CPU). Every launch goes through `scripts/launch_slot.sh <slot> <script> [args]`. The wrapper enforces lock files, cleans stale tmux sessions, and prevents collisions between teammates working in the same directory. Hard RAM gates before launch: Studio ≥50 GB, Mini ≥20 GB. No bare `nohup`, no `&`. **Example:** the wrapper caught seven race conditions during a multi-teammate sprint where two collaborators tried to launch on the same slot within seconds. Lock files made the conflict explicit; nobody lost a run.

## 6. Triple-write knowledge persistence

Every result lands in three places: Graphiti temporal knowledge graph (`mcp__graphiti__add_memory`, group_id="dhiraj") for cross-session semantic recall, dated `learnings/LEARNINGS_*.md` for the append-only audit trail, and `learnings/concepts/*.md` for the always-current per-concept wiki. **Example:** when revisiting "why did we kill the polarizer mechanism at α=2.0?" three months after the experiment, the concept page `learnings/concepts/polarizer.md` had the current rationale, the dated LEARNINGS file had the original ablation, and Graphiti returned the related episodes via semantic search — three independent recovery paths.

## 7. Action-driven meditation every 25 experiments

The project includes a formal meditation skill that runs after every 25 concluded experiments. It re-anchors on the goal, scans the last batch of experiments for drift and stuck-ness, generates reframes, validates each reframe with a ≤5-minute proof-of-concept in the same conversation, and closes with ≥3 committed experiment scripts plus queue entries. **No pure speculation.** **Example:** meditation 003 (2026-04-23) detected severe drift — 28 experiments since meditation 002 had produced zero new viable mechanisms, while the manuscript had a confirmed ΔW-proj section that was entirely absent. Verdict: stop mechanism search, switch to writing sprint, run only 2-3 background experiments. Three writing-focused scripts committed; experiment queue rewritten with P0 = paper. Drift fixed.

---

These patterns work because they make the **operating contract** between human and AI inspectable. The contract is text in `CLAUDE.md`, evolves through observed friction across sessions, and every session-start ritual loads it back into context. The result is a research collaboration that compounds across hundreds of experiments without losing its grip on the milestone.
