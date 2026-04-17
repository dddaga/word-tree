---
name: compact
description: Project-aware compaction for the neuro_graph SGNNET research session. Gathers live experiment state, key findings, and decisions made this session, then compacts with that context preserved. Invoke when context is getting full or before ending a session. Replaces bare /compact for this project.
---

# Project-Aware Compact

Gather project state, build a structured preserve-instructions message, then compact with it.

## Steps

### 1. Gather live state

Read these files in parallel:
- `learnings/EXPERIMENT_QUEUE.md` — which slots are RUNNING, their step/config
- `.monitor/session_state.md` — last Haiku distillation (if exists)

Also check recent results: `ls -t results/*.json | head -5` and read the newest one.

### 2. Build the preserve-instructions message

Construct a terse message covering:

```
SGNNET research session state — preserve the following:

RUNNING EXPERIMENTS:
<slot>: <step> — <config summary> — <key metric if available>
...

KEY FINDINGS THIS SESSION:
- <step>: <result with delta-pp> [CONFIRMED/HYPOTHESIS]
...

DECISIONS MADE:
- <what changed, e.g. "sparsity default updated to 0.98">

ACTIVE HYPOTHESES:
- <compound experiments, pending T2 validations>

NEXT ACTIONS:
- <what was about to happen>

PROJECT STATE:
- Efficiency champion: step605 K=1 KD student — 95.95% @ 0.20M FLOPs, 34,976 params
- D_very (sparsity=0.98): +0.36pp T1 single-seed, multi-seed + T2 running
- ΔW-proj: +1.49pp T1, ±0.20pp seed variance (halves variance)
- Paper scope: vision-only (text gap confirmed negative)
- Tier protocol: T0=20ep/50%, T1=75ep/50%, T2=150ep/100%
```

### 3. Compact with context

After building the message above, use the built-in compact passing that message as the summary argument. This preserves the non-recoverable session state through the context window reduction.

Say "Compacting with project context..." and show the summary you're passing before compacting.
