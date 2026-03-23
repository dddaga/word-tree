# Phase 2: Dense Baseline Benchmark - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-23
**Phase:** 02-dense-baseline-benchmark
**Areas discussed:** FLOPs counting, Soft label quality thresholds, metrics.py API shape

---

## FLOPs Counting

| Option | Description | Selected |
|--------|-------------|----------|
| Manual formula | Hardcode 2×(25088×4096 + 4096×4096 + 4096×10) — zero deps | |
| Hook-based model walk | Walk nn.Linear layers dynamically — no new dep, general | |
| Profiling library (thop/fvcore) | Add thop or fvcore to requirements.txt | ✓ |

**User's choice:** Profiling library — then narrowed to `thop` after research
**Notes:** User asked whether FLOPs profiling happens during training (it does not — one-time static measurement during eval script). After clarification, preferred a library. Research confirmed `thop` is MPS-safe (pure Python introspection), Python 3.14 compatible, actively maintained by Ultralytics. `fvcore` ruled out as heavier and less tested on Apple Silicon.

---

## Soft Label Quality Thresholds

| Option | Description | Selected |
|--------|-------------|----------|
| Warn-only, never fail | Warnings for entropy/balance issues, but always pass if accuracy matches | ✓ |
| Fail on entropy collapse | false if mean entropy < 0.01 nats | |
| Strict — fail on all checks | false on any mismatch, entropy collapse, or class imbalance | |

**User's choice:** Warn-only
**Notes:** Phase 2 is diagnostic. Phase 1 UAT already confirmed the store is correct. `soft_label_accuracy_check: true` as long as argmax accuracy matches within 0.1%. Warnings printed for entropy < 0.01 or class < 5% of val.

---

## metrics.py API Shape

### scores parameter type

| Option | Description | Selected |
|--------|-------------|----------|
| Softmax probabilities | Caller passes softmax(logits) — consistent with HDF5 format | ✓ |
| Raw logits | Function applies softmax internally | |

**User's choice:** Softmax probabilities
**Notes:** Consistent with Phase 1 D-02 (soft labels stored as T=1 softmax). Caller responsibility to apply softmax.

### MetricsDict return type

| Option | Description | Selected |
|--------|-------------|----------|
| Plain dict | JSON-serializable directly, no boilerplate | ✓ |
| TypedDict | IDE autocomplete, more structured | |

**User's choice:** Plain dict
**Notes:** Single-experiment script — TypedDict overhead not warranted.

---

## Claude's Discretion

- `src/utils/` directory layout
- Whether `eval_baseline.py` reuses VGGExtractor or loads VGG16 directly
- `results/` directory creation pattern
- Batch size for val eval pass

## Deferred Ideas

None
