# Phase 2: Dense Baseline Benchmark - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Run frozen pretrained VGG16 (eval mode, no training) on the Imagenette val set from raw images. Measure top-1 accuracy, per-class precision/recall/F1, and mAP. Count FC parameter total and FLOPs. Cross-check stored soft labels from Phase 1 for fidelity against direct eval. Output: `results/baseline_vgg16.json`.

</domain>

<decisions>
## Implementation Decisions

### FLOPs Counting
- **D-01:** Use `thop` for FLOPs profiling (`pip install thop`). Add to `requirements.txt`. Call `thop.profile(model, inputs=(dummy_input,))` once statically during `eval_baseline.py` — not during any training loop. `thop` is pure Python introspection, MPS-safe, and has no distutils dependency (Python 3.14 compatible).
- **D-02:** `count_flops(model, input_shape)` in `src/utils/metrics.py` uses `thop.profile` under the hood. Returns integer MACs for the FC portion of VGG16.

### Soft Label Quality Thresholds
- **D-03:** Warn-only mode. `soft_label_accuracy_check` is set to `true` in `results/baseline_vgg16.json` as long as the accuracy match between argmax(soft_labels) and direct VGG16 eval is within **0.1%**. Print warnings to stdout (do not fail) if: mean entropy < 0.01 nats (collapsed distribution) or any class represents < 5% of the val set (balance check). Phase 2 is diagnostic — Phase 1 UAT already confirmed the store is correct.

### metrics.py API
- **D-04:** `scores` parameter to `compute_all_metrics` is **softmax probabilities** (float32, shape [N, 10]). Caller is responsible for applying softmax before passing. This is consistent with the stored soft labels format (T=1 softmax, per Phase 1 D-02).
- **D-05:** `MetricsDict` return type is a **plain dict** (not TypedDict or NamedTuple). JSON-serializable directly. Keys match the `baseline_vgg16.json` schema in ROADMAP.md.
- **D-06:** mAP computed via sklearn `average_precision_score` with one-vs-rest binary approach per class, then averaged across 10 classes. Per-class precision/recall/F1 via `sklearn.metrics.precision_recall_fscore_support`.

### Claude's Discretion
- `src/utils/` directory layout (no existing pattern — first utility module)
- Whether `eval_baseline.py` re-uses `VGGExtractor` or loads VGG16 directly (VGGExtractor is already eval+frozen+MPS, recommended to reuse)
- `results/` directory creation (mkdir -p pattern, consistent with Phase 1 `data/`)
- Batch size for val eval pass (256 recommended, consistent with Phase 1)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 2 Spec
- `.planning/ROADMAP.md` §Phase 2 — Locked deliverables: function signatures (`compute_all_metrics`, `count_params`, `count_flops`), `baseline_vgg16.json` schema, Plan 2.1 and 2.2 verification criteria

### Requirements
- `.planning/REQUIREMENTS.md` §Dense Baseline — BASE-01 through BASE-05 acceptance criteria

### Project Constraints
- `.planning/PROJECT.md` §Constraints — 250-line file limit, top-down code style, MPS hardware, 3-day timeline

### Phase 1 Decisions (still in effect)
- `.planning/phases/01-data-pipeline/01-CONTEXT.md` — D-02 (T=1 soft labels, no temperature at storage time), D-05 (HDF5 schema), D-06 (data/ directory layout)

### Architecture Background
- `sparse_geometric_network_report.md` — SGNNET spec; context for why dense baseline matters and what mAP/param comparison feeds into

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/data/extractor.py` — `VGGExtractor`: loads VGG16 pretrained, eval mode, frozen, MPS device auto-detect. `extract_all(dataloader)` returns `(features, soft_labels, labels)`. Hook on `model.avgpool`. **Reuse for Phase 2 eval pass** — no need to reload VGG16.
- `src/data/store.py` — `TensorStore.get_split('val')` loads full val soft labels for Plan 2.2 quality check.
- `src/data/dataset.py` — `ImagenetteDataset` with standard VGG16 preprocessing (resize 224, normalize ImageNet mean/std). `IMAGENET_INDICES` and `CLASS_NAMES` available.

### Established Patterns
- Device detection: `"mps" if torch.backends.mps.is_available() else "cpu"` (from extractor.py)
- `os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")` at top of scripts that use MPS
- `num_workers=0` for DataLoader on macOS (multiprocessing + MPS spawn incompatible on Python 3.14)
- `pin_memory=False` on MPS (auto-detected in Phase 1)

### Integration Points
- `src/utils/metrics.py` is a NEW module — create `src/utils/__init__.py` alongside it
- `results/baseline_vgg16.json` is consumed by Phase 4 (SGNNET comparison) and Phase 6 (final report)
- `thop` must be added to `requirements.txt` before planning

</code_context>

<specifics>
## Specific Ideas

- VGG16 FC params are known: 25088×4096 + 4096 + 4096×4096 + 4096 + 4096×10 + 10 = ~123.6M — `count_params(model)` should count only the `model.classifier` parameters, not the full VGG16
- `thop.profile` returns MACs (multiply-accumulate ops); ROADMAP uses "FLOPs" loosely to mean MACs — document this in comments
- Plan 2.2 verification: argmax accuracy from soft labels should match Plan 2.1 direct eval within 0.1%; expected ~99% top-1 for VGG16 on Imagenette

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-dense-baseline-benchmark*
*Context gathered: 2026-03-23*
