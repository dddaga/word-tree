# Phase 2: Dense Baseline Benchmark - Research

**Researched:** 2026-03-23
**Domain:** PyTorch model evaluation, metrics computation, FLOPs profiling
**Confidence:** HIGH

## Summary

Phase 2 runs a frozen pretrained VGG16 on the Imagenette val set (3,925 images, 10 classes) to establish the dense baseline benchmark. No training occurs -- this is pure inference + metric collection. The phase creates a reusable `src/utils/metrics.py` module (used again in Phases 4 and 6) and two scripts: `eval_baseline.py` for direct VGG16 evaluation and `verify_soft_labels.py` for cross-checking Phase 1's stored soft labels.

The core technical work is (a) computing per-class precision/recall/F1/AP using sklearn on softmax probability outputs, (b) counting FC-only parameters (123,642,856 verified), and (c) profiling FC FLOPs using `thop`. All components are well-understood; the main risks are getting the mAP computation right (one-vs-rest binary AP per class) and ensuring `thop` installs cleanly on Python 3.14.

**Primary recommendation:** Reuse `VGGExtractor` from Phase 1 for the eval pass (it already handles model loading, eval mode, freezing, MPS device, and 10-class logit selection). Build `metrics.py` with a clean functional API that accepts softmax probabilities -- this module will be reused verbatim in Phase 4 (SGNNET) and Phase 6 (comparative analysis).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Use `thop` for FLOPs profiling (`pip install thop`). Add to `requirements.txt`. Call `thop.profile(model, inputs=(dummy_input,))` once statically during `eval_baseline.py` -- not during any training loop. `thop` is pure Python introspection, MPS-safe, and has no distutils dependency (Python 3.14 compatible).
- **D-02:** `count_flops(model, input_shape)` in `src/utils/metrics.py` uses `thop.profile` under the hood. Returns integer MACs for the FC portion of VGG16.
- **D-03:** Warn-only mode. `soft_label_accuracy_check` is set to `true` in `results/baseline_vgg16.json` as long as the accuracy match between argmax(soft_labels) and direct VGG16 eval is within 0.1%. Print warnings to stdout (do not fail) if: mean entropy < 0.01 nats (collapsed distribution) or any class represents < 5% of the val set (balance check). Phase 2 is diagnostic -- Phase 1 UAT already confirmed the store is correct.
- **D-04:** `scores` parameter to `compute_all_metrics` is softmax probabilities (float32, shape [N, 10]). Caller is responsible for applying softmax before passing. This is consistent with the stored soft labels format (T=1 softmax, per Phase 1 D-02).
- **D-05:** `MetricsDict` return type is a plain dict (not TypedDict or NamedTuple). JSON-serializable directly. Keys match the `baseline_vgg16.json` schema in ROADMAP.md.
- **D-06:** mAP computed via sklearn `average_precision_score` with one-vs-rest binary approach per class, then averaged across 10 classes. Per-class precision/recall/F1 via `sklearn.metrics.precision_recall_fscore_support`.

### Claude's Discretion
- `src/utils/` directory layout (no existing pattern -- first utility module)
- Whether `eval_baseline.py` re-uses `VGGExtractor` or loads VGG16 directly (VGGExtractor is already eval+frozen+MPS, recommended to reuse)
- `results/` directory creation (mkdir -p pattern, consistent with Phase 1 `data/`)
- Batch size for val eval pass (256 recommended, consistent with Phase 1)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BASE-01 | Pretrained VGG16 (frozen, eval mode) run on Imagenette val set -- no training | Reuse `VGGExtractor` from Phase 1 which already handles frozen eval mode + MPS |
| BASE-02 | Top-1 accuracy (overall and per-class) recorded as the benchmark | `compute_all_metrics` using `np.argmax` for predictions + sklearn classification report |
| BASE-03 | mAP computed across all 10 Imagenette classes (each class treated as binary) | sklearn `average_precision_score` with one-hot labels, one-vs-rest per class |
| BASE-04 | Per-class precision, recall, and F1 recorded for all 10 classes | sklearn `precision_recall_fscore_support` with `average=None` for per-class |
| BASE-05 | VGG16 FC parameter count (~123.6M) and FLOPs per inference recorded | `count_params` via `sum(p.numel())` on classifier; `count_flops` via `thop.profile` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10.0 | Model loading, inference, tensor ops | Already installed in d_env |
| torchvision | (matches torch 2.10) | VGG16 pretrained weights, transforms | Already installed in d_env |
| scikit-learn | 1.8.0 | Metrics: AP, precision/recall/F1 | Already installed in d_env |
| thop | 0.1.1.post2209072238 | FLOPs/MACs profiling | Locked by D-01; dry-run install succeeds on Python 3.14 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (installed) | Array ops for metric computation | Softmax argmax, one-hot encoding |
| h5py | (installed) | Read stored soft labels for Plan 2.2 | `TensorStore.get_split('val')` |
| tqdm | (installed) | Progress bars during val inference | `eval_baseline.py` loop |

**Installation (new dependency only):**
```bash
d_env/bin/pip install thop>=0.1.1
```
Then add `thop>=0.1.1` to `requirements.txt`.

## Architecture Patterns

### Recommended Project Structure
```
src/
  utils/
    __init__.py          # NEW - empty or re-exports
    metrics.py           # NEW - compute_all_metrics, count_params, count_flops
scripts/
  eval_baseline.py       # NEW - Plan 2.1
  verify_soft_labels.py  # NEW - Plan 2.2
results/
  baseline_vgg16.json    # NEW - output artifact
```

### Pattern 1: Reuse VGGExtractor for Inference
**What:** `VGGExtractor` already loads VGG16 pretrained, sets eval mode, freezes weights, handles MPS device detection, and selects 10 Imagenette logits from the 1000-class output. `extract_all(dataloader)` returns `(features, soft_labels, labels)` where `soft_labels` are already softmax probabilities.
**When to use:** `eval_baseline.py` should instantiate `VGGExtractor` and call `extract_all` on the val dataloader. The returned `soft_labels` are the scores needed by `compute_all_metrics`. No need to write a separate VGG16 loading path.
**Example:**
```python
# In eval_baseline.py
from src.data.extractor import VGGExtractor
from src.data.dataset import get_dataloader, IMAGENETTE_CLASSES

ext = VGGExtractor()
loader = get_dataloader(split="val", batch_size=256, num_workers=0)
_, soft_labels, labels = ext.extract_all(loader, desc="Evaluating VGG16 on val")
# soft_labels: [3925, 10] float32, already softmax
```

### Pattern 2: Functional Metrics API
**What:** `compute_all_metrics` is a pure function that takes numpy arrays and returns a plain dict. No class state needed. This makes it trivially reusable in Phases 4 and 6.
**When to use:** Any eval script that has softmax scores and ground-truth labels.
**Example:**
```python
def compute_all_metrics(
    scores: np.ndarray,       # [N, 10] softmax probabilities
    labels: np.ndarray,       # [N] integer ground-truth
    class_names: list[str],   # 10 class names
) -> dict:
    preds = np.argmax(scores, axis=1)
    top1_acc = float(np.mean(preds == labels))
    # Per-class metrics via sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(class_names))),
    )
    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        per_class_acc[name] = float(np.mean(preds[mask] == labels[mask]))
    # mAP: one-vs-rest binary AP per class
    labels_onehot = np.eye(len(class_names))[labels]  # [N, 10]
    per_class_ap = {}
    for i, name in enumerate(class_names):
        per_class_ap[name] = float(average_precision_score(
            labels_onehot[:, i], scores[:, i],
        ))
    mAP = float(np.mean(list(per_class_ap.values())))
    # Assemble result dict matching ROADMAP schema
    ...
```

### Pattern 3: Script Entry Point with MPS Env Setup
**What:** All scripts follow the established pattern: set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` at top, use `num_workers=0`, `pin_memory=False`.
**Example:**
```python
"""Evaluate frozen VGG16 on Imagenette val set."""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import json
from pathlib import Path
# ... rest of imports
```

### Anti-Patterns to Avoid
- **Loading VGG16 separately from VGGExtractor:** Duplicates device detection, weight loading, and logit selection logic. Reuse VGGExtractor.
- **Computing FLOPs in a loop:** D-01 mandates a single static `thop.profile` call, not per-batch profiling.
- **Returning Tensor from metrics functions:** Metrics should work with numpy arrays for JSON serialization. Convert tensors to numpy at the boundary.
- **Using `model.eval()` without `torch.no_grad()`:** Both are needed -- `eval()` sets dropout/BN behavior, `no_grad()` disables gradient tracking. VGGExtractor already uses `torch.no_grad()` in `extract_batch`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Average Precision | Custom PR curve integration | `sklearn.metrics.average_precision_score` | Edge cases with tied scores, zero-support classes |
| Precision/Recall/F1 | Manual TP/FP/FN counting | `sklearn.metrics.precision_recall_fscore_support` | Handles multi-class correctly, zero-division warnings |
| FLOPs counting | Manual layer-by-layer multiplication | `thop.profile` | Handles all layer types, bias terms, activation ops |
| One-hot encoding | Manual loop | `np.eye(num_classes)[labels]` | Vectorized, correct shape |

**Key insight:** sklearn's metrics API handles edge cases (zero-support classes, tied predictions) that custom implementations commonly miss. `thop` accounts for bias parameters and non-linear layers that manual FLOPs calculations often forget.

## Common Pitfalls

### Pitfall 1: MACs vs FLOPs Confusion
**What goes wrong:** `thop.profile` returns MACs (multiply-accumulate operations), but documentation and literature often use "FLOPs" loosely. 1 MAC = 2 FLOPs in strict accounting.
**Why it happens:** No industry standard. PyTorch ecosystem tools return MACs labeled as "FLOPs".
**How to avoid:** Store as MACs, document in comments and JSON output that the value is MACs. The ROADMAP uses "FLOPs" loosely to mean MACs (per CONTEXT.md specifics section). Use a descriptive key like `"flops_fc_per_inference"` and add a `"flops_note": "MACs from thop.profile"` field.
**Warning signs:** Values that are exactly 2x expected.

### Pitfall 2: thop Profiling the Full Model Instead of FC Only
**What goes wrong:** `thop.profile(model, inputs=(dummy_input,))` profiles the ENTIRE VGG16 (CNN + FC), but we only want FC FLOPs since SGNNET replaces only the FC layers.
**Why it happens:** `thop.profile` takes the full model by default.
**How to avoid:** Profile `model.classifier` separately with a dummy input of shape `[1, 25088]` (the FC input dimension). This gives FC-only MACs.
```python
from thop import profile
dummy_fc_input = torch.randn(1, 25088)
macs, _ = profile(model.classifier, inputs=(dummy_fc_input,), verbose=False)
```
**Warning signs:** FLOPs value is ~15.5B (full VGG16) instead of ~124M (FC only).

### Pitfall 3: average_precision_score Requires Proper One-vs-Rest Setup
**What goes wrong:** Passing multi-class labels directly to `average_precision_score` produces wrong results.
**Why it happens:** `average_precision_score` expects binary true labels (0/1) and continuous scores for a single class.
**How to avoid:** One-hot encode labels, then loop over classes calling `average_precision_score(y_true_binary[:, c], scores[:, c])` for each class c.
**Warning signs:** mAP values that seem implausibly low or high.

### Pitfall 4: VGG16 FC Has 1000 Outputs, Not 10
**What goes wrong:** Counting all 123.6M params as the baseline, but VGG16's final layer is 4096->1000 while SGNNET targets 4096->10. The comparison may not be strictly apples-to-apples.
**Why it happens:** VGG16 was trained on 1000 ImageNet classes.
**How to avoid:** The ROADMAP and CONTEXT.md explicitly state to count all `model.classifier` parameters (123,642,856). This is intentional -- it represents the full dense FC cost that SGNNET replaces. Document this in the JSON output. SGNNET's 10-class output is part of what makes it so much smaller.

### Pitfall 5: num_workers > 0 on macOS with MPS
**What goes wrong:** DataLoader with multiprocessing workers crashes on Python 3.14 + MPS.
**Why it happens:** `spawn` start method + MPS backend incompatibility on macOS.
**How to avoid:** Always pass `num_workers=0` when calling `get_dataloader`. The existing `get_dataloader` defaults to `num_workers=4`, so the caller MUST override it.
**Warning signs:** `RuntimeError` or hang at DataLoader iteration start.

## Code Examples

### compute_all_metrics Skeleton
```python
# Source: sklearn docs + CONTEXT.md D-04/D-05/D-06
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)

def compute_all_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute full classification metrics from softmax probabilities.

    Parameters
    ----------
    scores : np.ndarray, shape [N, C]
        Softmax probability outputs (rows sum to 1).
    labels : np.ndarray, shape [N]
        Ground-truth integer class labels.
    class_names : list[str]
        Human-readable class names, length C.

    Returns
    -------
    dict
        JSON-serializable metrics dict matching ROADMAP schema.
    """
    num_classes = len(class_names)
    preds = np.argmax(scores, axis=1)
    top1_acc = float(np.mean(preds == labels))

    # Per-class precision, recall, F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None,
        labels=list(range(num_classes)), zero_division=0,
    )

    # Per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        mask = labels == i
        per_class_acc.append(float(np.mean(preds[mask] == i)) if mask.sum() > 0 else 0.0)

    # mAP: one-vs-rest AP per class
    labels_onehot = np.eye(num_classes, dtype=np.float32)[labels]
    per_class_ap = []
    for i in range(num_classes):
        ap = float(average_precision_score(labels_onehot[:, i], scores[:, i]))
        per_class_ap.append(ap)
    mAP = float(np.mean(per_class_ap))

    # Assemble per-class dict
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "accuracy": per_class_acc[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "AP": per_class_ap[i],
        }

    return {
        "top1_accuracy": top1_acc,
        "mAP": mAP,
        "per_class": per_class,
    }
```

### count_params for FC Only
```python
def count_params(model: torch.nn.Module) -> int:
    """Count total parameters in model.classifier (FC layers only)."""
    return sum(p.numel() for p in model.classifier.parameters())
```

### count_flops for FC Only
```python
import torch
from thop import profile

def count_flops(model: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """Count MACs for model.classifier using thop.

    Parameters
    ----------
    model : torch.nn.Module
        Full VGG16 model (only classifier is profiled).
    input_shape : tuple
        Shape of FC input, e.g. (1, 25088).

    Returns
    -------
    int
        MACs (multiply-accumulate operations) for the FC portion.
    """
    dummy = torch.randn(*input_shape)
    macs, _ = profile(model.classifier, inputs=(dummy,), verbose=False)
    return int(macs)
```

### baseline_vgg16.json Expected Schema
```json
{
    "model": "VGG16 (frozen)",
    "fc_params": 123642856,
    "top1_accuracy": 0.97,
    "mAP": 0.98,
    "per_class": {
        "tench": {"accuracy": 0.98, "precision": 0.97, "recall": 0.98, "f1": 0.97, "AP": 0.99},
        "English springer": {"accuracy": 0.97, "...": "..."}
    },
    "flops_fc_per_inference": 123000000,
    "flops_note": "MACs from thop.profile on model.classifier",
    "soft_label_accuracy_check": true
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torchvision.models.vgg16(pretrained=True)` | `vgg16(weights=VGG16_Weights.IMAGENET1K_V1)` | torchvision 0.13+ | Old `pretrained` kwarg deprecated |
| Manual FLOPs counting | `thop.profile` | Stable since 2022 | Handles all layer types automatically |
| `sklearn.metrics.classification_report` (string) | `precision_recall_fscore_support` (arrays) | Always available | Returns numeric arrays, not formatted strings |

**Deprecated/outdated:**
- `pretrained=True` kwarg in torchvision model constructors -- use `weights=` enum instead (already correct in VGGExtractor)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (not yet installed -- Wave 0 gap) |
| Config file | none -- see Wave 0 |
| Quick run command | `d_env/bin/python -m pytest tests/ -x -q` |
| Full suite command | `d_env/bin/python -m pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BASE-01 | VGG16 runs on val set, frozen, eval mode | smoke | `d_env/bin/python scripts/eval_baseline.py` (produces JSON) | -- Wave 0 |
| BASE-02 | Top-1 accuracy recorded in JSON | smoke | Check `results/baseline_vgg16.json` has `top1_accuracy` key | -- Wave 0 |
| BASE-03 | mAP recorded in JSON | smoke | Check `results/baseline_vgg16.json` has `mAP` key | -- Wave 0 |
| BASE-04 | Per-class P/R/F1 recorded | smoke | Check `per_class` dict has all 10 classes with required keys | -- Wave 0 |
| BASE-05 | FC param count and FLOPs recorded | smoke | Check `fc_params` and `flops_fc_per_inference` keys exist | -- Wave 0 |

### Sampling Rate
- **Per task commit:** `d_env/bin/python scripts/eval_baseline.py && python -c "import json; d=json.load(open('results/baseline_vgg16.json')); print(d.keys())"`
- **Per wave merge:** Full JSON schema validation
- **Phase gate:** `baseline_vgg16.json` exists with all required keys; `soft_label_accuracy_check` is `true`

### Wave 0 Gaps
- [ ] `tests/test_metrics.py` -- unit tests for `compute_all_metrics`, `count_params`, `count_flops` with small synthetic data
- [ ] `pytest` install: `d_env/bin/pip install pytest` -- not currently in requirements.txt
- [ ] Verification can be done via script execution + JSON schema check (simpler than full pytest for this eval-only phase)

**Note:** Given this phase is purely evaluation (no training, no model modification), the primary validation is that scripts run successfully and produce correctly-structured JSON output. Unit testing `compute_all_metrics` with synthetic data is valuable for catching edge cases in the metrics math.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| torch | Model inference | yes | 2.10.0 | -- |
| torchvision | VGG16 weights | yes | (matches torch) | -- |
| scikit-learn | Metrics computation | yes | 1.8.0 | -- |
| thop | FLOPs profiling | no (not yet installed) | 0.1.1 (available on PyPI) | Manual calculation |
| MPS backend | GPU acceleration | yes | -- | CPU fallback |
| d_env venv | Python 3.14 runtime | yes | 3.14.3 | -- |
| h5py | Read stored soft labels | yes | (installed) | -- |

**Missing dependencies with no fallback:**
- None -- all critical dependencies are available or installable

**Missing dependencies with fallback:**
- `thop`: Not yet installed, but dry-run confirms installable. If install fails, FC FLOPs can be computed manually: `(25088*4096 + 4096) + (4096*4096 + 4096) + (4096*1000 + 1000)` MACs for Linear layers (manual fallback is acceptable since FC layers are simple).

## Open Questions

1. **Exact class name format in JSON keys**
   - What we know: ROADMAP shows `"english_springer"` (snake_case) but `IMAGENETTE_CLASSES` in dataset.py uses `"English springer"` (title case with space)
   - What's unclear: Which format to use in `baseline_vgg16.json`
   - Recommendation: Use the human-readable format from `IMAGENETTE_CLASSES` (e.g., `"English springer"`) since `class_names` is passed directly to `compute_all_metrics`. This keeps the JSON readable and consistent with the dataset module. The ROADMAP example was illustrative, not prescriptive.

2. **VGG16 classifier includes dropout layers**
   - What we know: `model.classifier` is `Linear->ReLU->Dropout->Linear->ReLU->Dropout->Linear`. Dropout is inactive in eval mode but `count_params` counts all params in the Sequential.
   - What's unclear: Whether dropout layers have params (they don't -- 0 learnable params)
   - Recommendation: No issue. Dropout/ReLU have zero parameters. `count_params` on `model.classifier` = 123,642,856 (verified). This differs slightly from ROADMAP's 123,646,952 estimate (ROADMAP used approximate math: 25088*4096+4096+4096*4096+4096+4096*10+10 = 123,646,954). The actual number from PyTorch is authoritative.

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10.0 -- `torchvision.models.vgg16` verified on local d_env
- scikit-learn 1.8.0 -- `average_precision_score`, `precision_recall_fscore_support` verified importable
- VGG16 FC param count: 123,642,856 -- verified via `sum(p.numel() for p in model.classifier.parameters())`
- thop 0.1.1 -- available on PyPI, dry-run install succeeds on Python 3.14

### Secondary (MEDIUM confidence)
- thop API: `profile(model, inputs=(dummy,))` returns `(macs, params)` -- based on thop README and widespread PyTorch ecosystem usage

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries verified installed/installable on d_env with Python 3.14
- Architecture: HIGH - reusing Phase 1 VGGExtractor, sklearn metrics API is well-documented
- Pitfalls: HIGH - MACs/FLOPs confusion and FC-only profiling are well-known issues; num_workers=0 already established in Phase 1

**Research date:** 2026-03-23
**Valid until:** 2026-04-23 (stable domain, no fast-moving dependencies)
