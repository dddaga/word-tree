---
phase: 01-data-pipeline
plan: 02
subsystem: data
tags: [torch, vgg16, mps, h5py, hdf5, feature-extraction, soft-labels]

requires:
  - phase: 01-01
    provides: "VGGExtractor, TensorStore, IMAGENET_INDICES, get_dataloader"
provides:
  - "src/data/extractor.py — VGGExtractor with MPS, eval mode, no_grad, full memory cap removed"
  - "src/data/store.py — TensorStore with HDF5 read/write and CSV manifest"
  - "data/store.h5 — 13394 records: train/features [9469,25088], val/features [3925,25088], soft_labels, labels"
  - "data/manifest.csv — 13394 rows: index, split, class_name, class_idx, h5_idx"
  - "scripts/extract_features.py — reproducible extraction runner"
affects: [02-baseline, 03-sgnnet, 04-training, 05-pca]

tech-stack:
  added: [h5py, pandas]
  patterns: [mps-aware-dataloader, hook-based-feature-extraction, sample-chunked-hdf5]

key-files:
  created:
    - src/data/extractor.py
    - src/data/store.py
    - scripts/extract_features.py
    - data/store.h5
    - data/manifest.csv
  modified:
    - src/data/__init__.py
    - src/data/dataset.py

key-decisions:
  - "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 set in extractor.py — removes 50% MPS memory cap"
  - "num_workers=0 for extraction — multiprocessing spawn incompatible with MPS on macOS"
  - "pin_memory disabled on MPS — fixed in get_dataloader to auto-detect"
  - "torch.mps.empty_cache() called between train/val passes to reclaim unified memory"

requirements-completed: [DATA-02, DATA-03, DATA-04, DATA-05, DATA-06]

duration: ~3min
completed: 2026-03-23
---

# Phase 1 Plan 2: Feature Extraction & Tensor Store Summary

**VGG16 pre-FC features (25088-dim) and T=1 soft labels (10-dim) extracted for all 13,394 Imagenette images via Apple Metal (MPS), persisted to HDF5 with sample-level chunking and CSV manifest**

## Performance

- **Device:** MPS (Apple Metal)
- **Train extraction:** ~1m 24s (37 batches × 256)
- **Val extraction:** ~33s (16 batches × 256)
- **Files modified:** 5

## Accomplishments

- VGGExtractor: pretrained VGG16 frozen in eval mode, `torch.no_grad()` throughout, hook on `avgpool`
- Full MPS memory enabled via `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- HDF5 store written with locked schema (D-05): sample-level chunking, no compression (D-04)
- CSV manifest: 13,394 rows, all 10 classes in both splits
- Verified: shapes, dtypes, soft label sums ≈ 1.0, indexed access, manifest columns

## Task Commits

1. **Task 1: VGG16 Feature Extractor** — `0fdc0a6` (feat)
2. **Task 2: TensorStore + CSV Manifest modules** — `c0042e4` (feat)
3. **Task 2: Extraction run + pin_memory fix** — `5d4532f` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Bug] pin_memory=True not supported on MPS**
- DataLoader emitted warning; MPS and CPU share unified memory — pin_memory is a no-op
- Fix: auto-detect in `get_dataloader`: `pin_memory = not torch.backends.mps.is_available()`

**2. [Bug] num_workers > 0 crashes with multiprocessing spawn from script**
- macOS MPS + Python 3.14 multiprocessing spawn can't resolve `<stdin>` origin
- Fix: `num_workers=0` in extraction runner (single-threaded data loading)
- Impact: ~2.2s/batch vs potentially faster with workers; total time acceptable (~2min)

## Self-Check: PASSED

- `data/store.h5` shape: train [9469,25088] float32, val [3925,25088] float32 ✓
- Soft label sums ≈ 1.0 (atol 1e-4) ✓
- `data/manifest.csv`: 13394 rows, 10 unique classes, correct columns ✓
- `TensorStore.read("train", 42)` returns `(feat[25088], sl[10], label=7)` ✓

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-23*
