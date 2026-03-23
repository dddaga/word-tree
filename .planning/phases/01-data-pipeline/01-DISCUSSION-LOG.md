# Phase 1: Data Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-23
**Phase:** 01-data-pipeline
**Areas discussed:** Python environment, Soft label temperature, HDF5 compression & chunking, Imagenette download location

---

## Python Environment

| Option | Description | Selected |
|--------|-------------|----------|
| Recreate d_env with Python 3.12 | Most stable Python for PyTorch/h5py/sklearn | |
| Keep Python 3.14 in d_env | Stay on 3.14; install packages into existing venv | ✓ |
| Use system Python | Check pyenv/homebrew for best available version | |

**User's choice:** Keep Python 3.14 in d_env
**Notes:** User opted to stay on the existing venv without recreating it. If package installation fails on 3.14, fall back to pre-release wheels before escalating.

---

## Soft Label Temperature

| Option | Description | Selected |
|--------|-------------|----------|
| Raw T=1 softmax | Store plain VGG16 softmax; apply T=4 per-batch during training | ✓ |
| Pre-apply T=4 before storing | Softer distributions stored; simpler training code; locks in T=4 | |

**User's choice:** Raw T=1 softmax
**Notes:** User initially asked for clarification on what temperature does. After explanation of "dark knowledge" and the flexibility benefit of raw storage, chose raw T=1. Temperature T=4.0 is applied at training time during KL divergence computation.

---

## HDF5 Compression & Chunking

| Option | Description | Selected |
|--------|-------------|----------|
| No compression, sample-level chunks | ~1.34GB, fastest random access | ✓ |
| gzip level-1 + sample-level chunks | ~400-600MB, small decompression cost | |

**User's choice:** No compression
**Notes:** User asked a clarifying question about using a "tensor store" — confirmed that HDF5 IS the tensor store (with CSV manifest for index mapping). This is already specified in the roadmap. User then confirmed no compression.

---

## Imagenette Download Location

| Option | Description | Selected |
|--------|-------------|----------|
| data/ inside project root | Self-contained; add data/ to .gitignore | ✓ |
| Absolute path outside project | For sharing dataset across multiple projects | |

**User's choice:** data/ inside project root
**Notes:** T9 drive has sufficient capacity for ~2.8GB total (images + HDF5 store). .gitignore will exclude the data/ directory.

---

## Claude's Discretion

- Extraction batch size (256 recommended)
- Imagenette download via torchvision.datasets.Imagenette
- Resumability check (skip re-extraction if store.h5 already present with correct shape)
- HDF5 write strategy (pre-allocate and fill)

## Deferred Ideas

None.
