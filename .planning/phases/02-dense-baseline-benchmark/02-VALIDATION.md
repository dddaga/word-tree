---
phase: 2
slug: dense-baseline-benchmark
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (Wave 0 installs) |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `d_env/bin/python -m pytest tests/ -x -q` |
| **Full suite command** | `d_env/bin/python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `d_env/bin/python scripts/eval_baseline.py` and check JSON output
- **After every plan wave:** Run `d_env/bin/python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green + `results/baseline_vgg16.json` exists with all keys
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 0 | BASE-01..05 | setup | `d_env/bin/pip install thop pytest && d_env/bin/python -c "import thop, pytest"` | Wave 0 | ⬜ pending |
| 2-01-02 | 01 | 1 | BASE-01..05 | unit | `d_env/bin/python -m pytest tests/test_metrics.py -x -q` | Wave 0 | ⬜ pending |
| 2-01-03 | 01 | 1 | BASE-01..05 | smoke | `d_env/bin/python scripts/eval_baseline.py && python -c "import json; d=json.load(open('results/baseline_vgg16.json')); assert all(k in d for k in ['top1_accuracy','mAP','per_class','fc_params','flops_fc_per_inference'])"` | Wave 0 | ⬜ pending |
| 2-02-01 | 02 | 2 | BASE-02..04 | smoke | `d_env/bin/python scripts/verify_soft_labels.py && python -c "import json; d=json.load(open('results/baseline_vgg16.json')); assert d['soft_label_accuracy_check'] == True"` | Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `d_env/bin/pip install thop>=0.1.1` — FLOPs profiling (not yet in requirements.txt)
- [ ] `d_env/bin/pip install pytest` — test framework (not yet installed)
- [ ] `tests/test_metrics.py` — unit tests for `compute_all_metrics`, `count_params`, `count_flops` with synthetic data
- [ ] `tests/__init__.py` — empty package marker
- [ ] `results/` directory — `mkdir -p results`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| VGG16 top-1 accuracy ~97-99% on Imagenette | BASE-02 | Expected range check requires domain knowledge | Run `eval_baseline.py`, confirm `top1_accuracy` is between 0.95 and 1.0 |
| mAP ~0.97-0.99 for VGG16 on Imagenette | BASE-03 | Same — sanity range | Confirm `mAP` is between 0.95 and 1.0 in `baseline_vgg16.json` |
| Soft label argmax matches direct eval within 0.1% | BASE-01 | Tolerance check | Confirm `soft_label_accuracy_check: true` in JSON and check stdout for warnings |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
