# CUDA 5060 Ti — Launch Checklist (Source of Truth)

Every script launched on slot `5060ti_cuda` must pass this checklist. `scripts/launch_slot.sh` greps for the required markers and refuses to launch otherwise (override: `--unsafe-cuda-launch`). This file is the **single source of truth** — the launch gate, the template, and the skill docs all read from here.

Hardware: RTX 5060 Ti 16GB, Blackwell SM_120, 448 GB/s GDDR7, 48 MB L2, 180W TDP.
Driver path: `/home/indra/sgnnet_bench/venv/bin/python3` on ssh host `5060ti`.

---

## Required markers (grep-enforced by `launch_slot.sh`)

| Marker | Why | Evidence |
|---|---|---|
| `SGNNET_Resonant_CUDA` or `SGNNET_AntiHebbian_CUDA` | Uses `torch.compile` fused routing loop | step500: eager 2.6% → compile 99.6% GPU util, **4× training speedup** |
| `pin_memory=True` | Pinned host memory enables DMA overlap | step408 (general CUDA folklore, +5–15%) |
| `non_blocking=True` | Async H→D transfer concurrent with compute | same as above |
| `use_amp=False` OR `GradScaler(enabled=False)` OR no `GradScaler` at all | fp16+GradScaler is **4.4× slower** than fp32 on Blackwell | step801 direct measurement |

Grep is literal substring — add the tokens anywhere in the script (import, call site, or a comment). The audit is intentionally lenient; the goal is to catch un-audited launches, not to pass judgment on edge cases.

## Explicit override

If your script genuinely requires the eager path (e.g., a diagnostic that requires Python-level introspection, a distillation step that mutates intermediate tensors, a Triton kernel probe), pass:

```bash
scripts/launch_slot.sh 5060ti_cuda scripts/bench_stepXXX.py --unsafe-cuda-launch [extra args...]
```

The override is logged to stderr. Use sparingly — each override should be justifiable in the script docstring.

---

## Known-good CUDA patterns (copy-paste from TEMPLATE_experiment.py)

### Model build (CUDA path)
```python
from src.sgnnet.model_smallworld        import SGNNET_SmallWorld
from src.sgnnet.model_resonant_cuda     import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA

base = SGNNET_SmallWorld(N_hidden=N, ...)
resonant = SGNNET_Resonant_CUDA.from_base(base, K_phase=8, ..., compile=True)
model = SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=1.0, variant="wpos")
```

### DataLoader (CUDA path)
```python
# CUDA-5060ti-validated
tr = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                num_workers=0, pin_memory=True)
# Note: num_workers=0 is optimal for in-RAM tensor datasets;
# increase only if loading from disk.
```

### Training loop transfer
```python
for x, _, y in tr:
    x = x.to(DEVICE, non_blocking=True)
    y = y.to(DEVICE, non_blocking=True)
    # ...
```

### AMP on Blackwell (DON'T do this — keep fp32)
```python
# WRONG for Blackwell training:
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = crit(model(x), y)
scaler.scale(loss).backward()

# RIGHT:
loss = crit(model(x), y)
loss.backward()
```

---

## Batch size guidance (advisory, not gated)

Current experiments default to `BATCH=128`. VRAM headroom at N=2048, D=16 is ~15 GB. Sensible alternatives:

| Batch | VRAM (est) | Use when |
|---|---|---|
| 128 | ~20 MB routing tensors | default (compatible with step410/step605 baselines) |
| 512 | ~80 MB | throughput-focused; amortize dispatch overhead further |
| 1024 | ~160 MB | large-batch studies or SST-2/AG-News with small N_in |

Larger batches reduce epoch wall-clock but may need LR warm-up. Include accuracy control in any batch-size sweep.

---

## Long-win (research-level, not gated)

These are paper-level optimizations still in the queue:

1. **Triton fused scatter-gather** — bench_step832. Expected 2–3× at B=128 (BW-bound).
2. **`torch.compile(fullgraph=True)`** — bench_step830 (audit graph breaks first).
3. **`max-autotune` for inference-only deployment** — +3.5% over `reduce-overhead`, 10–30 min compile cost. Not worth for training.

When these land with clean evidence, add them to the required-markers table above.

---

## How to update this file

1. Run the new optimization against a controlled baseline — record step # and delta.
2. If it materially beats the baseline (>2% throughput or >0.1pp accuracy), add a row to the required-markers table with grep token + evidence.
3. Update `scripts/launch_slot.sh` grep list in the `audit_cuda_script()` function.
4. Update `scripts/TEMPLATE_experiment.py` to include the new pattern.
5. Mention in CLAUDE.md Project State if it changes the defaults.

This file is version-controlled. Every change to required markers should reference the step that validated it.

*Last updated 2026-04-16 — seeded from Sonnet GPU-underutilization investigation (step410 observation).*
