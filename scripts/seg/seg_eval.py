"""Shared eval / target-cache / MAC helpers for the seg line.

Factored out for seg_step007 (distillation) rather than copied a third time. seg_step001 and
seg_step002 keep their own inlined copies deliberately — their results are already recorded and
rewriting their measurement code would put those numbers in question for no gain.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.seg.seg_common import (C, EVENT_CLASSES, build_teacher, list_images, load_image,
                                    rasterize)

ROOT = Path(__file__).parent.parent.parent
BATCH = 32


def cache_targets(split: str, n: int, res: int, grid: int, teacher_device: str = "cpu"):
    """One teacher pass, cached to disk. Filename matches seg_step001/002 so the cache is shared."""
    cp = ROOT / "data" / f"seg_targets_{split}_n{n}_r{res}_g{grid}.pt"
    if cp.exists():
        d = torch.load(cp)
        return d["x"], d["y"]
    paths = list_images(split, n)
    tdev = torch.device(teacher_device)
    teacher = build_teacher(tdev)
    xs, ys, t0 = [], [], time.time()
    for i in range(0, len(paths), BATCH):
        bx = torch.stack([load_image(p, res) for p in paths[i:i + BATCH]])
        with torch.no_grad():
            dets = teacher([b for b in bx.to(tdev)])
        for d in dets:
            ys.append(rasterize(d["boxes"].cpu(), d["labels"].cpu(), d["scores"].cpu(), res, grid))
        xs.append(bx)
        if i % (BATCH * 10) == 0:
            print(f"  teacher {split} {i}/{len(paths)} [{time.time()-t0:.0f}s]", flush=True)
    x, y = torch.cat(xs), torch.stack(ys)
    torch.save({"x": x, "y": y}, cp)
    return x, y


def count_macs(model, res: int) -> int:
    """Conv MACs for one frame. Groups-aware — depthwise arms would otherwise read ~64x too dear."""
    macs, hooks = [0], []

    def hook(m, i, o):
        macs[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, res, res, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    return macs[0]


def average_precision(score: torch.Tensor, target: torch.Tensor) -> float:
    tgt = (target > 0.3).float()
    if tgt.sum() == 0:
        return float("nan")
    hits = tgt[score.argsort(descending=True)]
    prec = hits.cumsum(0) / torch.arange(1, len(hits) + 1, device=hits.device)
    return float((prec * hits).sum() / hits.sum())


def evaluate(model, vx, vy, device) -> dict:
    model.eval()
    outs, bce = [], 0.0
    with torch.no_grad():
        for i in range(0, len(vx), BATCH):
            cl = model.class_logits(model(vx[i:i + BATCH].to(device)))
            t = vy[i:i + BATCH].to(device)
            bce += F.binary_cross_entropy_with_logits(cl, t, reduction="sum").item()
            outs.append(cl.cpu())
    cl = torch.cat(outs)
    aps = [average_precision(cl[:, c].flatten(), vy[:, c].flatten()) for c in range(C)]
    valid = [a for a in aps if a == a]
    return {"bce": bce / vy.numel(),
            "mAP": sum(valid) / len(valid) if valid else float("nan"),
            "ap_per_class": {EVENT_CLASSES[c]: round(aps[c], 4) for c in range(C)}}
