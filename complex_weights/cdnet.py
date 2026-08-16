# CDnet2014 scene loader (foreground segmentation, real data).
# GT codes: 0=bg, 50=shadow(->bg), 85=unknown(ignore), 170=non-ROI(ignore), 255=fg.
# Eval frames = temporalROI range; ROI.bmp restricts valid pixels.
# Loads a whole scene into CPU float tensors; batches move to GPU in the trainer.
import os
import numpy as np
import torch
from PIL import Image


def _roi_range(scene):
    with open(os.path.join(scene, "temporalROI.txt")) as f:
        a, b = f.read().split()
    return int(a), int(b)


def _load_img(path, size, nearest=False):
    im = Image.open(path)
    im = im.resize((size[1], size[0]), Image.NEAREST if nearest else Image.BILINEAR)
    return np.asarray(im)


def load_scene(scene, size=(128, 128), train_n=200, seed=0):
    """Returns dict with train/test tensors: x [N,3,H,W], y [N,1,H,W] fg in {0,1},
    v [N,1,H,W] valid mask (1=count in loss/metric)."""
    lo, hi = _roi_range(scene)
    roi = _load_img(os.path.join(scene, "ROI.bmp"), size, nearest=True)
    roi = torch.from_numpy((roi > 127).astype(np.float32))
    if roi.ndim == 3:
        roi = roi[..., 0]
    xs, ys, vs = [], [], []
    for i in range(lo, hi + 1):
        xp = os.path.join(scene, "input", f"in{i:06d}.jpg")
        gp = os.path.join(scene, "groundtruth", f"gt{i:06d}.png")
        if not (os.path.exists(xp) and os.path.exists(gp)):
            continue
        x = _load_img(xp, size).astype(np.float32) / 255.0
        if x.ndim == 2:
            x = np.stack([x] * 3, -1)
        g = _load_img(gp, size, nearest=True).astype(np.int32)
        if g.ndim == 3:
            g = g[..., 0]
        y = (g == 255).astype(np.float32)
        valid = ((g != 85) & (g != 170)).astype(np.float32)
        xs.append(torch.from_numpy(x).permute(2, 0, 1))
        ys.append(torch.from_numpy(y)[None])
        vs.append(torch.from_numpy(valid)[None] * roi[None])
    x = torch.stack(xs); y = torch.stack(ys); v = torch.stack(vs)
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    tr, te = perm[:train_n], perm[train_n:]
    return {
        "train": (x[tr], y[tr], v[tr]),
        "test": (x[te], y[te], v[te]),
        "meta": {"n": n, "train_n": len(tr), "test_n": len(te), "size": size},
    }


def f_measure_masked(logits, y, v, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float() * v
    yv = y * v
    tp = (pred * yv).sum()
    fp = (pred * (1 - yv) * v).sum()
    fn = ((1 - pred) * yv * v).sum()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return (2 * prec * rec / (prec + rec + 1e-6)).item()
