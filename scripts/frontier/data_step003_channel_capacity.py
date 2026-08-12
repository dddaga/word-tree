"""data_step003: does dropping a low-SNR channel help MORE at low model capacity?

USER HYPOTHESIS: a pixel's weak channels carry a worse signal-to-noise ratio than its
strong (high-intensity) channel. Zeroing the weak channel(s) removes variation/noise, so
a SMALL network (limited capacity to model that noise) should benefit; a LARGE network can
model the noise itself, so `full` should catch up and overtake. => a CROSSOVER vs capacity.

We sweep 3 input variants x rising model depth and read the saturation curves:
  full     : raw RGB (control)
  max_only : keep the argmax channel value, zero the other two   (variant 1)
  min_zero : zero ONLY the argmin (weakest) channel, keep top two (variant 2, conservative)

For each (mode, depth) we log best val-acc, param count, and wall train time -> which mode
saturates first, which peaks higher, and whether `full` overtakes as capacity grows.

Prior: data_step001 showed aggressive entropy cuts HURT at ONE fixed small CNN (max_only
-8.7pp). This probe asks the sharper question: is that penalty capacity-dependent?

Output: results/frontier/data_step003_channel_capacity__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--data_dir", default="data/cifar-10-batches-py")
parser.add_argument("--epochs", type=int, default=20)          # T0 budget
parser.add_argument("--train_frac", type=float, default=0.5)   # T0 = 50% data
parser.add_argument("--depths", default="1,2,3,4")             # conv blocks to sweep
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"data_step003_channel_capacity__{SLOT}.json"
MODES = ["full", "max_only", "min_zero"]
CHS = [32, 64, 128, 256]


def load_cifar10(data_dir):
    d = ROOT / data_dir
    xs, ys = [], []
    for i in range(1, 6):
        b = pickle.load(open(d / f"data_batch_{i}", "rb"), encoding="bytes")
        xs.append(b[b"data"]); ys += b[b"labels"]
    tx = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    tb = pickle.load(open(d / "test_batch", "rb"), encoding="bytes")
    vx = tb[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return tx, np.array(ys, np.int64), vx, np.array(tb[b"labels"], np.int64)


def simplify(x, mode):
    """x:(N,3,H,W) in [0,1]. Zero weak channel(s) per user's SNR hypothesis."""
    if mode == "full":
        return x
    if mode == "max_only":
        am = x.argmax(axis=1, keepdims=True)
        is_max = np.zeros_like(x, dtype=bool)
        np.put_along_axis(is_max, am, True, axis=1)
        return np.where(is_max, x, 0.0).astype(np.float32)
    if mode == "min_zero":
        amin = x.argmin(axis=1, keepdims=True)
        is_min = np.zeros_like(x, dtype=bool)
        np.put_along_axis(is_min, amin, True, axis=1)
        return np.where(is_min, 0.0, x).astype(np.float32)
    raise ValueError(mode)


class DepthCNN(nn.Module):
    """`depth` conv blocks, channels 32->64->128->256; MaxPool while spatial>4 then GAP."""
    def __init__(self, depth, n_cls=10):
        super().__init__()
        layers, cin, sp = [], 3, 32
        for ch in CHS[:depth]:
            layers += [nn.Conv2d(cin, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU()]
            if sp > 4:
                layers += [nn.MaxPool2d(2)]; sp //= 2
            cin = ch
        layers += [nn.AdaptiveAvgPool2d(1)]
        self.c = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, n_cls)

    def forward(self, x):
        return self.fc(self.c(x).flatten(1))


@torch.no_grad()
def accuracy(model, x, y):
    model.eval(); correct = 0
    for i in range(0, len(x), 512):
        b = torch.from_numpy(x[i:i + 512]).to(DEVICE)
        correct += (model(b).argmax(1).cpu() == torch.from_numpy(y[i:i + 512])).sum().item()
    return correct / len(x)


def train_cell(mode, depth, tx, ty, vx, vy, t0):
    torch.manual_seed(0)
    model = DepthCNN(depth).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    curve, tc = [], time.time()
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(len(tx))
        for i in range(0, len(tx), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(np.ascontiguousarray(tx[idx])).to(DEVICE)
            yb = torch.from_numpy(ty[idx]).to(DEVICE)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        curve.append(round(accuracy(model, vx, vy), 4))
    dt = round(time.time() - tc, 1)
    print(f"  {mode:<9} d{depth} best={max(curve):.4f} final={curve[-1]:.4f} "
          f"{n_p:>8,}p {dt:>5.0f}s [{time.time()-t0:.0f}s]", flush=True)
    return {"mode": mode, "depth": depth, "params": n_p, "best_acc": max(curve),
            "final_acc": curve[-1], "train_s": dt, "curve": curve}


def main():
    depths = [int(d) for d in args.depths.split(",")]
    if args.smoke_test:
        x = np.random.rand(4, 3, 32, 32).astype(np.float32)
        for m in MODES:
            assert simplify(x, m).shape == x.shape
        for d in depths:
            out = DepthCNN(d)(torch.from_numpy(x))
            print(f"  depth{d} out={tuple(out.shape)} params={sum(p.numel() for p in DepthCNN(d).parameters()):,}")
        sys.exit(0)

    print(f"{'='*66}\ndata_step003 channel x capacity  device={DEVICE}  depths={depths}")
    t0 = time.time()
    rtx, ty, rvx, vy = load_cifar10(args.data_dir)
    n = int(len(rtx) * args.train_frac)
    sel = np.sort(np.random.RandomState(0).choice(len(rtx), n, replace=False))
    rtx, ty = rtx[sel], ty[sel]
    print(f"  train {rtx.shape}  val {rvx.shape}  [{time.time()-t0:.0f}s]")

    res = {"step": "data_step003", "device": str(DEVICE), "epochs": args.epochs,
           "train_frac": args.train_frac, "depths": depths, "cells": []}
    pre = {m: (simplify(rtx, m), simplify(rvx, m)) for m in MODES}
    for depth in depths:
        for mode in MODES:
            tx, vx = pre[mode]
            res["cells"].append(train_cell(mode, depth, tx, ty, vx, vy, t0))
            OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))

    print(f"\n  saturation (best_acc); Δ vs full at each depth:")
    print(f"  {'depth':>5} " + " ".join(f"{m:>10}" for m in MODES))
    for depth in depths:
        row = {c["mode"]: c for c in res["cells"] if c["depth"] == depth}
        full = row["full"]["best_acc"]
        cells = " ".join(f"{row[m]['best_acc']:.4f}({row[m]['best_acc']-full:+.3f})"
                         if m != "full" else f"{full:.4f}( base )" for m in MODES)
        print(f"  {depth:>5} {cells}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
