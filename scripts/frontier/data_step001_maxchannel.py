"""data_step001: does INPUT-SIDE entropy reduction (dominant-channel simplification)
keep accuracy while shrinking the entropy a model must overcome?

USER HYPOTHESIS: an RGB image carries redundant colour entropy. Collapse each pixel
toward its dominant channel -> simpler, lower-entropy input -> the model has less to
"undo" -> a cheap model should learn as well (efficiency lever, not just accuracy).

LITERATURE (grounds it): Feature Squeezing, Xu et al. NDSS 2018 (arXiv:1704.01155) —
reducing colour bit-depth barely hurts CIFAR/ImageNet accuracy => colour entropy IS
largely redundant. This probe tests the user's SPECIFIC dominant-channel schemes and
measures BOTH accuracy delta AND realised input entropy (bits), at a FIXED small CNN.

Modes (per-pixel, on [0,1] RGB):
  baseline      : raw RGB (control)
  max_only      : keep the argmax channel's value, zero the other two (scheme A)
  max_preserve  : keep argmax value, halve the other two (softer A)
  argmax_bucket : argmax channel -> 0.49 (125/255), others -> 0.78 (200/255)  (scheme B:
                  discards magnitude, encodes only WHICH channel dominated -> 3-colour map)
  posterize2    : 2-bit/channel quantization (feature-squeezing literature control)

Output: results/frontier/data_step001_maxchannel_{MODE}__{SLOT}.json
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
parser.add_argument("--mode", default="all",
                    choices=["all", "baseline", "max_only", "max_preserve", "argmax_bucket", "posterize2"])
parser.add_argument("--data_dir", default="data/cifar-10-batches-py")
parser.add_argument("--epochs", type=int, default=20)          # T0 budget
parser.add_argument("--train_frac", type=float, default=0.5)   # T0 = 50% data
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"data_step001_maxchannel_{args.mode}__{SLOT}.json"


def load_cifar10(data_dir):
    d = ROOT / data_dir
    xs, ys = [], []
    for i in range(1, 6):
        b = pickle.load(open(d / f"data_batch_{i}", "rb"), encoding="bytes")
        xs.append(b[b"data"]); ys += b[b"labels"]
    tx = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    ty = np.array(ys, dtype=np.int64)
    tb = pickle.load(open(d / "test_batch", "rb"), encoding="bytes")
    vx = tb[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    vy = np.array(tb[b"labels"], dtype=np.int64)
    return tx, ty, vx, vy


def simplify(x, mode):
    """x: (N,3,H,W) float in [0,1]. Return same-shape transformed input."""
    if mode == "baseline":
        return x
    am = x.argmax(axis=1, keepdims=True)                       # (N,1,H,W) dominant channel
    is_max = np.zeros_like(x, dtype=bool)
    np.put_along_axis(is_max, am, True, axis=1)
    if mode == "max_only":
        return np.where(is_max, x, 0.0).astype(np.float32)
    if mode == "max_preserve":
        return np.where(is_max, x, x * 0.5).astype(np.float32)
    if mode == "argmax_bucket":
        return np.where(is_max, 125 / 255.0, 200 / 255.0).astype(np.float32)
    if mode == "posterize2":
        L = 3.0                                                # 2-bit = 4 levels
        return (np.round(x * L) / L).astype(np.float32)
    raise ValueError(mode)


def mean_input_entropy(x, bins=32):
    """Mean per-image Shannon entropy (bits) of the pixel-value histogram -> quantifies
    how much input entropy each mode leaves for the model to overcome."""
    ent = []
    for img in x[:2000]:                                       # sample for speed
        h, _ = np.histogram(img, bins=bins, range=(0, 1), density=False)
        p = h / h.sum(); p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return round(float(np.mean(ent)), 4)


class SmallCNN(nn.Module):
    """Fixed tiny CNN — held constant across modes so accuracy delta isolates the DATA."""
    def __init__(self, n_cls=10):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        return self.fc(self.c(x).flatten(1))


@torch.no_grad()
def accuracy(model, x, y):
    model.eval(); correct = 0
    for i in range(0, len(x), 512):
        b = torch.from_numpy(x[i:i + 512]).to(DEVICE)
        correct += (model(b).argmax(1).cpu() == torch.from_numpy(y[i:i + 512])).sum().item()
    return correct / len(x)


def run_mode(mode, raw_tx, ty, raw_vx, vy, t0):
    tx = simplify(raw_tx, mode); vx = simplify(raw_vx, mode)
    ent = mean_input_entropy(tx)
    torch.manual_seed(0)
    model = SmallCNN().to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    curve = []
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(len(tx))
        for i in range(0, len(tx), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(tx[idx]).to(DEVICE); yb = torch.from_numpy(ty[idx]).to(DEVICE)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = accuracy(model, vx, vy); curve.append(round(acc, 4))
        print(f"  {mode:<14} ep{ep+1}/{args.epochs} val_acc={acc:.4f} ent={ent}b [{time.time()-t0:.0f}s]", flush=True)
    return {"mode": mode, "input_entropy_bits": ent, "cnn_params": n_p,
            "final_acc": curve[-1], "best_acc": max(curve), "acc_curve": curve}


def main():
    if args.smoke_test:
        x = np.random.rand(4, 3, 32, 32).astype(np.float32)
        for m in ["baseline", "max_only", "max_preserve", "argmax_bucket", "posterize2"]:
            o = simplify(x, m); assert o.shape == x.shape
            print(f"  {m:<14} ok  entropy={mean_input_entropy(o):.3f}")
        sys.exit(0)

    modes = ["baseline", "max_only", "max_preserve", "argmax_bucket", "posterize2"] \
        if args.mode == "all" else [args.mode]
    print(f"{'='*66}\ndata_step001 modes={modes}  device={DEVICE}")
    t0 = time.time()
    tx, ty, vx, vy = load_cifar10(args.data_dir)
    n = int(len(tx) * args.train_frac)
    sel = np.sort(np.random.RandomState(0).choice(len(tx), n, replace=False))
    tx, ty = tx[sel], ty[sel]
    print(f"  train {tx.shape}  val {vx.shape}  [{time.time()-t0:.0f}s]")

    res = {"step": "data_step001", "device": str(DEVICE), "train_frac": args.train_frac,
           "epochs": args.epochs, "modes": {}}
    for mode in modes:
        r = run_mode(mode, tx, ty, vx, vy, t0)
        res["modes"][mode] = r
        out = ROOT / "results" / "frontier" / f"data_step001_maxchannel_{args.mode}__{SLOT}.json"
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=2))
    print(f"\n  {'mode':<14} {'entropy':>8} {'acc':>8}  Δvs_baseline")
    base = res["modes"].get("baseline", {}).get("best_acc")
    for m, r in res["modes"].items():
        d = f"{r['best_acc']-base:+.4f}" if base is not None else "n/a"
        print(f"  {m:<14} {r['input_entropy_bits']:>8.3f} {r['best_acc']:>8.4f}  {d}")
    print(f"-> data_step001_maxchannel_{args.mode}__{SLOT}.json")


if __name__ == "__main__":
    main()
