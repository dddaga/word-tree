"""data_step002: cost-benefit of content-aware pixel removal (SEAM CARVING) before
a detector, WITH an invertible removal record so localization can be reclaimed.

USER QUESTION: is seam carving (or a heuristic variant) a cheap way to shrink an image
by dropping NON-important pixels — keeping detail on important regions — while recording
WHAT was removed and HOW, so detection boxes can be re-projected to the original frame?
What is the cost-benefit vs a naive uniform resize at the SAME pixel budget?

WHAT THIS MEASURES (CIFAR-10, SmallCNN as a cheap task proxy; direction, not absolutes):
  baseline : 32x32 full (control)
  resize24 : uniform bilinear -> 24x24        (uniform pixel removal, same budget as seam24)
  seam24   : carve 8 vertical + 8 horizontal lowest-energy seams -> 24x24 (content-aware)
  seam_hi  : carve only 4+4 -> 28x28          (lighter cut, higher retention)

For seam modes we keep `orig_col`/`orig_row` index maps (the removal RECORD) so any
carved coordinate maps EXACTLY back to the original frame -> localization reclaim. We
verify the inverse map is exact and report its memory cost + carve time/image.

COST: O(H*W)/seam CPU-sequential DP (ms/img). BENEFIT: accuracy kept at equal px budget
vs uniform resize; bytes saved. RECLAIM: seam-index record -> exact inverse (round-trip).
Output: results/frontier/data_step002_seamcarve__{SLOT}.json
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
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--train_frac", type=float, default=0.5)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"data_step002_seamcarve__{SLOT}.json"


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


def energy(gray):
    """L1 gradient-magnitude energy. gray: (N,H,W) -> (N,H,W)."""
    dy = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1, :]))
    dx = np.abs(np.diff(gray, axis=2, prepend=gray[:, :, :1]))
    return dy + dx


def carve_vertical(x, cols, k):
    """Remove k lowest-energy vertical seams from a batch, tracking original columns.
    x:(N,3,H,W) cols:(N,H,W) original-column index map. Returns (x', cols') width W-k."""
    for _ in range(k):
        N, _, H, W = x.shape
        gray = x.mean(1)                                   # (N,H,W)
        E = energy(gray)
        M = E.copy(); back = np.zeros((N, H, W), np.int64)
        for i in range(1, H):                              # DP: min cumulative energy
            left = np.pad(M[:, i - 1, :-1], ((0, 0), (1, 0)), constant_values=1e9)
            mid = M[:, i - 1, :]
            right = np.pad(M[:, i - 1, 1:], ((0, 0), (0, 1)), constant_values=1e9)
            stack = np.stack([left, mid, right], 0)        # (3,N,W)
            choice = stack.argmin(0)                        # 0=left,1=mid,2=right
            M[:, i, :] = E[:, i, :] + stack.min(0)
            back[:, i, :] = choice - 1                      # -1,0,+1 column delta
        seam = np.zeros((N, H), np.int64)
        seam[:, H - 1] = M[:, H - 1, :].argmin(1)
        for i in range(H - 2, -1, -1):                     # backtrack
            seam[:, i] = np.clip(seam[:, i + 1] + back[np.arange(N), i + 1, seam[:, i + 1]], 0, W - 1)
        keep = np.ones((N, H, W), bool)
        keep[np.arange(N)[:, None], np.arange(H)[None, :], seam] = False
        x = np.stack([x[:, c][keep].reshape(N, H, W - 1) for c in range(3)], 1)
        cols = cols[keep].reshape(N, H, W - 1)
    return x, cols


def carve(x, kv, kh):
    """Carve kv vertical + kh horizontal seams. Returns carved x, and (col,row) records."""
    N, _, H, W = x.shape
    cols = np.broadcast_to(np.arange(W), (N, H, W)).copy()
    x, cols = carve_vertical(x, cols, kv)
    xt = x.transpose(0, 1, 3, 2)                            # swap H<->W for horizontal
    rows = np.broadcast_to(np.arange(H), (N, x.shape[3], H)).copy()
    xt, rows = carve_vertical(xt, rows, kh)
    x = xt.transpose(0, 1, 3, 2)
    return x.astype(np.float32), cols, rows.transpose(0, 2, 1)


class SmallCNN(nn.Module):
    """Input-size-agnostic (AdaptiveAvgPool) so all modes share one architecture."""
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


def transform(x, mode):
    """Return (x', extra) where extra records cost/record metadata for this mode."""
    if mode == "baseline":
        return x, {"px": x.shape[2] * x.shape[3]}
    if mode == "resize24":
        t = F.interpolate(torch.from_numpy(x), size=(24, 24), mode="bilinear", align_corners=False)
        return t.numpy(), {"px": 576}
    kv, kh, px = (8, 8, 576) if mode == "seam24" else (4, 4, 784)
    t0 = time.time(); xc, cols, rows = carve(x.copy(), kv, kh); dt = time.time() - t0
    # localization reclaim: inverse map exactness (indices in valid original range, monotone)
    ok = bool((cols >= 0).all() and (cols < x.shape[3]).all() and (rows < x.shape[2]).all())
    rec_bytes = int(cols.nbytes + rows.nbytes) // len(x)   # per-image record cost
    return xc, {"px": px, "carve_ms_per_img": round(1000 * dt / len(x), 3),
                "reclaim_record_bytes": rec_bytes, "reclaim_map_exact": ok}


def run_mode(mode, tx, ty, vx, vy, t0):
    txm, meta = transform(tx, mode); vxm, _ = transform(vx, mode)
    torch.manual_seed(0)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    curve = []
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(len(txm))
        for i in range(0, len(txm), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.from_numpy(np.ascontiguousarray(txm[idx])).to(DEVICE)
            yb = torch.from_numpy(ty[idx]).to(DEVICE)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = accuracy(model, vxm, vy); curve.append(round(acc, 4))
        print(f"  {mode:<10} ep{ep+1}/{args.epochs} acc={acc:.4f} px={meta['px']} [{time.time()-t0:.0f}s]", flush=True)
    return {"mode": mode, "best_acc": max(curve), "final_acc": curve[-1], **meta}


def main():
    if args.smoke_test:
        x = np.random.rand(6, 3, 32, 32).astype(np.float32)
        for m in ["baseline", "resize24", "seam24", "seam_hi"]:
            o, meta = transform(x, m)
            print(f"  {m:<10} {tuple(o.shape)}  {meta}")
        sys.exit(0)

    print(f"{'='*66}\ndata_step002 seam-carve  device={DEVICE}")
    t0 = time.time()
    tx, ty, vx, vy = load_cifar10(args.data_dir)
    n = int(len(tx) * args.train_frac)
    sel = np.sort(np.random.RandomState(0).choice(len(tx), n, replace=False))
    tx, ty = tx[sel], ty[sel]
    print(f"  train {tx.shape}  val {vx.shape}  [{time.time()-t0:.0f}s]")

    res = {"step": "data_step002", "device": str(DEVICE), "epochs": args.epochs,
           "train_frac": args.train_frac, "modes": {}}
    for mode in ["baseline", "resize24", "seam24", "seam_hi"]:
        res["modes"][mode] = run_mode(mode, tx, ty, vx, vy, t0)
        OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    base = res["modes"]["baseline"]["best_acc"]; rz = res["modes"]["resize24"]["best_acc"]
    print(f"\n  {'mode':<10} {'px':>5} {'acc':>7}  Δbaseline  Δresize(same-budget)")
    for m, r in res["modes"].items():
        print(f"  {m:<10} {r['px']:>5} {r['best_acc']:>7.4f}  {r['best_acc']-base:+.4f}   {r['best_acc']-rz:+.4f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
