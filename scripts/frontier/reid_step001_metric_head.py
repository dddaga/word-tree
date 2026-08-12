"""reid_step001: can an SGNNET-style micro-head produce DISCRIMINATIVE metric embeddings?

BRANCH B redirect (drone surveillance), load-bearing probe.
  Modern efficient detectors have NO fat FC head (det_step001 + YOLO/DETR research:
  any_Linear=False, backbone 60-80%). So SGNNET's proven readout advantage does NOT
  apply to the detector. It DOES map onto the per-track downstream heads: Re-ID
  embedding + action classifier, which run per-person-per-frame.

  THE UNKNOWN: the SGNNET champion only ever output CLASS LOGITS. Re-ID needs a
  discriminative EMBEDDING under a metric (triplet) loss. Never tested at the champion's
  compact budget. This probe: on frozen VGG pool5 features (25088-d, same space as the
  champion), train an embedding head under batch-hard triplet loss and measure retrieval
  Rank-1 / mAP. Compare a dense Linear head vs SGNNET-style low-rank+top-k heads.
  If SGNNET matches dense -> embedding readout viable -> drone reID bet stays alive.
  If top-k sparsity collapses discriminability -> only the action-classifier slot survives.

  cifar-100 features used as a 100-IDENTITY proxy (100 classes ~ 100 people).

Output: results/frontier/reid_step001_metric_head__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--store", default="data/store_cifar100.h5")
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--epochs", type=int, default=20)          # T0 budget
parser.add_argument("--iters_per_epoch", type=int, default=200)
parser.add_argument("--P", type=int, default=16, help="classes per batch")
parser.add_argument("--K", type=int, default=4, help="samples per class")
parser.add_argument("--margin", type=float, default=0.3)
parser.add_argument("--train_frac", type=float, default=0.5, help="T0 fraction (0.5)")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"reid_step001_metric_head__{SLOT}.json"


class DenseHead(nn.Module):
    def __init__(self, d_in, d_emb):
        super().__init__()
        self.proj = nn.Linear(d_in, d_emb)
        self.bn = nn.BatchNorm1d(d_emb)              # BN-neck (strong-baseline reID)

    def forward(self, x):
        return self.bn(self.proj(x))


class SGNNETHead(nn.Module):
    """SGNNET analog: low-rank down-proj + top-k sparse hidden -> embedding + BN-neck."""
    def __init__(self, d_in, d_emb, rank, k):
        super().__init__()
        self.down = nn.Linear(d_in, rank)
        self.up = nn.Linear(rank, d_emb)
        self.bn = nn.BatchNorm1d(d_emb)
        self.k = k

    def forward(self, x):
        h = F.gelu(self.down(x))
        kth = h.topk(self.k, dim=-1).values[..., -1:]
        h = h * (h >= kth)                            # hard top-k sparse routing
        return self.bn(self.up(h))


def build_head(name, d_in, d_emb):
    if name == "dense":      return DenseHead(d_in, d_emb)
    if name == "sgn_r64k16": return SGNNETHead(d_in, d_emb, 64, 16)
    if name == "sgn_r32k8":  return SGNNETHead(d_in, d_emb, 32, 8)
    raise ValueError(name)


def batch_hard_triplet(emb, labels, margin):
    """Standard reID batch-hard: hardest positive + hardest negative per anchor."""
    emb = F.normalize(emb, dim=1)
    dist = torch.cdist(emb, emb)                      # (B,B) euclidean on unit sphere
    same = labels[:, None] == labels[None, :]
    diff = ~same
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos = dist.masked_fill(~(same & ~eye), -1.0).max(1).values     # hardest (farthest) pos
    neg = dist.masked_fill(~diff, float("inf")).min(1).values      # hardest (nearest) neg
    return F.relu(pos - neg + margin).mean()


@torch.no_grad()
def evaluate(head, val_x, val_y):
    """Rank-1 + mAP retrieval, val as query+gallery, self excluded."""
    head.eval()
    embs = []
    for i in range(0, len(val_x), 512):
        b = torch.from_numpy(val_x[i:i + 512]).to(DEVICE).float()
        embs.append(F.normalize(head(b), dim=1).cpu())
    E = torch.cat(embs)
    y = torch.from_numpy(val_y)
    sim = E @ E.t()
    sim.fill_diagonal_(-2.0)
    order = sim.argsort(dim=1, descending=True)
    ranked = y[order]
    match = (ranked == y[:, None])
    rank1 = match[:, 0].float().mean().item()
    # mAP: average precision per query over all gallery
    cum = match.float().cumsum(1)
    prec = cum / torch.arange(1, match.size(1) + 1).float()
    ap = (prec * match.float()).sum(1) / match.float().sum(1).clamp(min=1)
    return rank1, ap.mean().item()


def train_head(name, tr_x, tr_y, val_x, val_y, cls_idx, classes, base_r1):
    head = build_head(name, tr_x.shape[1], args.emb_dim).to(DEVICE)
    n_p = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(args.epochs):
        head.train()
        for _ in range(args.iters_per_epoch):
            pick = np.random.choice(classes, args.P, replace=False)
            idx = np.concatenate([np.random.choice(cls_idx[c], args.K, replace=len(cls_idx[c]) < args.K) for c in pick])
            idx.sort()
            xb = torch.from_numpy(tr_x[idx]).to(DEVICE).float()
            yb = torch.from_numpy(tr_y[idx]).to(DEVICE)
            loss = batch_hard_triplet(head(xb), yb, args.margin)
            opt.zero_grad(); loss.backward(); opt.step()
        r1, mAP = evaluate(head, val_x, val_y)
        print(f"    {name:<12} ep{ep+1}/{args.epochs} R1={r1:.4f} mAP={mAP:.4f} loss={loss.item():.4f} [{time.time()-t0:.0f}s]", flush=True)
    r1, mAP = evaluate(head, val_x, val_y)
    return {"head": name, "n_params": n_p, "rank1": round(r1, 4), "mAP": round(mAP, 4),
            "vs_dense_r1": round(r1 - base_r1, 4) if base_r1 else None,
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    if args.smoke_test:
        for nm in ["dense", "sgn_r64k16", "sgn_r32k8"]:
            h = build_head(nm, 25088, 128)
            out = h(torch.randn(8, 25088))
            print(f"  {nm:<12} params={sum(p.numel() for p in h.parameters()):,} out={tuple(out.shape)}")
        sys.exit(0)

    print(f"{'='*66}\nreid_step001 metric-embedding head  device={DEVICE}  store={args.store}")
    with h5py.File(ROOT / args.store, "r") as hf:
        tr_lab = hf["train"]["labels"][:]
        n_keep = int(len(tr_lab) * args.train_frac)
        sel = np.sort(np.random.RandomState(0).choice(len(tr_lab), n_keep, replace=False))
        tr_x = hf["train"]["features"][:][sel]        # (n_keep, 25088) f32
        tr_y = tr_lab[sel].astype(np.int64)
        val_x = hf["val"]["features"][:]
        val_y = hf["val"]["labels"][:].astype(np.int64)
    classes = np.unique(tr_y)
    cls_idx = {c: np.where(tr_y == c)[0] for c in classes}
    print(f"  train {tr_x.shape} ({len(classes)} ids)  val {val_x.shape}  d_in={tr_x.shape[1]}")

    res = {"step": "reid_step001", "device": str(DEVICE), "store": args.store,
           "d_in": int(tr_x.shape[1]), "n_ids": int(len(classes)),
           "emb_dim": args.emb_dim, "train_frac": args.train_frac, "heads": {}}
    base_r1 = 0.0
    for nm in ["dense", "sgn_r64k16", "sgn_r32k8"]:
        r = train_head(nm, tr_x, tr_y, val_x, val_y, cls_idx, classes, base_r1)
        if nm == "dense":
            base_r1 = r["rank1"]; r["vs_dense_r1"] = 0.0
        res["heads"][nm] = r
        OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  RETRIEVAL (cifar100 as 100-identity proxy):")
    for nm, r in res["heads"].items():
        print(f"  {nm:<12} R1={r['rank1']:.4f} mAP={r['mAP']:.4f}  {r['n_params']:>9,}p  ΔR1={r['vs_dense_r1']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
