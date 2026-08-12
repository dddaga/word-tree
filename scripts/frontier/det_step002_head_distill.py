"""det_step002: can a compact SGNNET-style head REPRODUCE a detector's box head?

BRANCH B core experiment (DETECTION_TRAJECTORY B1). det_step001 showed FasterRCNN's box
head (TwoMLPHead 12544->1024->1024 + predictor) = 72% of params, VGG-FC-shaped. reid_step001
showed top-k sparsity COLLAPSES metric embeddings — but that was a triplet-EMBEDDING task,
NOT the readout role SGNNET is proven at. THIS tests the readout role directly:

  Teacher = the pretrained box_head+box_predictor (real learned weights). We harvest REAL
  on-distribution pooled ROI features (256x7x7=12544) by running the detector on imagenette
  images with a forward hook, then distil compact students to match teacher cls (KD/KL) +
  bbox deltas (smooth-L1). Metric: cls argmax AGREEMENT with teacher (overall + foreground-
  only, since background ROIs dominate) + bbox MSE + params vs the 14.4M teacher head.

Students: dense_reinit (teacher arch, capacity control) | lowrank_r256 (dense low-rank, no
sparsity — isolates top-k's effect) | sgn_r128k32, sgn_r64k16 (low-rank + hard top-k = SGNNET
analog). SGNNET matching teacher -> readout role transfers to detection (unlike reid
embeddings). Detector on CPU (avoids MPS roi_align gaps); student training on DEVICE.
Output: results/frontier/det_step002_head_distill__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--model", default="fasterrcnn_mobilenet_v3_large_fpn")
parser.add_argument("--img_dir", default="data/imagenette2-320/val")
parser.add_argument("--n_img", type=int, default=40)
parser.add_argument("--max_roi", type=int, default=12000)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--kd_temp", type=float, default=2.0)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"det_step002_head_distill__{SLOT}.json"
CONFIGS = ["dense_reinit", "lowrank_r256", "sgn_r128k32", "sgn_r64k16"]


class DenseHead(nn.Module):
    """Teacher architecture reinit — capacity/KD-feasibility control."""
    def __init__(self, d_in, n_cls, n_bb, hid=1024):
        super().__init__()
        self.fc6 = nn.Linear(d_in, hid); self.fc7 = nn.Linear(hid, hid)
        self.cls = nn.Linear(hid, n_cls); self.bb = nn.Linear(hid, n_bb)

    def forward(self, x):
        h = F.relu(self.fc7(F.relu(self.fc6(x))))
        return self.cls(h), self.bb(h)


class LowRankHead(nn.Module):
    """Dense low-rank, NO sparsity — isolates whether top-k (not rank) is what hurts."""
    def __init__(self, d_in, n_cls, n_bb, rank):
        super().__init__()
        self.down = nn.Linear(d_in, rank); self.up = nn.Linear(rank, rank)
        self.cls = nn.Linear(rank, n_cls); self.bb = nn.Linear(rank, n_bb)

    def forward(self, x):
        h = F.gelu(self.up(F.gelu(self.down(x))))
        return self.cls(h), self.bb(h)


class SGNNETHead(nn.Module):
    """SGNNET analog: low-rank down-proj + hard top-k sparse hidden -> cls + bbox arms."""
    def __init__(self, d_in, n_cls, n_bb, rank, k):
        super().__init__()
        self.down = nn.Linear(d_in, rank); self.up = nn.Linear(rank, rank)
        self.cls = nn.Linear(rank, n_cls); self.bb = nn.Linear(rank, n_bb); self.k = k

    def forward(self, x):
        h = F.gelu(self.down(x))
        kth = h.topk(self.k, dim=-1).values[..., -1:]
        h = F.gelu(self.up(h * (h >= kth)))
        return self.cls(h), self.bb(h)


def build(name, d_in, n_cls, n_bb):
    if name == "dense_reinit": return DenseHead(d_in, n_cls, n_bb)
    if name == "lowrank_r256": return LowRankHead(d_in, n_cls, n_bb, 256)
    if name == "sgn_r128k32":  return SGNNETHead(d_in, n_cls, n_bb, 128, 32)
    if name == "sgn_r64k16":   return SGNNETHead(d_in, n_cls, n_bb, 64, 16)
    raise ValueError(name)


def harvest(model):
    """Run detector on real images; hook captures pooled ROI feats -> teacher cls/bbox."""
    from torchvision.transforms.functional import to_tensor
    box_head, predictor = model.roi_heads.box_head, model.roi_heads.box_predictor
    grabbed = []
    hk = box_head.register_forward_hook(lambda m, i, o: grabbed.append(i[0].detach().flatten(1)))
    paths = sorted(Path(ROOT / args.img_dir).rglob("*.JPEG"))[:args.n_img]
    bufs, got = [], 0
    with torch.no_grad():
        for p in paths:
            grabbed.clear()
            model([to_tensor(Image.open(p).convert("RGB"))])
            if grabbed:
                bufs.append(grabbed[0].half()); got += grabbed[0].shape[0]
            if got >= args.max_roi:
                break
    hk.remove()
    X = torch.cat(bufs)[:args.max_roi]
    cls_t, bb_t = [], []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            c, b = predictor(box_head(X[i:i + 1024].float()))
            cls_t.append(c); bb_t.append(b)
    return X, torch.cat(cls_t), torch.cat(bb_t)


@torch.no_grad()
def evaluate(head, X, tgt_idx, bb_t):
    head.eval(); preds, se, n = [], 0.0, 0
    for i in range(0, len(X), 1024):
        pc, pb = head(X[i:i + 1024].float().to(DEVICE))
        preds.append(pc.argmax(1).cpu())
        se += ((pb.cpu() - bb_t[i:i + 1024]) ** 2).sum().item(); n += pb.numel()
    p = torch.cat(preds)
    fg = tgt_idx != 0
    fg_ag = (p[fg] == tgt_idx[fg]).float().mean().item() if fg.any() else 0.0
    return (p == tgt_idx).float().mean().item(), fg_ag, se / n


def train_head(name, X, cls_t, bb_t, d_in, n_cls, n_bb, t0):
    head = build(name, d_in, n_cls, n_bb).to(DEVICE)
    n_p = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    tgt_idx = cls_t.argmax(1); T = args.kd_temp; N = len(X)
    for ep in range(args.epochs):
        head.train(); perm = torch.randperm(N)
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            xb = X[idx].float().to(DEVICE)
            ct = cls_t[idx].to(DEVICE); bt = bb_t[idx].to(DEVICE)
            pc, pb = head(xb)
            kd = F.kl_div(F.log_softmax(pc / T, -1), F.softmax(ct / T, -1), reduction="batchmean") * T * T
            loss = kd + F.smooth_l1_loss(pb, bt)
            opt.zero_grad(); loss.backward(); opt.step()
    ag, fg, mse = evaluate(head, X, tgt_idx, bb_t)
    print(f"  {name:<13} agree={ag:.4f} fg_agree={fg:.4f} bbox_mse={mse:.4f} "
          f"{n_p:>9,}p [{time.time()-t0:.0f}s]", flush=True)
    return {"config": name, "params": n_p, "cls_agreement": round(ag, 4),
            "fg_agreement": round(fg, 4), "bbox_mse": round(mse, 5)}


def main():
    if args.smoke_test:
        for nm in CONFIGS:
            h = build(nm, 12544, 91, 364); c, b = h(torch.randn(4, 12544))
            print(f"  {nm:<13} cls={tuple(c.shape)} bb={tuple(b.shape)} {sum(p.numel() for p in h.parameters()):,}p")
        sys.exit(0)

    import torchvision.models.detection as det
    print(f"{'='*66}\ndet_step002 head distill  train_dev={DEVICE}  model={args.model}")
    t0 = time.time()
    model = getattr(det, args.model)(weights="DEFAULT").eval()   # detector stays on CPU
    d_in = 12544
    n_cls = model.roi_heads.box_predictor.cls_score.out_features
    n_bb = model.roi_heads.box_predictor.bbox_pred.out_features
    teacher_p = sum(p.numel() for p in model.roi_heads.box_head.parameters()) \
        + sum(p.numel() for p in model.roi_heads.box_predictor.parameters())
    X, cls_t, bb_t = harvest(model)
    fg = int((cls_t.argmax(1) != 0).sum())
    print(f"  harvested {X.shape} ROI feats  ({fg} foreground)  teacher_head={teacher_p:,}p [{time.time()-t0:.0f}s]")

    res = {"step": "det_step002", "model": args.model, "device": str(DEVICE),
           "n_roi": len(X), "n_fg": fg, "n_cls": n_cls, "n_bb": n_bb,
           "teacher_head_params": teacher_p, "configs": []}
    for nm in CONFIGS:
        r = train_head(nm, X, cls_t, bb_t, d_in, n_cls, n_bb, t0)
        r["pct_of_teacher"] = round(100 * r["params"] / teacher_p, 3)
        res["configs"].append(r)
        OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  teacher_head={teacher_p:,}p  ({fg}/{len(X)} fg ROIs)")
    print(f"  {'config':<13} {'agree':>7} {'fg_agree':>8} {'bbox_mse':>9} {'params':>10} {'%teach':>7}")
    for r in res["configs"]:
        print(f"  {r['config']:<13} {r['cls_agreement']:>7.4f} {r['fg_agreement']:>8.4f} "
              f"{r['bbox_mse']:>9.4f} {r['params']:>10,} {r['pct_of_teacher']:>6.3f}%")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
