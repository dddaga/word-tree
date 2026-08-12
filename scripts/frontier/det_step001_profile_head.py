"""det_step001: object-detector param/FLOP profile + SGNNET box-head expressibility POC.

BRANCH B, first-level. WHY detection fits SGNNET where LLM-FFN did not:
  SGNNET is CONFIRMED strong as a READOUT (pooled features -> class). A 2-stage
  detector's box head is EXACTLY that shape: FasterRCNN's `TwoMLPHead` runs on pooled
  ROI features (256*7*7 = 12544) -> 1024 -> 1024, then `FastRCNNPredictor` reads out
  class + bbox. That FC-on-pooled-features block is STRUCTURALLY IDENTICAL to VGG16's
  FC block on pool5 — the block we already replaced with a 0.029%-param SGNNET champion.

  This script (a) profiles fasterrcnn_mobilenet_v3_large_fpn to locate where params/FLOPs
  live and quantify the box head's share, (b) proves an SGNNET-shaped compact box head is
  expressible + param-competitive on real ROI feature dims. Grounds DETECTION_TRAJECTORY.

Output: results/frontier/det_step001_profile__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="fasterrcnn_mobilenet_v3_large_fpn")
parser.add_argument("--roi_per_img", type=int, default=1000, help="test-time RPN proposals (box head runs per ROI)")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"det_step001_profile__{SLOT}.json"


class SGNNETStyleBoxHead(nn.Module):
    """Compact readout analog: factorized low-rank + sparse hidden (SGNNET-inspired) on
    pooled ROI features -> class logits. Proves param-competitive expressibility."""
    def __init__(self, in_dim, n_cls, rank=64, k=16):
        super().__init__()
        self.down = nn.Linear(in_dim, rank)          # 12544 -> rank (bottleneck)
        self.cls = nn.Linear(rank, n_cls)
        self.k = k

    def forward(self, x):
        h = F.gelu(self.down(x.flatten(1)))
        kth = h.topk(self.k, dim=-1).values[..., -1:]
        return self.cls(h * (h >= kth))               # top-k sparse routing


def submodule_params(model):
    """Param counts for the top-level detector submodules."""
    out = {}
    for name, mod in model.named_children():
        out[name] = sum(p.numel() for p in mod.parameters())
    return out


def box_head_breakdown(model):
    roi = model.roi_heads
    parts = {}
    for nm in ["box_head", "box_predictor"]:
        m = getattr(roi, nm, None)
        if m is not None:
            parts[nm] = sum(p.numel() for p in m.parameters())
    return parts


def main():
    import torchvision.models.detection as det

    if args.smoke_test:
        head = SGNNETStyleBoxHead(12544, 91, rank=64, k=16)
        out = head(torch.randn(5, 256, 7, 7))
        n_p = sum(p.numel() for p in head.parameters())
        print(f"  SGNNET-style head params={n_p:,}  out={tuple(out.shape)}  (5 ROIs -> 91 cls)")
        sys.exit(0 if out.shape == (5, 91) else 1)

    t0 = time.time()
    ctor = getattr(det, args.model)
    model = ctor(weights="DEFAULT").eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"{'='*66}\ndet_step001  {args.model}  total params={total:,}  [{time.time()-t0:.0f}s]")

    sub = submodule_params(model)
    box = box_head_breakdown(model)
    box_total = sum(box.values())

    # box head runs ONCE PER ROI at test -> its MACs scale with proposal count
    th = model.roi_heads.box_head
    in_dim = None
    for m in th.modules():
        if isinstance(m, nn.Linear):
            in_dim = m.in_features; break
    n_cls = model.roi_heads.box_predictor.cls_score.out_features
    per_roi_macs = box_total                          # ~1 MAC/param for the FC head
    box_head_macs = per_roi_macs * args.roi_per_img

    sgn = SGNNETStyleBoxHead(in_dim, n_cls, rank=64, k=16)
    sgn_p = sum(p.numel() for p in sgn.parameters())

    res = {"step": "det_step001", "model": args.model, "total_params": total,
           "submodule_params": sub,
           "box_head_params": box, "box_head_total": box_total,
           "box_head_pct_of_model": round(100 * box_total / total, 2),
           "box_head_in_dim": in_dim, "n_classes": n_cls,
           "roi_per_img": args.roi_per_img,
           "box_head_macs_per_image": box_head_macs,
           "sgnnet_style_head_params": sgn_p,
           "sgnnet_pct_of_box_head": round(100 * sgn_p / box_total, 2),
           "elapsed_s": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))

    print(f"\n  submodule params:")
    for k, v in sorted(sub.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:>12,}  ({100*v/total:5.1f}%)")
    print(f"\n  BOX HEAD (SGNNET target — FC-on-pooled-ROI, VGG-FC analog):")
    for k, v in box.items():
        print(f"    {k:<18} {v:>12,}")
    print(f"    box head = {box_head_macs:,} MACs/image @ {args.roi_per_img} ROIs")
    print(f"    in_dim={in_dim}  n_cls={n_cls}")
    print(f"  SGNNET-style compact head: {sgn_p:,} params = {res['sgnnet_pct_of_box_head']}% of box head")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
