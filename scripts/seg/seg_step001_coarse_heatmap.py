"""seg_step001: coarse-grid event heat-map baseline, 128px in / 16x16 out (2026-08-14).

Establishes the ACCURACY CEILING and the MAC BASELINE for the drone reframe before
any efficiency work touches the encoder. Deliberately uses the expensive encoder
(VGG16 b1-b4) so seg_step002's encoder levers have an honest reference to beat.

Known going in (MAC arithmetic, not a guess): VGG b1-b4 @128 costs ~4.6 GMAC, i.e.
~15x a MobileNetV2-class budget, so this baseline CANNOT meet the 1 GMAC/frame drone
ceiling at any stride. That is the point of running it: the head is ~free (16K params,
4 MMAC), so 100% of the cost is encoder and step002 knows exactly where to cut.

Supervision: cached fasterrcnn_mobilenet_v3_large_fpn rasterised onto the grid. No
pixel masks, no annotation, no download — imagenette2-320 is already on disk.

Arms (--arm):
  B0  VGG b1-4 + M_FPM + 1x1 head, K=2, no decorrelation      (baseline)
  B1  B0 + anti-Hebbian decorrelation on the K-channel groups (explainability arm)
  B2  B0 without M_FPM — head straight off block4             (is the pyramid earning it?)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.seg.seg_common import (C, EVENT_CLASSES, CoarseSegNet, build_teacher,
                                    decor_penalty, list_images, load_image, rasterize)

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="B0", help="B0 / B1 / B2")
parser.add_argument("--device", default="auto")
# The detector runs ONCE to build the cache. On MPS fasterrcnn stalls indefinitely
# (26 min elapsed for one batch at 0% CPU), so the teacher gets its own device and
# defaults to cpu — at 128px the detector is cheap there.
parser.add_argument("--teacher_device", default="cpu")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--res", type=int, default=128)
parser.add_argument("--grid", type=int, default=16)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_val", type=int, default=500)
parser.add_argument("--decor_lambda", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="t0")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED, BATCH = args.seed, 32
SLOT = os.environ.get("SGN_SLOT", "local")
USE_MFPM = args.arm != "B2"
LAM = args.decor_lambda if args.arm == "B1" else 0.0


def cache_targets(split: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One teacher pass over the images, cached to disk — the teacher never runs again."""
    cp = ROOT / "data" / f"seg_targets_{split}_n{n}_r{args.res}_g{args.grid}.pt"
    if cp.exists():
        d = torch.load(cp)
        return d["x"], d["y"]
    paths = list_images(split, n)
    tdev = torch.device(args.teacher_device)
    teacher = build_teacher(tdev)
    xs, ys = [], []
    t0 = time.time()
    for i in range(0, len(paths), BATCH):
        bx = torch.stack([load_image(p, args.res) for p in paths[i:i + BATCH]])
        with torch.no_grad():
            dets = teacher([b for b in bx.to(tdev)])
        for d in dets:
            ys.append(rasterize(d["boxes"].cpu(), d["labels"].cpu(), d["scores"].cpu(),
                                args.res, args.grid))
        xs.append(bx)
        if i % (BATCH * 10) == 0:
            print(f"  teacher {split} {i}/{len(paths)} [{time.time()-t0:.0f}s]", flush=True)
    x, y = torch.cat(xs), torch.stack(ys)
    torch.save({"x": x, "y": y}, cp)
    print(f"  cached -> {cp}  x={tuple(x.shape)} y={tuple(y.shape)} "
          f"pos_frac={(y > 0.3).float().mean():.4f}", flush=True)
    return x, y


def count_macs(model) -> int:
    """Conv MACs for one 1x{res}x{res} frame — the number the drone budget is set in."""
    macs, hooks = [0], []
    def hook(m, i, o):
        macs[0] += o.numel() * m.in_channels * m.kernel_size[0] * m.kernel_size[1]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, args.res, args.res, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    return macs[0]


def average_precision(score: torch.Tensor, target: torch.Tensor) -> float:
    """Cell-level AP over a flattened grid; target binarised at the teacher threshold."""
    tgt = (target > 0.3).float()
    if tgt.sum() == 0:
        return float("nan")
    order = score.argsort(descending=True)
    hits = tgt[order]
    prec = hits.cumsum(0) / torch.arange(1, len(hits) + 1, device=hits.device)
    return float((prec * hits).sum() / hits.sum())


def evaluate(model, vx, vy) -> dict:
    model.eval()
    outs, bce = [], 0.0
    with torch.no_grad():
        for i in range(0, len(vx), BATCH):
            o = model(vx[i:i + BATCH].to(DEVICE))
            cl = model.class_logits(o)
            t = vy[i:i + BATCH].to(DEVICE)
            bce += F.binary_cross_entropy_with_logits(cl, t, reduction="sum").item()
            outs.append(cl.cpu())
    cl = torch.cat(outs)
    aps = [average_precision(cl[:, c].flatten(), vy[:, c].flatten()) for c in range(C)]
    valid = [a for a in aps if a == a]
    return {"bce": bce / vy.numel(),
            "mAP": sum(valid) / len(valid) if valid else float("nan"),
            "ap_per_class": {EVENT_CLASSES[c]: round(aps[c], 4) for c in range(C)}}


def build():
    torch.manual_seed(SEED)
    return CoarseSegNet(k=args.k, mfpm=USE_MFPM).to(DEVICE)


def main():
    if args.smoke_test:
        m = build()
        out = m(torch.zeros(2, 3, args.res, args.res, device=DEVICE))
        cl = m.class_logits(out)
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        macs = count_macs(m)
        ok = out.shape == (2, C * args.k, args.grid, args.grid) and cl.shape[1] == C
        print(f"  {args.arm} out={tuple(out.shape)} cls={tuple(cl.shape)} "
              f"trainable={n_p:,} MACs={macs/1e9:.2f}G {'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    print(f"\n{'='*70}\nseg_step001 — {args.arm}  {args.res}px -> {args.grid}x{args.grid} grid, "
          f"C={C} x K={args.k}\n  device={DEVICE}  decor_lambda={LAM}  mfpm={USE_MFPM}\n{'='*70}",
          flush=True)
    tx, ty = cache_targets("train", args.n_train)
    vx, vy = cache_targets("val", args.n_val)

    model = build()
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = count_macs(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    nb = (len(tx) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * nb,
                                                pct_start=0.1, anneal_strategy="cos")
    print(f"  trainable={n_p:,}  MACs/frame={macs/1e9:.3f}G  "
          f"(drone ceiling 1.0G)  train={len(tx)} val={len(vx)}", flush=True)

    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tx), generator=g)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            bx, by = tx[b].to(DEVICE), ty[b].to(DEVICE)
            out = model(bx)
            loss = F.binary_cross_entropy_with_logits(model.class_logits(out), by)
            if LAM:
                loss = loss + LAM * decor_penalty(out, args.k)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = evaluate(model, vx, vy)
        if m["mAP"] > best:
            best, best_ep = m["mAP"], ep + 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  e{ep+1:3d}/{args.epochs}  mAP={m['mAP']:.4f}  bce={m['bce']:.4f}  "
                  f"best={best:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    final = evaluate(model, vx, vy)
    print(f"  DONE: best_mAP={best:.4f} @ep{best_ep}  {final['ap_per_class']}  "
          f"{time.time()-t0:.0f}s")
    res = {"step": "seg_step001", "arm": args.arm, "res": args.res, "grid": args.grid,
           "k": args.k, "mfpm": USE_MFPM, "decor_lambda": LAM, "n_params": n_p,
           "macs": macs, "gmacs": round(macs / 1e9, 4), "best_mAP": round(best, 4),
           "best_ep": best_ep, "final": final, "n_train": len(tx), "n_val": len(vx),
           "elapsed_s": round(time.time() - t0, 1), "seed": SEED}
    p = ROOT / "results" / "seg" / f"seg_step001_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
