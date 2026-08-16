"""seg_step002: encoder FLOP reduction — the only lever that reaches the drone budget.

seg_step001 CONFIRMED the head is free (16K params, ~4 MMAC) and the VGG b1-b4 encoder
is 4.56 GMAC at 128px, ~5x over the 1 GMAC/frame ceiling. So 100% of the cut has to come
from the encoder. Four ORTHOGONAL levers, one per arm (Compounding Rule — they all share
the encoder signal path, so no stacking until each is measured alone):

  E0  VGG b1-4 pretrained, frozen<block4   4.90 G   reference (= seg_step001 B1)
  E1  same shape, RANDOM init              4.90 G   scratch control for E3/E4/E5
  E2  drop block4 (head off pool3)         DEPTH
  E3  width 0.5x                           WIDTH
  E4  depthwise-separable convs            FACTORISATION
  E5  stride-2 stem                        RESOLUTION

Pretrained arms (E0/E2) compare against each other; scratch arms (E3/E4/E5) compare
against E1 ONLY — reading them against E0 would confound the lever with loss of the
ImageNet init.

All arms carry M_FPM + decorrelation lambda=0.05, i.e. seg_step001's best config.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.seg.seg_common import (C, EVENT_CLASSES, build_teacher, decor_penalty,
                                    list_images, load_image, rasterize)
from scripts.seg.seg_encoders import ARMS, SegNet

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="E0", help=" / ".join(ARMS))
parser.add_argument("--device", default="auto")
parser.add_argument("--teacher_device", default="cpu")   # fasterrcnn HANGS on MPS
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


def cache_targets(split: str, n: int):
    """One teacher pass, cached to disk and shared with seg_step001 (same filename)."""
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
    return x, y


def count_macs(model) -> int:
    """Conv MACs for one frame. GROUPS-AWARE — a depthwise conv does in_ch/groups MACs
    per output element, and E4 is entirely depthwise, so ignoring groups would report
    the separable arm as ~64x more expensive than it is."""
    macs, hooks = [0], []
    def hook(m, i, o):
        macs[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
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
            cl = model.class_logits(model(vx[i:i + BATCH].to(DEVICE)))
            t = vy[i:i + BATCH].to(DEVICE)
            bce += F.binary_cross_entropy_with_logits(cl, t, reduction="sum").item()
            outs.append(cl.cpu())
    cl = torch.cat(outs)
    aps = [average_precision(cl[:, c].flatten(), vy[:, c].flatten()) for c in range(C)]
    valid = [a for a in aps if a == a]
    return {"bce": bce / vy.numel(),
            "mAP": sum(valid) / len(valid) if valid else float("nan"),
            "ap_per_class": {EVENT_CLASSES[c]: round(aps[c], 4) for c in range(C)}}


def main():
    torch.manual_seed(SEED)
    model = SegNet(args.arm, k=args.k).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = count_macs(model)

    if args.smoke_test:
        out = model(torch.zeros(2, 3, args.res, args.res, device=DEVICE))
        ok = out.shape == (2, C * args.k, args.grid, args.grid)
        print(f"  {args.arm} out={tuple(out.shape)} trainable={n_p:,} "
              f"MACs={macs/1e9:.3f}G {'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    print(f"\n{'='*70}\nseg_step002 — {args.arm}  {args.res}px -> {args.grid}x{args.grid}, "
          f"C={C} x K={args.k}\n  device={DEVICE}  seed={SEED}\n{'='*70}", flush=True)
    tx, ty = cache_targets("train", args.n_train)
    vx, vy = cache_targets("val", args.n_val)
    print(f"  trainable={n_p:,}  MACs/frame={macs/1e9:.3f}G  (drone ceiling 1.0G)  "
          f"train={len(tx)} val={len(vx)}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    nb = (len(tx) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * nb,
                                                pct_start=0.1, anneal_strategy="cos")
    best, best_ep, t0 = 0.0, 0, time.time()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tx), generator=g)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            out = model(tx[b].to(DEVICE))
            loss = F.binary_cross_entropy_with_logits(model.class_logits(out), ty[b].to(DEVICE))
            loss = loss + args.decor_lambda * decor_penalty(out, args.k)
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
    res = {"step": "seg_step002", "arm": args.arm, "lever": ARMS[args.arm][1],
           "pretrained": ARMS[args.arm][3], "res": args.res, "grid": args.grid, "k": args.k,
           "decor_lambda": args.decor_lambda, "n_params": n_p, "macs": macs,
           "gmacs": round(macs / 1e9, 4), "best_mAP": round(best, 4), "best_ep": best_ep,
           "final": final, "elapsed_s": round(time.time() - t0, 1), "seed": SEED}
    p = ROOT / "results" / "seg" / f"seg_step002_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
