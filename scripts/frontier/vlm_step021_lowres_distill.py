"""vlm_step021: can a tower be DISTILLED to run at a smaller pixel grid?

WHAT step020 FOUND, AND WHY IT IS NOT THE END OF THE LEVER. Cutting input resolution on the STOCK
d12 tower collapses top-1: 0.704 @512 -> 0.456 @384 -> 0.160 @256 -> 0.094 @192 (chance is 0.100),
while the tower gets 1.8x / 2.9x / 3.7x faster. Read alone that kills resolution. But the SAME
shape already appeared for depth, twice: naive d6 scored 0.120 and naive d3 0.100 -- exactly
chance -- and distillation lifted them to 0.560 and 0.460 (step004, step008). The trajectory's own
precedent is that a NAIVE structural cut to this tower is uninformative; the informative arm is the
distilled one. step020 was a rejection filter on naive resolution; this is the real question.

THE OBJECTIVE, AND THE ONE NON-OBVIOUS PIECE. Teacher = the stock d12 tower at 512, which emits 64
post-connector tokens on an 8x8 grid. A student at 256 emits 16 tokens on a 4x4 grid, so the
feature MSE used for every previous distillation in this line is SHAPE-INCOMPATIBLE. Resolved by
average-pooling the teacher's grid down to the student's before the loss: 8x8 -> 2x2 pool -> 4x4.
That is the honest pairing -- each student token is asked to reproduce the mean of the four teacher
tokens covering the same image region -- and it is exactly what the connector's own 4x4 pooling
already does one level up. Loss stays relative MSE, mse/mean(t^2), so numbers remain comparable to
step004/007/008.

WHAT MOVES AND WHAT DOES NOT. The student is a full-DEPTH (d12) trainable copy of the tower, so
depth is held fixed and resolution is the only structural variable. Patch embeddings,
post_layernorm, connector and text model stay frozen exactly as in every prior arm. Position
embeddings are interpolated by SigLIP itself at the smaller grid; the student is free to adapt to
them, which is the whole hypothesis -- step020's failure may be nothing more than a stock tower
being handed interpolated position embeddings it was never trained on.

WHY THIS MATTERS FOR THE DRONE. The tower is 82% of end-to-end latency at d12 and 71% at d6
(step006). Depth already bought 2.0x. If resolution can be distilled it composes on top for
~3x more, and unlike depth it also shrinks activation memory quadratically. Note step020 also
FALSIFIED the "two-for-one" hope: prefill_ms was flat at 13.3-13.9 ms while image tokens went
64 -> 4, so prefill is dominated by the text prompt and cutting image tokens buys nothing there.
Resolution is a TOWER-ONLY lever, same as depth. Do not re-quote the two-for-one claim.

PRE-REGISTERED GATES (fixed BEFORE the run):
  (1) VALIDITY. The teacher arm (d12 @512, no student) must land within +-2.0pp of step006's 0.711.
  (2) NAIVE CONTROL. Each resolution's naive (undistilled) arm is re-measured in THIS run, so the
      distilled number is compared against a control from the same code path and eval sample, not
      against step020's separate n=500 draw.
VERDICT RULE, fixed in advance:
  RECOVERABLE  iff distilled top-1 >= naive + 20pp at 256 (the size of the depth-distillation
               recovery, 0.120 -> 0.560, so this is the precedent's own bar, not a fitted one).
  PARTIAL      iff it clears naive by >5pp but not 20pp -- resolution adapts, but not like depth.
  IRRECOVERABLE iff <=5pp over naive: the information is gone from the pixels, not from the weights,
               and no amount of tower training gets it back. That is a REAL and publishable
               negative -- it says the 512 grid carries information this task needs.
T0 SCOUT: small train set and epoch budget. A CAPACITY-style null here is 'at this budget', never
a proven ceiling -- report the loss curve so a still-falling curve cannot be read as convergence.

Output: results/frontier/vlm_step021_lowres_distill_{TAG}__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.frontier.vlm_distill import batches, park, train_student
from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, sample_images
from scripts.frontier.vlm_res import ResEval

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=12, help="STOCK; resolution is the only variable")
parser.add_argument("--res", type=int, nargs="+", default=[256, 384])
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_eval", type=int, default=500)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_train, args.n_eval, args.epochs, args.res = 40, 20, 1, [256]

SLOT = os.environ.get("SGN_SLOT", "local")
TAG = "-".join(f"r{r}" for r in args.res) + f"_e{args.epochs}_t{args.n_train}_n{args.n_eval}"
OUT = ROOT / "results" / "frontier" / f"vlm_step021_lowres_distill_{TAG}__{SLOT}.json"
CKPT_DIR = ROOT / "results" / "frontier"
DATA = ROOT / "data" / "imagenette2-320"
ANCHOR, GATE_PP, RECOVER_PP, PARTIAL_PP = 0.711, 2.0, 20.0, 5.0


def pick_device(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device(args.device)


def pool_to(t, tokens):
    """(B, 64, H) teacher grid -> (B, tokens, H), by average pooling on the square grid.

    64 tokens is 8x8; `tokens` is (res/16)^2/16, also square. Each student token becomes the mean of
    the teacher tokens covering the same region -- the only shape-compatible pairing that preserves
    spatial correspondence. Asserts squareness rather than trusting it.

    NOTE (fixed after the first run): the original used avg_pool2d with an integer stride and
    asserted g % k == 0. That holds for 8x8 -> 4x4 (res 256) but CRASHES on 8x8 -> 6x6 (res 384),
    which is not an integer pool. adaptive_avg_pool2d handles both: it is identical to avg_pool2d
    when the grids divide evenly, so the res-256 cell is bit-unchanged, and for 6x6 it averages
    over the overlapping windows that cover each output cell -- still a spatial mean over the right
    image region. Squareness is still asserted; only the divisibility requirement is dropped.
    """
    b, n, h = t.shape
    g, k = int(round(n ** 0.5)), int(round(tokens ** 0.5))
    assert g * g == n and k * k == tokens, f"{n} -> {tokens} not a square-to-square pool"
    x = t.view(b, g, g, h).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(x, k).permute(0, 2, 3, 1).reshape(b, tokens, h)


def evaluate(ev, val, res):
    """top-1 + mean vision/prefill ms at the current vision stack and resolution."""
    ev.res = res
    ok = vis = pre = 0.0
    for path, wnid in val:
        b = ev.batch(Image.open(path).convert("RGB"))
        feats, v = ev.encode(b)
        emb, _ = ev.merge(b, feats)
        _, pred, p = ev.forward(emb)
        ok += LABELS[pred] == WNID2LABEL[wnid]; vis += v; pre += p
    n = len(val)
    return {"top1": ok / n, "vision_ms": vis / n, "prefill_ms": pre / n, "n": n}


def main():
    print("=" * 78, flush=True)
    print(f"vlm_step021 low-res distill  device={DEVICE}  depth={args.depth}  res={args.res}  "
          f"n_train={args.n_train} epochs={args.epochs}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
    ev = ResEval(model.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    ev.set_depth(args.depth)

    train = sample_images(DATA / "train", args.n_train, seed=args.seed)
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    print(f"  {len(train)} train / {len(val)} eval images", flush=True)

    res_full = {"teacher": evaluate(ev, val, 512)}
    d = abs(res_full["teacher"]["top1"] - ANCHOR) * 100
    print(f"  GATE validity  teacher d12@512 top1 {res_full['teacher']['top1']:.4f} vs {ANCHOR} "
          f"-> {'PASS' if d <= GATE_PP else 'FAIL'}", flush=True)

    # Teacher targets: cached ONCE at 512 with the full stack, then pooled per student resolution.
    ev.res = 512
    ev.set_depth(len(ev.full))
    tgt512, t0 = [], time.perf_counter()
    for i, chunk in enumerate(batches(train, args.bs)):
        ims = [Image.open(p).convert("RGB") for p, _ in chunk]
        flat, _ = ev.encode(ev.pixels(ims))
        tgt512.append(flat.view(len(ims), -1, flat.shape[-1]).cpu().half())
        if i % 25 == 0: print(f"    teacher cache {i * args.bs}/{len(train)}", flush=True)
    print(f"    teacher cache done in {time.perf_counter() - t0:.0f}s", flush=True)

    for res in args.res:
        tokens = (res // 16) ** 2 // 16
        print(f"\n=== res {res} ({tokens} tokens) ===", flush=True)
        ev.set_depth(args.depth)
        naive = evaluate(ev, val, res)
        print(f"  naive (stock tower, no training)  top1 {naive['top1']:.4f}", flush=True)

        ev.res = res
        park(ev, "cpu")
        targets = [pool_to(t.float(), tokens).half() for t in tgt512]
        student, hist = train_student(ev, train, targets, args.depth, args.epochs, args.bs,
                                      args.lr, DEVICE)
        park(ev, DEVICE)
        torch.save(student.state_dict(), CKPT_DIR / f"vlm_step021_student_r{res}.pt")

        ev.set_layers(student)
        dist = evaluate(ev, val, res)
        gain = (dist["top1"] - naive["top1"]) * 100
        verdict = ("RECOVERABLE" if gain >= RECOVER_PP else
                   "PARTIAL" if gain > PARTIAL_PP else "IRRECOVERABLE")
        print(f"  distilled top1 {dist['top1']:.4f}  (+{gain:.2f}pp over naive) -> {verdict}",
              flush=True)
        print(f"  vs teacher {res_full['teacher']['top1']:.4f} @ "
              f"{res_full['teacher']['vision_ms']:.2f} ms; student {dist['vision_ms']:.2f} ms "
              f"({res_full['teacher']['vision_ms'] / dist['vision_ms']:.2f}x tower)", flush=True)
        res_full[res] = {"tokens": tokens, "naive": naive, "distilled": dist, "gain_pp": gain,
                         "verdict": verdict, "hist": hist,
                         "tower_speedup": res_full["teacher"]["vision_ms"] / dist["vision_ms"]}
        ev.set_depth(args.depth)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step021", "depth": args.depth, "anchor": ANCHOR,
                               "n_train": len(train), "epochs": args.epochs, "lr": args.lr,
                               "seed": args.seed, "device": str(DEVICE), "cells": res_full},
                              indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
