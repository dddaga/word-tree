"""vlm_step011: does d6's parity survive when the object is SMALL? (part 3 SS11.6, the control task)

Every accuracy claim in this trajectory -- including "d6 reaches teacher parity at 16.58% fewer
params", the load-bearing claim of the whole drone recipe -- was measured on 10-way whole-object
Imagenette, where the target fills the frame. A drone does not see that. It sees the target at
distance, occupying a small fraction of the sensor. If the layers depth truncation deletes are the
ones carrying fine spatial detail, parity is an artefact of the test images and the recipe does not
transfer. Part 3 SS11.6 named this "before any drone claim"; SS14 closed the token lever and SS15
left this the last open validity question ahead of the drone write-up.

Manipulation: one variable, object scale. Each val image is resized to a fraction f of the canvas
and pasted centred on a neutral gray canvas of the ORIGINAL size, so the object occupies f^2 of the
pixels and the input resolution the processor sees is unchanged. f = 1.0 is the untouched image.

CONFOUND, stated up front: f < 1 changes object scale AND introduces uniform padding, which is
out-of-distribution for a photographic model. The two cannot be separated in the main effect
[acc(f) - acc(1.0)], so that column is read as descriptive only. The padding is applied IDENTICALLY
at both depths, so it differences out of the interaction -- which is why the double difference, not
the main effect, is the pre-registered quantity of interest.

PRE-REGISTERED DECISION RULE (written before the run):
  * Quantity of interest, same statistic and same shared-index bootstrap as vlm_step009/010:
        I(f) = [acc(d6,f) - acc(d6,1.0)] - [acc(d12,f) - acc(d12,1.0)]
    It asks whether shrinking the object costs the TRUNCATED tower more than the teacher.
  * ORTHOGONAL (parity survives; the drone claim generalises) iff I(f)'s 95% CI is contained in
    +-2.0pp. SUB-ADDITIVE (depth truncation destroys fine detail; the drone claim is SCOPED to
    large objects) iff upper < -2.0pp. SUPER-ADDITIVE iff lower > +2.0pp. Else INCONCLUSIVE,
    reported as such and not rounded into a pass.
  * +-2.0pp inherited from vlm_step006, where a 5.3pp headline swing turned out to be eval noise.
  * VALIDITY ANCHORS: f = 1.0 is the untouched image, so `d12_f100` must reproduce step006's 0.711
    and `d6_f100` its 0.713. If they do not, nothing else in the run is readable.
  * Read `agree` alongside top-1 in every cell. vlm_step009 CONFIRMED agree(d6,d12) = 0.499 at
    matched top-1: these towers are equally accurate and NOT functionally equivalent, so a matched
    top-1 at small f would still not license "the student sees what the teacher sees".
  * Timing columns are recorded but are NOT claims: this run is accuracy-only and may share a box.

Output: results/frontier/vlm_step011_fine_detail_{TAG}__{SLOT}.json  (includes per-image records)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))
if (_LIBS := ROOT / "vlm_libs").is_dir(): sys.path.insert(0, str(_LIBS))

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.frontier.vlm_distill import load_student
from scripts.frontier.vlm_eval import (LABELS, WNID2LABEL, VLMEval, boot_ci, mcnemar_exact,
                                       sample_images)
from scripts.frontier.vlm_quant import boot_dd, call_ortho, hits

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=6, help="the distilled arm; 12 is always the ref")
parser.add_argument("--ckpt", default="results/frontier/vlm_step004_tower_distill_d6_25ep9k__mini_mps.pt")
parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.25],
                    help="object side as a fraction of the canvas; 1.0 is the untouched image")
parser.add_argument("--n_eval", type=int, default=3900, help="seed 42 -> step006's 3859 images")
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.boot = 20, 200

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
    if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
GRAY = (128, 128, 128)  # neutral canvas; the confound this creates is differenced out by I(f)
TAG = f"d{args.depth}_" + "-".join(f"f{int(round(s*100))}" for s in args.scales)
OUT = ROOT / "results" / "frontier" / f"vlm_step011_fine_detail_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"


def cell(depth, s):
    return f"d{depth}_f{int(round(s * 100))}"


def shrink(img, s):
    """Paste `img` resized to a fraction `s` of its own size, centred on a gray canvas of the
    ORIGINAL size. Canvas size is held fixed so the processor's input resolution never varies and
    scale is the only thing that moves. s >= 1.0 is a no-op, so f=1.0 is the untouched image."""
    if s >= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), GRAY)
    canvas.paste(small, ((w - small.size[0]) // 2, (h - small.size[1]) // 2))
    return canvas


def evaluate(ev, tower, s, val):
    """One pass over the images at a fixed (depth, scale). Image order is fixed by sample_images,
    so index i is the same image in every pass and the pairing survives across all cells."""
    ev.set_layers(tower)
    preds, vis, pre, lps, t0 = [], 0.0, 0.0, [], time.perf_counter()
    for i, (path, _) in enumerate(val):
        lg, pred, vm, pm = ev.run(shrink(Image.open(path).convert("RGB"), s))
        preds.append(pred); vis += vm; pre += pm
        lps.append(F.log_softmax(lg, -1).cpu())
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"    [{i+1}/{len(val)}] {el:.0f}s eta {el/(i+1)*(len(val)-i-1):.0f}s", flush=True)
    n = len(val)
    return preds, lps, vis / n, pre / n


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"{'='*78}\nvlm_step011 fine detail  device={DEVICE}  depth={args.depth}  "
          f"scales={args.scales}  n_eval={args.n_eval}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    m.requires_grad_(False)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    ck = Path(args.ckpt)
    if not ck.is_absolute(): ck = ROOT / ck
    towers = [(12, ev.full), (args.depth, list(load_student(ev, ck, args.depth, DEVICE)))]
    print(f"  distilled tower <- {ck}")

    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    print(f"  {len(val)} eval images; {len(towers) * len(args.scales)} cells", flush=True)
    out, ref_lp = {}, None
    for d, tower in towers:
        for s in args.scales:
            print(f"  --- {cell(d, s)} ---", flush=True)
            out[cell(d, s)] = evaluate(ev, tower, s, val)
            if ref_lp is None: ref_lp = out[cell(d, s)][1]

    n = len(val)
    ref = cell(12, args.scales[0])
    gold = [LABELS.index(WNID2LABEL[w]) for _, w in val]
    recs = [{"gold": g, "path": p.name, **{c: out[c][0][i] for c in out}}
            for i, ((p, _), g) in enumerate(zip(val, gold))]
    res = {"step": "vlm_step011", "device": str(DEVICE), "depth": args.depth, "ckpt": str(ck),
           "scales": args.scales, "n_eval": n, "seed": args.seed, "cells": {}, "drop": {},
           "interactions": {}, "records": recs}
    print(f"\n  {'cell':<12} {'top1':>6} {'agree':>6} {'KL':>7} {'vis_ms':>8} {'pre_ms':>8}")
    for d, _ in towers:
        for s in args.scales:
            preds, lps, v, p_ = out[cell(d, s)]
            kl = sum(float(F.kl_div(lp, r, log_target=True, reduction="sum"))
                     for lp, r in zip(lps, ref_lp)) / n
            row = {"depth": d, "scale": s,
                   "top1": sum(int(a == b) for a, b in zip(preds, gold)) / n,
                   "agree_with_ref": sum(int(a == b) for a, b in zip(preds, out[ref][0])) / n,
                   "kl_vs_ref": kl, "vision_ms": round(v, 2), "prefill_ms": round(p_, 2)}
            res["cells"][cell(d, s)] = row
            print(f"  {cell(d,s):<12} {row['top1']:>6.3f} {row['agree_with_ref']:>6.3f} {kl:>7.3f} "
                  f"{v:>8.1f} {p_:>8.1f}")

    print(f"\n  cost of shrinking, per depth (DESCRIPTIVE ONLY -- confounded by the gray padding)")
    for d, _ in towers:
        for s in (x for x in args.scales if x != args.scales[0]):
            a, b_ = hits(recs, cell(d, s)), hits(recs, cell(d, args.scales[0]))
            b = sum(1 for i in range(n) if a[i] and not b_[i])
            c = sum(1 for i in range(n) if b_[i] and not a[i])
            lo, hi = boot_ci(a, b_, args.boot)
            res["drop"][cell(d, s)] = {"delta": (sum(a) - sum(b_)) / n, "boot95_lo": lo,
                                       "boot95_hi": hi, "b": b, "c": c,
                                       "p_two_sided": mcnemar_exact(b, c)}
            print(f"  {cell(d,s):<12} d {(sum(a)-sum(b_))/n:+.4f}  boot95 [{lo:+.4f}, {hi:+.4f}]  "
                  f"b={b} c={c} p={mcnemar_exact(b,c):.5f}")

    print(f"\n  double difference I(f) = [d{args.depth} shrink cost] - [d12 shrink cost]  <- THE CALL")
    for s in (x for x in args.scales if x != args.scales[0]):
        pt, lo, hi = boot_dd(recs, [cell(args.depth, s), cell(args.depth, args.scales[0]),
                                    cell(12, s), cell(12, args.scales[0])], args.boot)
        v = call_ortho(lo, hi, lever="shrinking the object")
        res["interactions"][f"f{int(round(s*100))}"] = {"I": pt, "boot95_lo": lo, "boot95_hi": hi,
                                                        "verdict": v}
        print(f"  f={s:<6} I {pt:+.4f}  boot95 [{lo:+.4f}, {hi:+.4f}]  -> {v}")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
