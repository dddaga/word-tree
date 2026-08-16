"""vlm_step037 -- what SHAPE is the prefill excursion? Sections 49-54 measured it as a per-pass MEAN
over n_eval images and inferred "per-launch host dispatch jitter" from that mean alone.

The mean cannot carry that inference. Section 54's captured rep15 drew prefill 7.08 ms against a 2.75
base: +4.33 ms x 500 images = +2.17 s of extra prefill inside one pass (wall 51.6 s vs 48.6/49.9 for
its neighbours, so the excess is real and not a timer artifact). Three very different mechanisms
produce that same mean, and `ev.forward()` already returns per-image prefill -- the harness has simply
been summing it away.

This script keeps the per-image vector and classifies the shape of every departure pass.

PRE-REGISTERED, decided before any draw (excess = pass_sum - median_pass_sum):
  SPIKE     -- a single image carries >=50% of the excess. The excursion is a one-off STALL, not
               jitter. Section 49's dispatch attribution is FALSIFIED as stated; the hunt moves to
               allocator reclaim / recapture / driver / page-fault territory.
  SUSTAINED -- the departure pass's per-image MEDIAN shifts by >=50% of the mean shift, i.e. nearly
               every image is slower. A whole-pass state change (clock, cache residency, memory
               layout), also NOT per-launch jitter.
  JITTER    -- neither: the excess is spread across many images with no dominant contributor and the
               pass median barely moves. Section 49's mechanism SUPPORTED, on evidence that can
               actually carry it.
Reported with median and MIN-MAX per section 49; per-image vectors of departure passes are dumped in
full so the classification can be re-read later without a rerun.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, VLMEval, sample_images
from scripts.frontier.vlm_graph import GraphEval
from scripts.frontier.vlm_width import build_width_student, n_params

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--student_depth", type=int, default=6)
parser.add_argument("--ratio", type=float, default=0.25)
parser.add_argument("--ckpt", default="vlm_step023_mlp_r0.25_r0.25_e50_t9352_n500_d6.pt")
parser.add_argument("--modes", nargs="+", default=["compile"],
                    help="compile first: section 54 puts its departure rate at 6/20, so it reaches "
                         "the shape question ~6x cheaper than the captured cell")
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--repeats", type=int, default=12)
parser.add_argument("--n_eval", type=int, default=500)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag_suffix", default="")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.warmup, args.repeats = 20, 3, 2

SLOT = os.environ.get("SGN_SLOT", "local")
TAG = (f"{args.dtype}_{'-'.join(m.replace('-', '') for m in args.modes)}"
       f"_r{args.repeats}_n{args.n_eval}{args.tag_suffix}")
OUT = ROOT / "results" / "frontier" / f"vlm_step037_tail_shape_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
CKPT = ROOT / "results" / "frontier" / args.ckpt
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}
GRAPH_MODES = {"reduce-overhead", "max-autotune"}
# Section 54 committed >=4.5 for compile (base 4.17-4.30) and >=3.0 for capture (base 2.7x). Both are
# ~+10% on their base; carried over verbatim so this arm cannot retune the threshold to a count.
DEPARTURE_MS = {"compile": 4.5, "reduce-overhead": 3.0}
SHARE_GATE = 0.50  # fraction of the excess that makes a single image, or the median shift, decisive
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def stat(xs):
    """Median with MIN-MAX. Section 49: IQR discards the one-sided tail this line exists to measure."""
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"median": med, "min": s[0], "max": s[-1], "tail_ms": s[-1] - med}


def timed_pass(ev, val, warmup):
    """Identical to step033's pass except the per-image prefill vector is KEPT, not summed away."""
    for path, _ in val[:warmup]:
        ev.run(Image.open(path).convert("RGB"))
    t0, ok, vis, per_image = time.perf_counter(), 0.0, 0.0, []
    for path, wnid in val:
        b = ev.batch(Image.open(path).convert("RGB"))
        feats, v = ev.encode(b)
        emb, _ = ev.merge(b, feats)
        _, pred, p = ev.forward(emb)
        ok += LABELS[pred] == WNID2LABEL[wnid]
        vis += v
        per_image.append(p)
    n = len(val)
    return {"top1": ok / n, "vision_ms": vis / n, "prefill_ms": sum(per_image) / n,
            "prefill_per_image": per_image,
            "prefill_img_median": stat(per_image)["median"],
            "prefill_img_max": max(per_image),
            "end2end_ms": (vis + sum(per_image)) / n, "wall_s": time.perf_counter() - t0,
            "load1": os.getloadavg()[0]}


def build(mode):
    torch._dynamo.reset()
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=DTYPE[args.dtype])
    cls = GraphEval if mode in GRAPH_MODES else VLMEval
    ev = cls(m.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    kw = {"dynamic": False} if mode == "compile" else {"dynamic": False, "mode": mode}
    ev.m.model.vision_model = torch.compile(ev.m.model.vision_model, **kw)
    ev.m.model.text_model = torch.compile(ev.m.model.text_model, **kw)
    return ev


def classify(dep, base_runs, n):
    """Shape of one departure pass, against the median of the non-departure passes of its own cell."""
    base_mean = stat([r["prefill_ms"] for r in base_runs])["median"]
    base_img_med = stat([r["prefill_img_median"] for r in base_runs])["median"]
    excess_ms = (dep["prefill_ms"] - base_mean) * n           # total extra ms in the pass
    top = max(dep["prefill_per_image"])
    top_share = (top - base_img_med) / excess_ms if excess_ms > 0 else 0.0
    med_share = (dep["prefill_img_median"] - base_img_med) * n / excess_ms if excess_ms > 0 else 0.0
    if top_share >= SHARE_GATE:
        v = "SPIKE"
    elif med_share >= SHARE_GATE:
        v = "SUSTAINED"
    else:
        v = "JITTER"
    return {"pass_mean_ms": dep["prefill_ms"], "excess_ms": excess_ms, "max_image_ms": top,
            "top_image_share": top_share, "median_shift_share": med_share,
            "base_mean_ms": base_mean, "base_image_median_ms": base_img_med, "shape": v}


def main():
    print(f"{'=' * 78}\nvlm_step037 tail shape  device={DEVICE}  dtype={args.dtype}  "
          f"modes={args.modes}  R={args.repeats}  n_eval={args.n_eval}", flush=True)
    assert CKPT.exists(), f"missing student checkpoint: {CKPT}"
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    sd = torch.load(CKPT, map_location="cpu")
    cells = {}

    for mode in args.modes:
        gate, runs = DEPARTURE_MS[mode], []
        for rep in range(args.repeats):
            ev = build(mode)  # rebuilt per repeat, per section 49
            inter = ev.full[0].mlp.fc1.out_features
            st = build_width_student(ev, args.student_depth, int(round(inter * args.ratio)), "cpu")
            st.load_state_dict(sd)
            ev.set_layers(st.to(DEVICE).to(DTYPE[args.dtype]))
            r = timed_pass(ev, val, args.warmup)
            r["departure"] = r["prefill_ms"] >= gate
            runs.append(r)
            print(f"  {mode} rep{rep}  top1 {r['top1']:.4f}  tower {r['vision_ms']:6.2f}  "
                  f"prefill {r['prefill_ms']:6.2f} (img med {r['prefill_img_median']:5.2f}  "
                  f"img max {r['prefill_img_max']:7.2f})  wall {r['wall_s']:.1f}s  "
                  f"load1 {r['load1']:.2f}{'  <== DEPARTURE' if r['departure'] else ''}", flush=True)
            del ev
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        deps = [r for r in runs if r["departure"]]
        base = [r for r in runs if not r["departure"]]
        shapes = [classify(d, base, len(val)) for d in deps] if base else []
        cells[mode] = {"n_params": n_params(st), "gate_ms": gate,
                       "departures": len(deps), "repeats": args.repeats,
                       "shapes": shapes, "runs": runs,
                       "stats": {m: stat([x[m] for x in runs])
                                 for m in ("vision_ms", "prefill_ms", "end2end_ms")}}
        print(f"    {mode}: {len(deps)}/{args.repeats} departures (gate {gate} ms)", flush=True)
        for s in shapes:
            print(f"      pass {s['pass_mean_ms']:.2f} ms  excess {s['excess_ms']:.0f} ms  "
                  f"max image {s['max_image_ms']:.1f} ms  top-image share {s['top_image_share']:.2f}  "
                  f"median-shift share {s['median_shift_share']:.2f} -> {s['shape']}", flush=True)
        if not deps:
            print(f"      no departures: shape question UNANSWERED for {mode}, not resolved. "
                  f"Section 51/53 rule -- a zero count bounds a rate, never a mechanism.", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step037", "n_eval": len(val), "repeats": args.repeats,
                               "seed": args.seed, "device": str(DEVICE), "ckpt": args.ckpt,
                               "dtype": args.dtype, "modes": args.modes, "ratio": args.ratio,
                               "departure_gate_ms": DEPARTURE_MS, "share_gate": SHARE_GATE,
                               "cells": cells}, indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
