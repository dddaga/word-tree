"""vlm_step038 -- WHY does one image stall 246-802 ms, and does it survive without the rebuild?

Section 55 established the stall is SPIKE (one image carries the whole pass excess), deterministic in
the REP axis (16 paired reps of step036 vs step037 agree to 0.047 ms; departures at fixed indices 0, 3,
5, 10, 11 with monotone magnitudes) and random in the IMAGE axis (67, 175, 139, 248, 356 of 500). Any
mechanism must spend a FIXED budget per rep at an UNPREDICTABLE moment. Allocator reclaim fits: cost
set by allocation history (deterministic in rep), fired when the next allocation happens to cross a
threshold (arbitrary in image). Nothing else proposed fits both axes.

Two arms, one run:
  rebuild -- step037's protocol (per-rep _dynamo.reset / del ev / empty_cache) plus instrumentation.
             Reproduces the stall by construction, so the counters have something to explain. Per-image
             `reserved` is a cheap C call; `num_alloc_retries` is the blocking cudaFree-and-retry path.
  soak    -- build ONCE, run the same R passes back to back. A deployed drone builds once. This is the
             only version of the question the paper needs, and it costs one run.

PRE-REGISTERED, decided before any draw:
  soak CLEAN (0 departures at the same gate) and rebuild DIRTY -> the stall is an EVAL-PROTOCOL
    ARTIFACT of per-rep rebuilding, like the section 44 GraphEval clone. No latency guarantee at risk,
    section 55's drone paragraph is retracted, and the harness gets a standing note.
  soak DIRTY -> a deployment defect. The >=802 ms per-frame worst case stands and is the number a
    real-time claim must survive.
  A stall image whose `reserved` jumps, or a pass whose `num_alloc_retries` increments on exactly the
    departure -> allocator reclaim CONFIRMED as the mechanism.
  Departures with flat reserved AND zero retries -> allocator hypothesis KILLED; the hunt moves to
    host page faults / driver preemption (section 55 open items 2-3).
  soak clean at R=16 bounds a rate, never a mechanism (section 51/53) -- power stated, not assumed.

Rerun under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to test the fix directly; the env var is
recorded in the JSON so the two runs are never confused.
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
parser.add_argument("--arms", nargs="+", default=["rebuild", "soak"])
parser.add_argument("--mode", default="compile", help="section 55 measured the stall in this cell")
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--repeats", type=int, default=16)
parser.add_argument("--n_eval", type=int, default=500)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag_suffix", default="")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.warmup, args.repeats = 20, 3, 2

SLOT = os.environ.get("SGN_SLOT", "local")
ALLOC_CONF = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
TAG = f"{args.dtype}_{'-'.join(args.arms)}_r{args.repeats}_n{args.n_eval}{args.tag_suffix}"
OUT = ROOT / "results" / "frontier" / f"vlm_step038_stall_mechanism_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
CKPT = ROOT / "results" / "frontier" / args.ckpt
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}
GRAPH_MODES = {"reduce-overhead", "max-autotune"}
# Carried over VERBATIM from step036/step037 so no arm can retune a threshold to a count.
DEPARTURE_MS = {"compile": 4.5, "reduce-overhead": 3.0}
STALL_MS = 50.0  # a per-image stall; base images run 4.0-5.5 ms, spikes 246-802 ms. No middle ground.
MB = 1 << 20
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def stat(xs):
    """Median with MIN-MAX. Section 49: IQR discards the one-sided tail this line exists to measure."""
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"median": med, "min": s[0], "max": s[-1], "tail_ms": s[-1] - med}


def retries():
    if DEVICE.type != "cuda":
        return 0
    st = torch.cuda.memory_stats()
    return int(st.get("num_alloc_retries", 0))


def timed_pass(ev, val, warmup):
    """step037's pass plus per-image `reserved` bytes and per-pass alloc-retry counters."""
    for path, _ in val[:warmup]:
        ev.run(Image.open(path).convert("RGB"))
    r0 = retries()
    t0, ok, vis, per_image, reserved = time.perf_counter(), 0.0, 0.0, [], []
    for path, wnid in val:
        b = ev.batch(Image.open(path).convert("RGB"))
        feats, v = ev.encode(b)
        emb, _ = ev.merge(b, feats)
        _, pred, p = ev.forward(emb)
        ok += LABELS[pred] == WNID2LABEL[wnid]
        vis += v
        per_image.append(p)
        reserved.append(torch.cuda.memory_reserved() if DEVICE.type == "cuda" else 0)
    n = len(val)
    stalls = [{"idx": i, "ms": p, "reserved_mb": reserved[i] / MB,
               "reserved_jump_mb": (reserved[i] - reserved[i - 1]) / MB if i else 0.0}
              for i, p in enumerate(per_image) if p >= STALL_MS]
    return {"top1": ok / n, "vision_ms": vis / n, "prefill_ms": sum(per_image) / n,
            "prefill_per_image": per_image, "reserved_mb": [x / MB for x in reserved],
            "prefill_img_median": stat(per_image)["median"], "prefill_img_max": max(per_image),
            "stalls": stalls, "alloc_retries": retries() - r0,
            "reserved_start_mb": reserved[0] / MB, "reserved_end_mb": reserved[-1] / MB,
            "end2end_ms": (vis + sum(per_image)) / n, "wall_s": time.perf_counter() - t0,
            "load1": os.getloadavg()[0]}


def build(mode, sd):
    torch._dynamo.reset()
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=DTYPE[args.dtype])
    cls = GraphEval if mode in GRAPH_MODES else VLMEval
    ev = cls(m.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    kw = {"dynamic": False} if mode == "compile" else {"dynamic": False, "mode": mode}
    ev.m.model.vision_model = torch.compile(ev.m.model.vision_model, **kw)
    ev.m.model.text_model = torch.compile(ev.m.model.text_model, **kw)
    inter = ev.full[0].mlp.fc1.out_features
    st = build_width_student(ev, args.student_depth, int(round(inter * args.ratio)), "cpu")
    st.load_state_dict(sd)
    ev.set_layers(st.to(DEVICE).to(DTYPE[args.dtype]))
    return ev, st


def main():
    print(f"{'=' * 78}\nvlm_step038 stall mechanism  device={DEVICE}  mode={args.mode}  "
          f"arms={args.arms}  R={args.repeats}  n_eval={args.n_eval}\n"
          f"PYTORCH_CUDA_ALLOC_CONF={ALLOC_CONF or '(default)'}", flush=True)
    assert CKPT.exists(), f"missing student checkpoint: {CKPT}"
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    sd = torch.load(CKPT, map_location="cpu")
    gate, cells = DEPARTURE_MS[args.mode], {}

    for arm in args.arms:
        runs, ev, st = [], None, None
        for rep in range(args.repeats):
            if arm == "rebuild" or ev is None:
                ev, st = build(args.mode, sd)
            r = timed_pass(ev, val, args.warmup)
            r["departure"] = r["prefill_ms"] >= gate
            runs.append(r)
            s = r["stalls"][0] if r["stalls"] else None
            print(f"  {arm} rep{rep}  top1 {r['top1']:.4f}  tower {r['vision_ms']:6.2f}  "
                  f"prefill {r['prefill_ms']:6.2f}  reserved {r['reserved_start_mb']:.0f}->"
                  f"{r['reserved_end_mb']:.0f} MB  retries {r['alloc_retries']}  "
                  f"wall {r['wall_s']:.1f}s  load1 {r['load1']:.2f}"
                  + (f"  <== STALL img{s['idx']} {s['ms']:.0f} ms "
                     f"(jump {s['reserved_jump_mb']:+.0f} MB)" if s else ""), flush=True)
            if arm == "rebuild":
                del ev
                ev = None
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        deps = [r for r in runs if r["departure"]]
        stalls = [s for r in runs for s in r["stalls"]]
        cells[arm] = {"n_params": n_params(st), "gate_ms": gate, "stall_ms": STALL_MS,
                      "departures": len(deps), "repeats": args.repeats, "runs": runs,
                      "n_stalls": len(stalls), "total_retries": sum(r["alloc_retries"] for r in runs),
                      "stats": {m: stat([x[m] for x in runs])
                                for m in ("vision_ms", "prefill_ms", "end2end_ms")}}
        print(f"    {arm}: {len(deps)}/{args.repeats} departures (gate {gate} ms), "
              f"{len(stalls)} stalls >={STALL_MS:.0f} ms, "
              f"{cells[arm]['total_retries']} alloc retries", flush=True)
        if arm == "soak" and not deps:
            print(f"      soak CLEAN at R={args.repeats}: pre-registered read is EVAL-PROTOCOL "
                  f"ARTIFACT. Bounds a rate, NOT a mechanism (section 51/53) -- state the power.",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step038", "n_eval": len(val), "repeats": args.repeats,
                               "seed": args.seed, "device": str(DEVICE), "ckpt": args.ckpt,
                               "dtype": args.dtype, "arms": args.arms, "mode": args.mode,
                               "ratio": args.ratio, "alloc_conf": ALLOC_CONF,
                               "departure_gate_ms": gate, "stall_gate_ms": STALL_MS,
                               "cells": cells}, indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
