"""vlm_step033 -- CUDA graphs on prefill: does removing per-launch host dispatch kill BOTH the
median and the tail?

Section 49 (VLM_TRAJECTORY_part15) measured, over 5 repeats per cell, that every stage of this
pipeline is deterministic to ~0.1% EXCEPT compiled prefill, which shows a one-sided upward excursion
at ~1-in-5 incidence (base mode 4.17-4.23 ms, excursions 4.67 and 5.36). Prefill is the dispatch-
bound stage, so the excursion is consistent with host-side scheduling jitter made visible once
compile shrank the GPU work. Graph capture removes exactly that per-launch dispatch: one replay
instead of N launches. If the hypothesis holds this lowers the median AND removes the tail.

Route: `torch.compile(mode="reduce-overhead")`, inductor's cudagraphs path, not hand-rolled capture.
Shapes are static by construction (fixed prompt + 64 image tokens, nosplit), the precondition
capture needs, and this reuses the already-validated harness.

CORRECTNESS RISK, GATED: `choose()` reuses the prefill's `past_key_values` across 10 teacher-forced
label passes through the SAME text_model, and cudagraph outputs live in graph-owned memory a later
replay overwrites. `vlm_graph.GraphEval` copies the cache out (outside the timed window); the UNSAFE
verdict below is the check that it worked. A fast wrong answer is not a result.

PRE-REGISTERED, per cell vs the `compile` (default inductor) baseline:
  WIN     -- prefill median AND max-median (the tail) both fall by >0.03 ms, section 49's measured
             base-mode dispersion. Dispatch hypothesis SUPPORTED.
  PARTIAL -- median falls, tail survives => dispatch is NOT the cause and section 49's cause
             hypothesis is FALSIFIED; the tail needs a different explanation.
  NULL    -- neither moves. Prefill is not launch-bound; the prefill axis closes.
  UNSAFE  -- top-1 moves past the accuracy gate. Latency discarded regardless of value.
Reported as median with MIN-MAX, never IQR: section 49 showed IQR discards the tail by construction.
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
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--student_depth", type=int, default=6)
parser.add_argument("--ratio", type=float, default=0.25)
parser.add_argument("--ckpt", default="vlm_step023_mlp_r0.25_r0.25_e50_t9352_n500_d6.pt")
parser.add_argument("--modes", nargs="+", default=["compile", "reduce-overhead"],
                    help="inductor modes; 'compile' = default inductor, the section 49 baseline")
parser.add_argument("--towers", nargs="+", default=["student"], choices=["teacher", "student"])
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--repeats", type=int, default=5)
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
       f"_{'-'.join(args.towers)}_r{args.repeats}_n{args.n_eval}{args.tag_suffix}")
OUT = ROOT / "results" / "frontier" / f"vlm_step033_cudagraph_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
CKPT = ROOT / "results" / "frontier" / args.ckpt
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}
GRAPH_MODES = {"reduce-overhead", "max-autotune"}  # inductor modes that capture cudagraphs
ACC_GATE = 0.02        # top-1 movement above this => cudagraphs corrupted the KV cache
BASE_DISPERSION = 0.03  # ms; section 49's measured base-mode spread, the resolution floor
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def stat(xs):
    """Median with MIN-MAX. Section 49: IQR hides the one-sided tail this arm exists to measure."""
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"median": med, "min": s[0], "max": s[-1], "tail_ms": s[-1] - med}


def timed_pass(ev, val, warmup):
    for path, _ in val[:warmup]:
        ev.run(Image.open(path).convert("RGB"))
    t0, ok, vis, pre = time.perf_counter(), 0.0, 0.0, 0.0
    for path, wnid in val:
        b = ev.batch(Image.open(path).convert("RGB"))
        feats, v = ev.encode(b)
        emb, _ = ev.merge(b, feats)
        _, pred, p = ev.forward(emb)
        ok += LABELS[pred] == WNID2LABEL[wnid]
        vis += v
        pre += p
    n = len(val)
    return {"top1": ok / n, "vision_ms": vis / n, "prefill_ms": pre / n,
            "end2end_ms": (vis + pre) / n, "wall_s": time.perf_counter() - t0,
            "load1": os.getloadavg()[0]}


def build(mode):
    torch._dynamo.reset()
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=DTYPE[args.dtype])
    # GraphEval only for graph-capturing modes: it clones the KV cache out of graph memory, and
    # running that clone in the baseline cell would make the comparison non-one-variable.
    cls = GraphEval if mode in GRAPH_MODES else VLMEval
    ev = cls(m.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    kw = {"dynamic": False} if mode == "compile" else {"dynamic": False, "mode": mode}
    ev.m.model.vision_model = torch.compile(ev.m.model.vision_model, **kw)
    ev.m.model.text_model = torch.compile(ev.m.model.text_model, **kw)
    return ev


def main():
    print(f"{'=' * 78}\nvlm_step033 cudagraph  device={DEVICE}  dtype={args.dtype}  modes={args.modes}  "
          f"towers={args.towers}  R={args.repeats}  n_eval={args.n_eval}", flush=True)
    assert CKPT.exists(), f"missing student checkpoint: {CKPT}"
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    sd = torch.load(CKPT, map_location="cpu")
    cells = {}

    for mode in args.modes:
        for tower in args.towers:
            runs = []
            for rep in range(args.repeats):
                # Rebuilt every repeat: reusing a compiled/captured object would measure one
                # capture session R times and report its variance as zero (section 49).
                ev = build(mode)
                inter = ev.full[0].mlp.fc1.out_features
                if tower == "teacher":
                    ev.set_depth(args.depth)
                    pr = n_params(nn.ModuleList(list(ev.full[:args.depth])))
                else:
                    st = build_width_student(ev, args.student_depth,
                                             int(round(inter * args.ratio)), "cpu")
                    st.load_state_dict(sd)
                    ev.set_layers(st.to(DEVICE).to(DTYPE[args.dtype]))
                    pr = n_params(st)
                r = timed_pass(ev, val, args.warmup)
                runs.append(r)
                print(f"  {mode}_{tower} rep{rep}  top1 {r['top1']:.4f}  "
                      f"tower {r['vision_ms']:6.2f}  prefill {r['prefill_ms']:6.2f}  "
                      f"e2e {r['end2end_ms']:6.2f} ms  wall {r['wall_s']:.1f}s  "
                      f"load1 {r['load1']:.2f}", flush=True)
                del ev
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
            key = f"{mode}_{tower}"
            cells[key] = {"runs": runs, "tower_params": pr,
                          "top1": [x["top1"] for x in runs],
                          "stats": {m: stat([x[m] for x in runs])
                                    for m in ("vision_ms", "prefill_ms", "end2end_ms")}}
            for m in ("vision_ms", "prefill_ms", "end2end_ms"):
                s = cells[key]["stats"][m]
                print(f"    {key:24s} {m:11s} median {s['median']:6.2f}  "
                      f"min {s['min']:6.2f}  max {s['max']:6.2f}  tail {s['tail_ms']:5.2f}",
                      flush=True)

    base = args.modes[0]
    verdicts = {}
    for mode in args.modes[1:]:
        for tower in args.towers:
            b, c = cells.get(f"{base}_{tower}"), cells.get(f"{mode}_{tower}")
            if not (b and c):
                continue
            acc = abs(max(c["top1"]) - max(b["top1"]))
            d_med = b["stats"]["prefill_ms"]["median"] - c["stats"]["prefill_ms"]["median"]
            d_tail = b["stats"]["prefill_ms"]["tail_ms"] - c["stats"]["prefill_ms"]["tail_ms"]
            if acc > ACC_GATE:
                v = "UNSAFE"
            elif d_med > BASE_DISPERSION and d_tail > BASE_DISPERSION:
                v = "WIN"
            elif d_med > BASE_DISPERSION:
                v = "PARTIAL"
            else:
                v = "NULL"
            verdicts[f"{mode}_{tower}"] = {"acc_delta": acc, "prefill_median_gain_ms": d_med,
                                           "tail_gain_ms": d_tail, "verdict": v}
            print(f"  {mode}_{tower} vs {base}: top1 delta {acc:.4f} (gate {ACC_GATE})  "
                  f"median gain {d_med:+.2f} ms  tail gain {d_tail:+.2f} ms -> {v}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step033", "n_eval": len(val), "repeats": args.repeats,
                               "seed": args.seed, "device": str(DEVICE), "ckpt": args.ckpt,
                               "dtype": args.dtype, "modes": args.modes, "towers": args.towers,
                               "acc_gate": ACC_GATE, "base_dispersion_ms": BASE_DISPERSION,
                               "verdicts": verdicts, "cells": cells}, indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
