"""vlm_step032 -- how repeatable is this harness's latency number, actually?

Section 47 (VLM_TRAJECTORY_part14) found the ship cell's e2e moving 7.95 -> 9.41 ms (+18%) between
two clean runs of an identical config, and diagnosed it from an internal control: the language model
is byte-identical between the teacher and student cells, so a within-pair prefill difference is pure
instrument error. That control put the floor near +/-0.5 ms absolute -- ~4% of a 13 ms fp32 prefill
but ~25% of a 4-5 ms compiled one, i.e. the 5.0 ms verdict bar sits inside its own noise.

This script measures the floor directly instead of inferring it, with the three changes section 47
pre-registered: (a) R repeats of every cell inside ONE process, reported as median + IQR, so no
point estimate is ever quoted again; (b) CPU load and wall-clock logged per repeat, because an empty
GPU is demonstrably not a sufficient validity gate for a dispatch-bound stage; (c) the
teacher-minus-student prefill delta promoted to an automatic gate -- it is zero by construction and
would have flagged both the step030 duplicate and the ship-cell spread before anything was written
down.

PRE-REGISTERED: max relative IQR over cells <= 5% -> the 7.95-9.41 range collapses to a point and
section 47's WIN is restored (or retired) on evidence. > 5% -> absolute latency on this box is not
publishable and the paper quotes RATIOS with intervals, never milliseconds.
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
from scripts.frontier.vlm_width import build_width_student, n_params

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--student_depth", type=int, default=6)
parser.add_argument("--ratio", type=float, default=0.25)
parser.add_argument("--ckpt", default="vlm_step023_mlp_r0.25_r0.25_e50_t9352_n500_d6.pt")
parser.add_argument("--cells", nargs="+", default=["fp32:eager", "bf16:compile"],
                    help="dtype:mode pairs; each runs teacher and student")
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--n_eval", type=int, default=500)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag_suffix", default="")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.warmup, args.repeats, args.cells = 20, 3, 2, ["fp32:eager"]

SLOT = os.environ.get("SGN_SLOT", "local")
TAG = f"{'-'.join(c.replace(':', '') for c in args.cells)}_r{args.repeats}_n{args.n_eval}{args.tag_suffix}"
OUT = ROOT / "results" / "frontier" / f"vlm_step032_repeat_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
CKPT = ROOT / "results" / "frontier" / args.ckpt
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}
IQR_BAR = 5.0          # percent of median; the pre-registered publishability threshold
PREFILL_GATE_MS = 0.5  # teacher-vs-student prefill delta is zero by construction; flag above this
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def quart(xs):
    """Median and IQR by linear interpolation. R is small (5-9), so no numpy dependency is worth it."""
    s = sorted(xs)

    def q(p):
        i = p * (len(s) - 1)
        lo = int(i)
        return s[lo] + (i - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])
    med = q(0.5)
    return med, q(0.75) - q(0.25), 100.0 * (q(0.75) - q(0.25)) / med if med else float("nan")


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


def build(dtype, mode):
    torch._dynamo.reset()
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=DTYPE[dtype])
    ev = VLMEval(m.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    if mode == "compile":
        ev.m.model.vision_model = torch.compile(ev.m.model.vision_model, dynamic=False)
        ev.m.model.text_model = torch.compile(ev.m.model.text_model, dynamic=False)
    return ev


def main():
    print("=" * 78, flush=True)
    print(f"vlm_step032 repeatability  device={DEVICE}  cells={args.cells}  R={args.repeats}  "
          f"n_eval={args.n_eval}  student=d{args.student_depth} x r{args.ratio}", flush=True)
    assert CKPT.exists(), f"missing student checkpoint: {CKPT}"
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    sd = torch.load(CKPT, map_location="cpu")
    cells = {}

    for spec in args.cells:
        dtype, mode = spec.split(":")
        for tower in ("teacher", "student"):
            runs = []
            for rep in range(args.repeats):
                # Rebuilt every repeat, not just every cell: a repeat that reuses the compiled
                # object would measure one compile session R times and report its variance as zero.
                ev = build(dtype, mode)
                inter = ev.full[0].mlp.fc1.out_features
                if tower == "teacher":
                    ev.set_depth(args.depth)
                    pr = n_params(nn.ModuleList(list(ev.full[:args.depth])))
                else:
                    st = build_width_student(ev, args.student_depth,
                                             int(round(inter * args.ratio)), "cpu")
                    st.load_state_dict(sd)
                    ev.set_layers(st.to(DEVICE).to(DTYPE[dtype]))
                    pr = n_params(st)
                r = timed_pass(ev, val, args.warmup)
                runs.append(r)
                print(f"  {dtype}_{mode}_{tower} rep{rep}  top1 {r['top1']:.4f}  "
                      f"tower {r['vision_ms']:6.2f}  prefill {r['prefill_ms']:6.2f}  "
                      f"e2e {r['end2end_ms']:6.2f} ms  wall {r['wall_s']:.1f}s  "
                      f"load1 {r['load1']:.2f}", flush=True)
                del ev
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
            key = f"{dtype}_{mode}_{tower}"
            stat = {m: dict(zip(("median", "iqr", "iqr_pct"),
                                quart([x[m] for x in runs])))
                    for m in ("vision_ms", "prefill_ms", "end2end_ms")}
            cells[key] = {"runs": runs, "stats": stat, "tower_params": pr,
                          "top1": [x["top1"] for x in runs]}
            for m in ("vision_ms", "prefill_ms", "end2end_ms"):
                s = stat[m]
                print(f"    {key:24s} {m:11s} median {s['median']:6.2f}  "
                      f"IQR {s['iqr']:5.2f} ({s['iqr_pct']:5.1f}%)", flush=True)

    # Gate (c): the LM is byte-identical across towers, so prefill must not depend on the tower.
    gates = {}
    for spec in args.cells:
        d, mo = spec.split(":")
        t, s = cells.get(f"{d}_{mo}_teacher"), cells.get(f"{d}_{mo}_student")
        if t and s:
            delta = abs(t["stats"]["prefill_ms"]["median"] - s["stats"]["prefill_ms"]["median"])
            gates[spec] = {"prefill_delta_ms": delta, "pass": bool(delta <= PREFILL_GATE_MS)}
            print(f"  GATE prefill-invariance {spec}: |teacher-student| = {delta:.2f} ms "
                  f"-> {'PASS' if delta <= PREFILL_GATE_MS else 'FAIL'} (bar {PREFILL_GATE_MS})",
                  flush=True)

    worst = max((c["stats"][m]["iqr_pct"] for c in cells.values()
                 for m in ("vision_ms", "prefill_ms", "end2end_ms")), default=float("nan"))
    verdict = "RESOLVABLE" if worst <= IQR_BAR else "NOT-RESOLVABLE"
    print(f"\n  worst relative IQR over all cells/stages {worst:.1f}% vs bar {IQR_BAR}% "
          f"-> {verdict}", flush=True)
    print("  RESOLVABLE -> quote medians with IQR; NOT-RESOLVABLE -> quote RATIOS only, never ms.",
          flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step032", "n_eval": len(val), "repeats": args.repeats,
                               "seed": args.seed, "device": str(DEVICE), "ckpt": args.ckpt,
                               "student_depth": args.student_depth, "ratio": args.ratio,
                               "iqr_bar_pct": IQR_BAR, "worst_iqr_pct": worst,
                               "prefill_gates": gates, "verdict": verdict,
                               "cells": cells}, indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
