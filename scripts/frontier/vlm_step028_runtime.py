"""vlm_step028: is the latency wall ARCHITECTURAL or RUNTIME? bf16 x torch.compile, no training.

WHY THIS ARM. Section 44 measured every stage against the fp32 roofline and found the pipeline's two
halves are bound by DIFFERENT things: the vision tower runs at 50-62% of roofline (compute-bound --
which is why step025/026's FLOP cuts bought real wall-time), while LM prefill runs at 10.5%, i.e.
~90% of its 14.03 ms is kernel-launch and Python dispatch across 30 thin layers at batch 1 and seq
~90, not arithmetic. step022 confirmed this independently without meaning to: varying ONLY the image
token count, prefill went 14.10 ms at 64 tokens -> 13.90 ms at 4 tokens, so a 16x sequence-length cut
bought 1.4%. A compute-bound prefill cannot do that.

The consequence is that end-to-end is CAPPED at ~2.8x with prefill pinned near 14 ms, however far the
tower is squeezed -- step025 reached 1.61x of that cap and step026's much harder cut only 1.70x. So
the tower is finished as a LATENCY axis (it remains the params/bytes axis, which is what the drone
goal actually cares about), and the next latency win must be a RUNTIME change. A stage at 10.5% of
roofline is slow because of how it is EXECUTED, not how it is DESIGNED.

WHY IT MAY BE COMPOUNDED WITHOUT AN ISOLATION ABLATION. bf16 and compilation change dispatch cost and
memory traffic; the step025 cut changed layer count and MLP width. Disjoint mechanisms on disjoint
signal paths, so the compounding rule is satisfied. This arm nevertheless measures BOTH towers under
ALL runtimes rather than assuming it -- the 2x2 grid is what makes the orthogonality measured.

NO TRAINING. Inference-only re-timing of the teacher and the d6 x r=0.25 student, reusing step027's
checkpoint. Same eval protocol, same 500-image sample, same seed, so top-1 stays comparable to every
number in the line and a bf16 accuracy regression cannot hide. The r=0.25 student is used rather
than step025's r=0.5 one because section 44 moved the drone operating point there (0.2502x tower
params at -2.00pp, McNemar p=0.4437); the VERDICT is stated on prefill and is tower-independent
either way, but the tower cells should be timed on the tower we actually intend to ship.

PRE-REGISTERED GATES (fixed BEFORE the run):
  (1) VALIDITY. The fp32-eager teacher cell must reproduce top1 0.7040 and roughly 26.2 ms tower /
      14.0 ms prefill. This cell is a pure replay of step023/024/025/026's teacher; if it moves, the
      runtime harness is the variable and nothing else in the grid is interpretable.
  (2) ACCURACY GUARD, per cell. bf16 is a numerics change, so every cell reports top-1 and any cell
      more than 2.0pp below its own fp32-eager counterpart is reported as an accuracy REGRESSION no
      matter what it did to latency. Speed bought by broken numerics is not a win.
VERDICT RULE, fixed in advance, stated on PREFILL because prefill is the thing section 44 diagnosed:
  WIN      iff best-runtime prefill < 5.0 ms. The overhead diagnosis is CONFIRMED, the end-to-end cap
           lifts from ~2.8x to ~7x, and further tower work becomes worth doing again.
  PARTIAL  iff prefill lands in [5.0, 10.0) ms -- overhead is real but only partly removable by these
           two levers, and the residual needs naming (CUDA graphs, ONNX/TensorRT export, batching).
  LOSS     iff prefill >= 10.0 ms. The overhead diagnosis is FALSIFIED: prefill is bound by something
           structural, section 44's consequence 3 is wrong, and the roofline reading needs revisiting.
This is a real falsifiable prediction -- a LOSS overturns the section that motivated the arm.

Output: results/frontier/vlm_step028_runtime_{TAG}__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, VLMEval, sample_images
from scripts.frontier.vlm_width import build_width_student, n_params

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=12, help="teacher depth")
parser.add_argument("--student_depth", type=int, default=6)
parser.add_argument("--ratio", type=float, default=0.25, help="student MLP width ratio (step027 arm)")
parser.add_argument("--ckpt", default="vlm_step023_mlp_r0.25_r0.25_e50_t9352_n500_d6.pt",
                    help="step027 student checkpoint, relative to results/frontier/")
parser.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"])
parser.add_argument("--compile", nargs="+", default=["eager", "compile"])
parser.add_argument("--n_eval", type=int, default=500)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag_suffix", default="", help="appended to the output TAG (step029 re-timing)")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.warmup, args.dtypes, args.compile = 20, 3, ["fp32"], ["eager"]

SLOT = os.environ.get("SGN_SLOT", "local")
TAG = f"{'-'.join(args.dtypes)}_{'-'.join(args.compile)}_n{args.n_eval}{args.tag_suffix}"
OUT = ROOT / "results" / "frontier" / f"vlm_step028_runtime_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
CKPT = ROOT / "results" / "frontier" / args.ckpt
ANCHOR, GATE_PP, ACC_GUARD_PP = 0.7040, 2.0, 2.0
WIN_MS, PARTIAL_MS = 5.0, 10.0
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}


def pick_device(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device(args.device)


def evaluate(ev, val, warmup):
    """Warm up (compile traces, autotune, clocks) BEFORE the timed pass, else the first images pay
    compilation and the 'compile' cells report the cost of compiling instead of the cost of running."""
    for path, _ in val[:warmup]:
        ev.run(Image.open(path).convert("RGB"))
    ok = vis = pre = 0.0
    correct = []
    for path, wnid in val:
        b = ev.batch(Image.open(path).convert("RGB"))
        feats, v = ev.encode(b)
        emb, _ = ev.merge(b, feats)
        _, pred, p = ev.forward(emb)
        hit = LABELS[pred] == WNID2LABEL[wnid]
        correct.append(int(hit))
        ok += hit; vis += v; pre += p
    n = len(val)
    return {"top1": ok / n, "vision_ms": vis / n, "prefill_ms": pre / n,
            "end2end_ms": (vis + pre) / n, "n": n, "correct": correct}


def build(dtype, mode):
    """Fresh model per runtime cell. Rebuilt rather than mutated because torch.compile installs
    wrappers and dtype casts are lossy -- reusing one object would leak state between cells."""
    # Dynamo's compile cache is keyed on the CODE OBJECT, which is shared by every cell, so guard
    # variants accumulate across cells and trip config.recompile_limit -- in step028 that happened
    # on the third compile cell and silently degraded the two bf16 ones. Each cell builds a fresh
    # model, so carrying the cache between them was never correct. See section 45 caveat 2.
    torch._dynamo.reset()
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=DTYPE[dtype])
    ev = VLMEval(m.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    if mode == "compile":
        # Only the two stages that are TIMED are compiled. The prompt and token count are fixed, so
        # every prefill sees one static shape and there is no recompilation inside the timed pass.
        ev.m.model.vision_model = torch.compile(ev.m.model.vision_model, dynamic=False)
        ev.m.model.text_model = torch.compile(ev.m.model.text_model, dynamic=False)
    return ev


def main():
    print("=" * 78, flush=True)
    print(f"vlm_step028 runtime  device={DEVICE}  dtypes={args.dtypes}  modes={args.compile}  "
          f"n_eval={args.n_eval}  student=d{args.student_depth} x r{args.ratio}", flush=True)
    assert CKPT.exists(), f"missing step027 checkpoint: {CKPT}"
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    sd = torch.load(CKPT, map_location="cpu")
    cells, base = {}, None

    for dtype in args.dtypes:
        for mode in args.compile:
            ev = build(dtype, mode)
            inter = ev.full[0].mlp.fc1.out_features
            for tower in ("teacher", "student"):
                if tower == "teacher":
                    ev.set_depth(args.depth)
                    pr = n_params(nn.ModuleList(list(ev.full[:args.depth])))
                else:
                    st = build_width_student(ev, args.student_depth,
                                             int(round(inter * args.ratio)), "cpu")
                    st.load_state_dict(sd)
                    ev.set_layers(st.to(DEVICE).to(DTYPE[dtype]))
                    pr = n_params(st)
                key = f"{dtype}_{mode}_{tower}"
                r = evaluate(ev, val, args.warmup)
                r["tower_params"] = pr
                if dtype == "fp32" and mode == "eager" and tower == "teacher":
                    base = r
                    d = abs(r["top1"] - ANCHOR) * 100
                    print(f"  GATE validity  fp32 eager teacher top1 {r['top1']:.4f} vs {ANCHOR} "
                          f"-> {'PASS' if d <= GATE_PP else 'FAIL'}", flush=True)
                ref = cells.get(f"fp32_eager_{tower}")
                r["acc_regression"] = bool(ref and (ref["top1"] - r["top1"]) * 100 > ACC_GUARD_PP)
                cells[key] = r
                print(f"  {key:24s} top1 {r['top1']:.4f}  tower {r['vision_ms']:6.2f}  "
                      f"prefill {r['prefill_ms']:6.2f}  e2e {r['end2end_ms']:6.2f} ms  "
                      f"params {pr/1e6:.2f}M{'  ACC-REGRESSION' if r['acc_regression'] else ''}",
                      flush=True)
            del ev
            if DEVICE.type == "cuda": torch.cuda.empty_cache()

    best = min(c["prefill_ms"] for k, c in cells.items() if not c["acc_regression"])
    verdict = ("WIN" if best < WIN_MS else "PARTIAL" if best < PARTIAL_MS else "LOSS")
    print(f"\n  best prefill (accuracy-clean cells only) {best:.2f} ms vs "
          f"fp32-eager {base['prefill_ms']:.2f} ms -> {verdict}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step028", "anchor": ANCHOR, "n_eval": len(val),
                               "seed": args.seed, "device": str(DEVICE), "ckpt": args.ckpt,
                               "student_depth": args.student_depth, "ratio": args.ratio,
                               "best_prefill_ms": best, "verdict": verdict,
                               "cells": cells}, indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
