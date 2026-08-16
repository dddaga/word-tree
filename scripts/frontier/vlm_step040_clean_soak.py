"""vlm_step040 -- does a BUILD-ONCE deployment hit the stall? The soak question, asked cleanly.

Section 56: step038 ran `rebuild` and `soak` arms and BOTH drew 0/16. Its soak arm therefore cannot
answer anything -- with its own control clean, "soak clean" is indistinguishable from "the whole run
was suppressed". step039 then reran step037's script byte-unmodified and drew 5/16 at the same rep
indices and magnitudes (0/3/5/10/11; max |delta| 0.03 ms across three processes). The only difference
between step038's rebuild arm and step037 is step038's instrumentation: a `memory_reserved()` call
after EVERY image plus a `memory_stats()` per pass, which takes the caching-allocator lock 500x a pass.
**The probe suppressed the effect it was built to measure.**

So this script re-asks the soak question with NO per-image CUDA calls. `timed_pass` is step037's
verbatim -- per-image prefill comes from `ev.forward()`, which the harness already returns, so nothing
here touches the allocator that step037 did not also touch.

  rebuild -- section 49's protocol: `_dynamo.reset()` / `del ev` / `empty_cache()` per rep. This is the
             CONTROL and it is load-bearing. step038's lesson is that a control which fails to
             reproduce voids the whole experiment's reads, so this arm is checked FIRST and the soak
             arm is only interpretable if it lands at 4-6/16 near the known indices.
  soak    -- build ONCE, run R passes back to back. What a deployed drone does.

PRE-REGISTERED, decided before any draw. Gate carried over VERBATIM from step036/037/039 (>=4.5 ms):
  rebuild NOT in 3-7/16 -> THIS RUN IS VOID. Report the non-replication and interpret nothing else.
    Stated first and on purpose: it is the branch step038 lacked and the reason its reads were unusable.
  rebuild DIRTY + soak CLEAN -> the stall is an EVAL-PROTOCOL ARTIFACT of per-rep rebuilding, like the
    section 44 GraphEval clone. Section 55's drone paragraph is retracted, the >=808 ms per-frame worst
    case does NOT bind a deployed model, and the harness gets a standing note. Power is stated, not
    assumed: a clean soak at R=16 bounds a RATE, never a mechanism (section 51/53).
  rebuild DIRTY + soak DIRTY -> a DEPLOYMENT DEFECT. The >=808 ms per-frame worst case stands as the
    number a real-time claim must survive, and building once is not the fix.
Whichever fires, the soak arm's departure INDICES are reported: if soak departs at rep indices unlike
the rebuild arm's, the two are not the same phenomenon and neither branch above applies cleanly.
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
parser.add_argument("--arms", nargs="+", default=["rebuild", "soak"],
                    help="rebuild FIRST -- it is the control and the run is void without it")
parser.add_argument("--mode", default="compile", help="section 55/56 measured the stall in this cell")
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--repeats", type=int, default=16)
parser.add_argument("--control_repeats", type=int, default=None,
                    help="rebuild-arm R; defaults to --repeats (reproduces step040 exactly). Set "
                         "below --repeats to buy soak depth -- control stays IN-PROCESS regardless")
parser.add_argument("--n_eval", type=int, default=500)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag_suffix", default="")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.warmup, args.repeats = 20, 3, 2

SLOT = os.environ.get("SGN_SLOT", "local")
TAG = f"{args.dtype}_{'-'.join(args.arms)}_r{args.repeats}_n{args.n_eval}{args.tag_suffix}"
OUT = ROOT / "results" / "frontier" / f"vlm_step040_clean_soak_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
CKPT = ROOT / "results" / "frontier" / args.ckpt
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}
GRAPH_MODES = {"reduce-overhead", "max-autotune"}
# Carried over VERBATIM from step036/037/038/039 so no arm can retune a threshold to a count.
DEPARTURE_MS = {"compile": 4.5, "reduce-overhead": 3.0}
STALL_MS = 50.0  # base images run 4.0-5.5 ms, spikes 246-809 ms. No middle ground to argue about.
KNOWN_INDICES = [0, 3, 5, 10, 11]  # step036/037/039 agree on these; the control is checked against it
CONTROL_MIN, CONTROL_MAX = 3, 7     # rebuild must land here or the run is VOID
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def stat(xs):
    """Median with MIN-MAX. Section 49: IQR discards the one-sided tail this line exists to measure."""
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"median": med, "min": s[0], "max": s[-1], "tail_ms": s[-1] - med}


def timed_pass(ev, val, warmup):
    """step037's pass, VERBATIM. No CUDA calls beyond what step037 already made -- that is the point."""
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
    stalls = [{"idx": i, "ms": p} for i, p in enumerate(per_image) if p >= STALL_MS]
    return {"top1": ok / n, "vision_ms": vis / n, "prefill_ms": sum(per_image) / n,
            "prefill_per_image": per_image, "prefill_img_median": stat(per_image)["median"],
            "prefill_img_max": max(per_image), "stalls": stalls,
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
    print(f"{'=' * 78}\nvlm_step040 clean soak  device={DEVICE}  mode={args.mode}  arms={args.arms}  "
          f"R={args.repeats}  n_eval={args.n_eval}  (NO per-image CUDA instrumentation)", flush=True)
    assert CKPT.exists(), f"missing student checkpoint: {CKPT}"
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    sd = torch.load(CKPT, map_location="cpu")
    gate, cells = DEPARTURE_MS[args.mode], {}

    for arm in args.arms:
        runs, ev, st = [], None, None
        R = (args.control_repeats or args.repeats) if arm == "rebuild" else args.repeats
        for rep in range(R):
            if arm == "rebuild" or ev is None:
                ev, st = build(args.mode, sd)
            r = timed_pass(ev, val, args.warmup)
            r["departure"] = r["prefill_ms"] >= gate
            runs.append(r)
            s = r["stalls"][0] if r["stalls"] else None
            print(f"  {arm} rep{rep}  top1 {r['top1']:.4f}  tower {r['vision_ms']:6.2f}  "
                  f"prefill {r['prefill_ms']:6.2f} (img med {r['prefill_img_median']:5.2f}  "
                  f"img max {r['prefill_img_max']:7.2f})  wall {r['wall_s']:.1f}s  "
                  f"load1 {r['load1']:.2f}"
                  + (f"  <== DEPARTURE, stall img{s['idx']} {s['ms']:.0f} ms" if s else ""), flush=True)
            if arm == "rebuild":
                del ev
                ev = None
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        idx = [i for i, r in enumerate(runs) if r["departure"]]
        cells[arm] = {"n_params": n_params(st), "gate_ms": gate, "stall_ms": STALL_MS,
                      "departures": len(idx), "departure_indices": idx, "repeats": R,
                      "runs": runs, "n_stalls": sum(len(r["stalls"]) for r in runs),
                      "stats": {m: stat([x[m] for x in runs])
                                for m in ("vision_ms", "prefill_ms", "end2end_ms")}}
        print(f"    {arm}: {len(idx)}/{R} departures (gate {gate} ms) at reps {idx}; "
              f"known indices {KNOWN_INDICES}", flush=True)

    ctl = cells.get("rebuild")
    assert ctl is None or args.smoke_test or ctl["repeats"] == 16, "band [3,7] assumes R=16"
    if ctl is not None and not CONTROL_MIN <= ctl["departures"] <= CONTROL_MAX:
        print(f"\n  *** RUN VOID: control arm drew {ctl['departures']}/{ctl['repeats']}, outside "
              f"{CONTROL_MIN}-{CONTROL_MAX}. A control that fails to reproduce voids the reads "
              f"(section 56). Interpret NOTHING about the soak arm from this run.", flush=True)
    elif ctl is not None and "soak" in cells:
        sk, sr = cells["soak"]["departures"], cells["soak"]["repeats"]
        print(f"\n  control OK ({ctl['departures']}/{ctl['repeats']} at {ctl['departure_indices']}). "
              + (f"soak {sk}/{sr} ({sr * len(val)} frames): EVAL-PROTOCOL ARTIFACT -- bounds a RATE, "
                 f"not a mechanism." if sk == 0 else
                 f"soak {sk}/{sr} at {cells['soak']['departure_indices']}: DEPLOYMENT "
                 f"DEFECT -- building once is not the fix."), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step040", "n_eval": len(val), "repeats": args.repeats,
                               "seed": args.seed, "device": str(DEVICE), "ckpt": args.ckpt,
                               "dtype": args.dtype, "arms": args.arms, "mode": args.mode,
                               "ratio": args.ratio, "departure_gate_ms": gate,
                               "stall_gate_ms": STALL_MS, "known_indices": KNOWN_INDICES,
                               "control_band": [CONTROL_MIN, CONTROL_MAX],
                               "cells": cells}, indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
