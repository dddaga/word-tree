"""vlm_step043 -- END-TO-END FRAME latency, not model latency. What does preprocessing actually cost?

Section 58: a 500-image pass is ~58 s wall, of which prefill is 2.1 s and the vision tower 2.0 s --
**~54 s, 93%, is unaccounted HOST work**, ~108 ms/image. The headline `e2e 6.76 ms` is MODEL-ONLY and
was quoted as a drone number throughout sections 44-56 without that qualifier. Section 57 closed the
stall as an eval-protocol artifact, so preprocessing is the only unresolved latency item left.

Three arms, one run, same 500 images, same student:
  disk -- what the harness does today. Times decode / processor / H2D / tower / prefill separately.
  ram  -- images pre-decoded to PIL ONCE before timing. Isolates JPEG decode by removing it; a
          camera-fed drone never decodes a JPEG, so this is the honest floor for the CPU path.
  gpu  -- uint8 HWC uploaded raw, then resize + rescale + normalize ON DEVICE. Upload shrinks 4x
          (uint8 vs fp32) and the per-pixel math moves to the card.

EQUIVALENCE IS CHECKED, NOT ASSUMED. `gpu` reimplements the processor, so it reports max abs diff vs
the CPU `pixel_values` on the first `--equiv_n` images AND its own top1; section 44's `GraphEval`
clone is the precedent for what an unverified reimplementation costs. diff <= --equiv_tol AND top1
matching the disk arm -> drop-in, budget quotable. Otherwise latency is a FLOOR with the accuracy
claim WITHHELD, in those words -- a faster non-equivalent path is not a result.

PRE-REGISTERED: this does not move the efficiency claim (21.28M params, 42.6 MB bf16, 8.00x fewer
bytes -- counted, not timed, independent of preprocessing). It fixes the latency SENTENCE: the paper
quotes whichever arm matches the deployment story, named as that arm, and `6.76 ms` is never again
quoted unqualified. **Run ALONE** -- a co-tenant on the card invalidates every number here.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, VLMEval, sample_images
from scripts.frontier.vlm_width import build_width_student, n_params

parser = argparse.ArgumentParser()
P = parser.add_argument
P("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct"); P("--student_depth", type=int, default=6)
P("--ratio", type=float, default=0.25); P("--dtype", default="bf16"); P("--tag_suffix", default="")
P("--ckpt", default="vlm_step023_mlp_r0.25_r0.25_e50_t9352_n500_d6.pt"); P("--seed", type=int, default=42)
P("--arms", nargs="+", default=["disk", "ram", "gpu"]); P("--n_eval", type=int, default=500)
P("--warmup", type=int, default=20); P("--equiv_n", type=int, default=8)
P("--equiv_tol", type=float, default=1e-2); P("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.warmup, args.equiv_n = 20, 3, 2

SLOT = os.environ.get("SGN_SLOT", "local")
TAG = f"{args.dtype}_{'-'.join(args.arms)}_n{args.n_eval}{args.tag_suffix}"
OUT = ROOT / "results" / "frontier" / f"vlm_step043_frame_latency_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
CKPT = ROOT / "results" / "frontier" / args.ckpt
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def stat(xs):
    """Median with MIN-MAX. Section 49: IQR discards the one-sided tail this line exists to measure."""
    s, n = sorted(xs), len(xs)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"median": med, "min": s[0], "max": s[-1], "mean": sum(s) / n}


def gpu_cfg(ev):
    """Pull the processor's OWN constants -- never hardcode what the config already states."""
    ip, mk = ev.p.image_processor, lambda v: torch.tensor(v, device=DEVICE).view(1, -1, 1, 1)
    side = ip.size.get("longest_edge") or ip.size.get("height") or max(ip.size.values())
    return int(side), float(ip.rescale_factor), mk(ip.image_mean), mk(ip.image_std)


def gpu_pixels(u8, side, scale, mean, std):
    """uint8 HWC on host -> normalized fp32 NCHW on device. Upload is uint8, math is on the card."""
    x = u8.to(DEVICE, non_blocking=True).permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False, antialias=True)
    return ((x * scale) - mean) / std


def load_u8(path):
    img = Image.open(path).convert("RGB")
    return img, torch.from_numpy(np.asarray(img).copy())


def run_arm(ev, arm, val, cfg):
    """One pass. Every stage timed separately; device stages are sync-bracketed like vlm_eval.

    The `gpu` arm builds the TEXT batch once, outside the loop -- not a shortcut: the prompt is fixed
    and `do_image_splitting=False` fixes the image-token count at 64, so input_ids are bit-identical
    across the pass, and a deployment tokenizes its fixed prompt once too. Per image the arm pays
    only the uint8 upload plus on-device resize/rescale/normalize.
    """
    side, scale, mean, std = cfg
    pre = None
    if arm in ("ram", "gpu"):  # decode ONCE, outside the timed loop -- that is the point of the arm
        pre = [load_u8(p) for p, _ in val]
    fixed = None
    if arm == "gpu":
        f = ev.p(text=ev.prompt, images=[pre[0][0]], return_tensors="pt", do_image_splitting=False)
        fixed = {k: v.to(DEVICE) for k, v in f.items()}
    t, ok = {k: [] for k in ("decode", "proc", "h2d", "tower", "prefill")}, 0
    for i, (path, wnid) in enumerate(val):
        t0 = time.perf_counter()
        img, u8 = pre[i] if pre is not None else load_u8(path)
        t["decode"].append((time.perf_counter() - t0) * 1000)

        if arm == "gpu":
            t["proc"].append(0.0)  # amortized: fixed prompt, fixed token count. See docstring.
            sync(); t0 = time.perf_counter()
            b = dict(fixed)
            b["pixel_values"] = gpu_pixels(u8, side, scale, mean, std)[None]
            sync(); t["h2d"].append((time.perf_counter() - t0) * 1000)
        else:
            t0 = time.perf_counter()
            b = ev.p(text=ev.prompt, images=[img], return_tensors="pt", do_image_splitting=False)
            t["proc"].append((time.perf_counter() - t0) * 1000)
            sync(); t0 = time.perf_counter()
            b = {k: v.to(DEVICE) for k, v in b.items()}
            sync(); t["h2d"].append((time.perf_counter() - t0) * 1000)

        feats, v = ev.encode(b)
        emb, _ = ev.merge(b, feats)
        _, pred, p = ev.forward(emb)
        ok += LABELS[pred] == WNID2LABEL[wnid]
        t["tower"].append(v)
        t["prefill"].append(p)
    n, s = len(val), {k: stat(v) for k, v in t.items()}
    frame = [sum(t[k][i] for k in t) for i in range(n)]
    return {"top1": ok / n, "stages": s, "frame_ms": stat(frame),
            "model_only_ms": s["tower"]["median"] + s["prefill"]["median"],
            "preproc_ms": s["decode"]["median"] + s["proc"]["median"] + s["h2d"]["median"]}


def equivalence(ev, val, cfg):
    """max abs diff between the GPU path and the processor's own pixel_values. Section 44's lesson."""
    d = 0.0
    for path, _ in val[:args.equiv_n]:
        img, u8 = load_u8(path)
        ref = ev.p(text=ev.prompt, images=[img], return_tensors="pt",
                   do_image_splitting=False)["pixel_values"].to(DEVICE).float()
        d = max(d, (ref - gpu_pixels(u8, *cfg).reshape(ref.shape)).abs().max().item())
    return d


def main():
    print(f"{'=' * 78}\nvlm_step043 frame latency  device={DEVICE}  arms={args.arms}  "
          f"n_eval={args.n_eval}", flush=True)
    assert CKPT.exists(), f"missing student checkpoint: {CKPT}"
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=DTYPE[args.dtype])
    ev = VLMEval(m.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    inter = int(round(ev.full[0].mlp.fc1.out_features * args.ratio))
    st = build_width_student(ev, args.student_depth, inter, "cpu")
    st.load_state_dict(torch.load(CKPT, map_location="cpu"))
    ev.set_layers(st.to(DEVICE).to(DTYPE[args.dtype]))
    cfg = gpu_cfg(ev)
    for path, _ in val[:args.warmup]:
        ev.run(Image.open(path).convert("RGB"))
    diff = equivalence(ev, val, cfg) if "gpu" in args.arms else None
    cells = {}
    for arm in args.arms:
        r = cells[arm] = run_arm(ev, arm, val, cfg)
        g = "  ".join(f"{k} {v['median']:.2f}" for k, v in r["stages"].items())
        print(f"  {arm:5s} top1 {r['top1']:.4f}  {g}  | FRAME {r['frame_ms']['median']:.2f} ms "
              f"(max {r['frame_ms']['max']:.1f})  model-only {r['model_only_ms']:.2f}", flush=True)

    if diff is not None:
        eq = diff <= args.equiv_tol and cells["gpu"]["top1"] == cells.get("disk", cells["gpu"])["top1"]
        print(f"  gpu equivalence: max|diff| {diff:.4g} (tol {args.equiv_tol}) -> "
              + ("DROP-IN, budget quotable" if eq else
                 "NOT EQUIVALENT -- latency is a FLOOR, accuracy claim WITHHELD"), flush=True)
        cells["gpu"]["equiv_max_abs_diff"], cells["gpu"]["equivalent"] = diff, bool(eq)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step043", "n_eval": len(val), "seed": args.seed,
                               "device": str(DEVICE), "ckpt": args.ckpt, "dtype": args.dtype,
                               "arms": args.arms, "ratio": args.ratio, "n_params": n_params(st),
                               "equiv_tol": args.equiv_tol, "cells": cells}, indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
