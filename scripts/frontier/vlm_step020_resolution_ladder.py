"""vlm_step020: INPUT RESOLUTION as the drone latency lever the trajectory never pulled.

WHY THIS AND NOT MORE lm_head WORK. part 8 §32 measured the int8 readout end-to-end and it moved
ZERO milliseconds (0.94x vs fp16 -- slower). step006 says why: on the SAME harness, vision_ms is
64.85 of 79.18 ms total at d12 (82%) and 35.05 of 49.59 at d6 (71%). The tower is the bottleneck
and depth is the only tower lever pulled so far (settled at d6, part 7 §19).

THE LEVER. SigLIP here is patch 16 on a 512 grid = 1024 patches, and the connector pools 4x4, so
the token count the DECODER sees is patches/16. Shrinking the pixel grid therefore cuts BOTH terms
at once -- unlike depth, which only cuts the tower:
    512 -> 1024 patches -> 64 tokens      256 -> 256 patches -> 16 tokens
    384 ->  576 patches -> 36 tokens      192 -> 144 patches ->  9 tokens
                                          128 ->   64 patches ->  4 tokens
Measured on the 5060ti (drone proxy, fp32, tower only, 30 iters after 8 warmup):
    res  512    384    256    192    128
    d12  24.68  13.71  8.10   6.27   3.83   ms
    d6   12.63   7.04  4.21   3.30   2.09   ms
So d6@256 is 5.9x under d12@512 on the tower. Scaling is SUBLINEAR in patches (4x the patches
costs 3x the time), i.e. there is a fixed-overhead floor below ~192 -- expect diminishing returns
at the bottom of the ladder, not a free fall.

WHAT THIS SCRIPT ADDS. Latency was never in doubt; ACCURACY is. This runs the ladder end-to-end
through the real constrained-choice protocol and reports top-1 against the d12@512 anchor.

ONE VARIABLE. Depth is held at 12 STOCK -- no distilled checkpoint anywhere in this experiment.
That is deliberate: the d6 tower was distilled at 512, so feeding it 256 would confound
"resolution hurts" with "the student is off-distribution", which is a DIFFERENT question (and the
one vlm_step011/012 are already answering). Compose with depth only after resolution is
characterised alone. Per the compounding rule: resolution and depth both sit on the tower's
signal path, so the 2x2 gets run, never assumed.

HOW THE GRID IS SHRUNK. The processor pads to a fixed 512x512 whatever `size` you pass (verified:
longest_edge 512/384/256/192/128 all return a 512x512 tensor), so resolution CANNOT be set through
the processor. It is set by bilinear-downsampling the processor's own normalised output. That
keeps mean/std preprocessing identical across arms, so the only thing that changes between cells
is the pixel grid the tower sees. SigLIP interpolates its position embeddings, so every size runs.

PRE-REGISTERED GATES (fixed BEFORE the run, per the core tenet):
  (1) VALIDITY. The res=512 d12 cell must land within +-2.0pp of step006's 0.711. It is the same
      config by construction, so a miss means the plumbing below is wrong and NOTHING else in the
      file may be quoted.
  (2) TOKEN-COUNT. Each cell asserts connector tokens == (res/16)^2/16 and that exactly that many
      image slots survive in the merged sequence. A silent mismatch would make a fast cell look
      good for the wrong reason.
This is a TIER 0 SCOUT: a rejection filter, not a top-N selector. Every resolution with a
positive-or-neutral delta advances to T1, and the reported number is a scout number at reduced n.

Output: results/frontier/vlm_step020_resolution_ladder_{TAG}__{SLOT}.json
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

from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, VLMEval, sample_images, sync

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=12, help="STOCK depth; 12 keeps this unconfounded")
parser.add_argument("--res", type=int, nargs="+", default=[512, 384, 256, 192, 128])
parser.add_argument("--n_eval", type=int, default=500, help="T0 scout budget")
parser.add_argument("--seed", type=int, default=42, help="42 reproduces step004/006's eval set")
parser.add_argument("--split", default="val")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.res = 20, [512, 256]

SLOT = os.environ.get("SGN_SLOT", "local")
TAG = f"d{args.depth}_" + "-".join(f"r{r}" for r in args.res) + f"_n{args.n_eval}"
OUT = ROOT / "results" / "frontier" / f"vlm_step020_resolution_ladder_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
ANCHOR = 0.711          # step006 d12 top-1, the validity gate target
GATE_PP = 2.0


def pick_device(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device(args.device)


class ResEval(VLMEval):
    """VLMEval with a settable pixel grid.

    Two overrides, both minimal. `encode` downsamples the processor's normalised pixels to
    self.res before the tower runs -- that is the whole intervention. `merge` then splices the
    resulting T' features into the FIRST T' image slots and drops the remaining (64 - T') slots
    from the sequence, reusing the same drop-mask shape as the parent's `prune`. The prompt is
    built by the processor and always carries 64 image placeholders, so without this the parent's
    `assert len(pos) == len(feats)` would fire at every resolution below 512.
    """

    res = 512

    def encode(self, b, grad=False):
        sync(self.dev); t0 = time.perf_counter()
        px = b["pixel_values"]
        if self.res != px.shape[-1]:
            sh = px.shape
            px = F.interpolate(px.reshape(-1, *sh[-3:]), size=(self.res, self.res),
                               mode="bilinear", align_corners=False, antialias=True)
            px = px.reshape(*sh[:-3], *px.shape[-3:])
        with torch.set_grad_enabled(grad):
            lh = self.m.model.get_image_features(pixel_values=px).last_hidden_state
            feats = self.m.model.connector(lh)
        sync(self.dev)
        return feats.reshape(-1, feats.shape[-1]), (time.perf_counter() - t0) * 1000

    def merge(self, b, feats):
        ids = b["input_ids"]
        emb = self.m.get_input_embeddings()(ids).clone()
        pos = (ids[0] == self.img_id).nonzero(as_tuple=True)[0]
        t = len(feats)
        assert t <= len(pos), f"{t} feats > {len(pos)} slots"
        emb[0, pos[:t]] = feats.to(emb.dtype)
        if t == len(pos): return emb, pos
        keep = torch.ones(emb.shape[1], dtype=torch.bool, device=self.dev)
        keep[pos[t:]] = False
        return emb[:, keep], pos[:t]


def cell(ev, res, val):
    """One resolution. Returns per-image records + timing means."""
    ev.res = res
    want = (res // 16) ** 2 // 16
    rec, t0 = [], time.perf_counter()
    for i, (path, wnid) in enumerate(val):
        img = Image.open(path).convert("RGB")
        b = ev.batch(img)
        feats, vis_ms = ev.encode(b)
        assert len(feats) == want, f"res {res}: {len(feats)} tokens, expected {want}"
        emb, pos = ev.merge(b, feats)
        assert len(pos) == want, f"res {res}: {len(pos)} slots survived, expected {want}"
        _, pred, pre_ms = ev.forward(emb)
        rec.append({"path": path.name, "wnid": wnid, "pred": pred,
                    "ok": int(LABELS[pred] == WNID2LABEL[wnid]),
                    "vision_ms": vis_ms, "prefill_ms": pre_ms})
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"    [{i+1}/{len(val)}] {el:.0f}s eta {el/(i+1)*(len(val)-i-1):.0f}s", flush=True)
    n = len(rec)
    return {"res": res, "tokens": want, "patches": (res // 16) ** 2, "n": n,
            "top1": sum(r["ok"] for r in rec) / n,
            "vision_ms": sum(r["vision_ms"] for r in rec) / n,
            "prefill_ms": sum(r["prefill_ms"] for r in rec) / n,
            "records": rec}


def main():
    print("=" * 78, flush=True)
    print(f"vlm_step020 resolution ladder  device={DEVICE}  depth={args.depth} STOCK  "
          f"res={args.res}  n_eval={args.n_eval}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
    proc = AutoProcessor.from_pretrained(args.model)
    ev = ResEval(model.eval().to(DEVICE), proc, DEVICE)
    ev.set_depth(args.depth)

    val = sample_images(DATA / args.split, args.n_eval, seed=args.seed)
    print(f"  {len(val)} eval images; {len(args.res)} cells", flush=True)

    cells = {}
    for res in args.res:
        print(f"  --- res {res} ({(res//16)**2} patches -> {(res//16)**2//16} tokens) ---", flush=True)
        cells[res] = cell(ev, res, val)

    base = cells[max(args.res)]
    print("\n  res  tokens   top1    d_pp   vision_ms  prefill_ms  total_ms  tower_x", flush=True)
    for res in args.res:
        c = cells[res]
        tot = c["vision_ms"] + c["prefill_ms"]
        c["delta_pp"] = (c["top1"] - base["top1"]) * 100
        c["total_ms"], c["tower_speedup"] = tot, base["vision_ms"] / c["vision_ms"]
        print(f"  {res:4d} {c['tokens']:6d}  {c['top1']:.4f} {c['delta_pp']:+6.2f}  "
              f"{c['vision_ms']:9.3f} {c['prefill_ms']:11.3f} {tot:9.3f} {c['tower_speedup']:8.2f}x",
              flush=True)

    if 512 in cells:
        d = abs(cells[512]["top1"] - ANCHOR) * 100
        print(f"\n  GATE validity  res512 d12 top1 {cells[512]['top1']:.4f} vs step006 {ANCHOR} "
              f"(d {cells[512]['top1']-ANCHOR:+.4f}) -> {'PASS' if d <= GATE_PP else 'FAIL'}", flush=True)
    print("  GATE token-count  asserted per image in every cell -> PASS (no assert fired)", flush=True)
    print("  T0 SCOUT: rejection filter. Every res with delta >= -0.5pp advances to T1.", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step020", "model": args.model, "device": str(DEVICE),
                               "depth": args.depth, "stock": True, "n_eval": len(val),
                               "seed": args.seed, "split": args.split, "anchor": ANCHOR,
                               "cells": cells}, indent=2))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
