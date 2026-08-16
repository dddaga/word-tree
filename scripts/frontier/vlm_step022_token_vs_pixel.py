"""vlm_step022: is the wall the TOKEN BUDGET or the PIXEL GRID? A training-free ceiling test.

WHY. step021 distilled a d12 tower to run at 256 px (16 tokens) and got PARTIAL: naive 0.1600 ->
distilled 0.2980 (+13.80pp), well short of the teacher's 0.7040. But the student was NOT failing to
fit its target -- rel_mse fell to 0.0545 and cosine reached 0.9738, a tighter fit than any depth
distillation in this line ever achieved. A student that matches its target that well and still
scores 0.298 is telling us the TARGET is the problem, not the optimisation.

And the target is confounded by construction: step021's teacher signal is the 512-px teacher's 64
tokens AVERAGE-POOLED to 16. So a 16-token student was asked to reproduce a BLURRED teacher. Two
different things could cap it and step021 cannot separate them:
  (H1) PIXEL GRID   -- 256 px genuinely destroys information the task needs.
  (H2) TOKEN BUDGET -- 16 decoder tokens cannot carry the answer no matter how good the pixels are.

THE TEST, AND WHY IT NEEDS NO TRAINING. Run the tower at FULL 512 px (so the pixels are perfect, by
construction), then average-pool its 64 output tokens down to k tokens and feed those to the frozen
decoder. That is an ORACLE 16-token model: no student can beat it, because it is handed the real
teacher's features and only then has its token count cut. So:
  oracle@16 ~ 0.30  -> the 16-token budget is the wall. Pixels are exonerated; step021's student is
                      already near its ceiling and more epochs will not save it. Resolution is then
                      the WRONG knob and the next lever is token count itself (or a smarter pool).
  oracle@16 ~ 0.70  -> tokens are fine and the 256-px grid is the wall (H1). step021's student then
                      has real headroom and the still-falling loss curve says buy more epochs.
Anything between reads as a mix, reported as such rather than forced into one bucket.

This costs one eval pass per k -- no training, no checkpoints, minutes not hours. It should have
been run BEFORE step021, and that is the lesson: when a distillation underperforms, first bound
what its target could possibly deliver.

THE POOLING LADDER. k in {64, 36, 16, 9, 4} mirrors step020's resolution ladder exactly, so the two
curves are directly comparable at matched token counts. Any gap between oracle@k and the step020
res-cell with the same k is the pure cost of the pixel grid, with token count held fixed -- which
is the decomposition step020 could not do on its own.

PRE-REGISTERED GATES:
  (1) VALIDITY. k=64 is a no-op pool (identity), so that cell MUST reproduce the d12@512 anchor
      within +-2.0pp of 0.711. If it does not, the pooling path is broken and nothing here is
      quotable.
  (2) MONOTONICITY IS NOT ASSUMED. Reported, not required -- a non-monotone curve is a finding
      about the decoder's token handling, not a bug to explain away.
INTERPRETATION LIMIT: average pooling is ONE way to spend a smaller token budget, and a bad result
bounds average-pooled tokens, not all possible 16-token encodings. Stated here so the negative is
not over-read into "16 tokens are impossible".

Output: results/frontier/vlm_step022_token_vs_pixel_{TAG}__{SLOT}.json
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

from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, sample_images
from scripts.frontier.vlm_res import ResEval

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=12, help="STOCK; token count is the only variable")
parser.add_argument("--tokens", type=int, nargs="+", default=[64, 36, 16, 9, 4])
parser.add_argument("--n_eval", type=int, default=500)
parser.add_argument("--seed", type=int, default=42, help="42 reproduces step020/021's eval set")
parser.add_argument("--split", default="val")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.tokens = 20, [64, 16]

SLOT = os.environ.get("SGN_SLOT", "local")
TAG = "k" + "-".join(str(k) for k in args.tokens) + f"_n{args.n_eval}"
OUT = ROOT / "results" / "frontier" / f"vlm_step022_token_vs_pixel_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
ANCHOR, GATE_PP = 0.711, 2.0

# step020's ladder at matched token counts, for the pixel-vs-token decomposition.
STEP020 = {64: 0.7040, 36: 0.4560, 16: 0.1600, 9: 0.0940, 4: 0.1060}


def pick_device(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device(args.device)


def pool_tokens(feats, k):
    """(T, H) tower output on a square grid -> (k, H). Identity when k == T."""
    t, h = feats.shape
    if k == t: return feats
    g, s = int(round(t ** 0.5)), int(round(k ** 0.5))
    assert g * g == t and s * s == k, f"{t} -> {k} not a square-to-square pool"
    x = feats.view(1, g, g, h).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(x, s).permute(0, 2, 3, 1).reshape(k, h)


def cell(ev, k, val):
    """One token budget, at FULL 512-px pixels. Pixels perfect by construction."""
    ev.res = 512
    ok = vis = pre = 0.0
    t0 = time.perf_counter()
    for i, (path, wnid) in enumerate(val):
        b = ev.batch(Image.open(path).convert("RGB"))
        feats, v = ev.encode(b)
        assert len(feats) == 64, f"expected 64 teacher tokens, got {len(feats)}"
        emb, pos = ev.merge(b, pool_tokens(feats, k))
        assert len(pos) == k, f"k={k}: {len(pos)} slots survived, expected {k}"
        _, pred, p = ev.forward(emb)
        ok += LABELS[pred] == WNID2LABEL[wnid]; vis += v; pre += p
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"    [{i+1}/{len(val)}] {el:.0f}s eta {el/(i+1)*(len(val)-i-1):.0f}s", flush=True)
    n = len(val)
    return {"tokens": k, "n": n, "top1": ok / n,
            "vision_ms": vis / n, "prefill_ms": pre / n}


def main():
    print("=" * 78, flush=True)
    print(f"vlm_step022 token-vs-pixel  device={DEVICE}  depth={args.depth} STOCK  "
          f"tokens={args.tokens}  n_eval={args.n_eval}  pixels=512 FULL", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
    ev = ResEval(model.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    ev.set_depth(args.depth)

    val = sample_images(DATA / args.split, args.n_eval, seed=args.seed)
    print(f"  {len(val)} eval images; {len(args.tokens)} cells", flush=True)

    cells = {}
    for k in args.tokens:
        print(f"  --- k={k} tokens (512 px, pooled) ---", flush=True)
        cells[k] = cell(ev, k, val)

    print("\n  k   oracle  step020@k   pixel_cost_pp   (oracle - res-cell at same tokens)",
          flush=True)
    for k in args.tokens:
        c = cells[k]
        ref = STEP020.get(k)
        if ref is not None:
            c["step020_top1"], c["pixel_cost_pp"] = ref, (c["top1"] - ref) * 100
            print(f"  {k:3d}  {c['top1']:.4f}   {ref:.4f}      {c['pixel_cost_pp']:+7.2f}",
                  flush=True)
        else:
            print(f"  {k:3d}  {c['top1']:.4f}      --            --", flush=True)

    if 64 in cells:
        d = abs(cells[64]["top1"] - ANCHOR) * 100
        print(f"\n  GATE validity  k=64 (identity pool) top1 {cells[64]['top1']:.4f} vs {ANCHOR} "
              f"-> {'PASS' if d <= GATE_PP else 'FAIL'}", flush=True)
    if 16 in cells:
        o = cells[16]["top1"]
        call = ("TOKEN BUDGET is the wall (step021's student is near its ceiling)" if o < 0.40 else
                "PIXEL GRID is the wall (step021's student has headroom -- buy more epochs)"
                if o > 0.60 else "MIXED -- both contribute; report both, force neither")
        print(f"  CALL  oracle@16 = {o:.4f} vs step021 distilled 0.2980 -> {call}", flush=True)
    print("  LIMIT: bounds AVERAGE-POOLED token budgets, not all possible k-token encodings.",
          flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step022", "model": args.model, "device": str(DEVICE),
                               "depth": args.depth, "pixels": 512, "n_eval": len(val),
                               "seed": args.seed, "split": args.split, "anchor": ANCHOR,
                               "step021_distilled_r256": 0.2980, "cells": cells}, indent=2))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
