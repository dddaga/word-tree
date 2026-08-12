"""vlm_step010: quantize the two lookup tables -- the largest untouched block in the model.

`embed_tokens` and `lm_head` are 49280x576 each: 28,385,280 params apiece, 11.07% of the model each,
**22.14% together** -- a bigger block than the 16.58% depth truncation deletes. They are lookups and
a readout, not transforms, so part 3 SS11.7 named them the one param-side win the profile endorses.
Untouched by everything vlm_step004-009 measured, and the stubborn goal is MEMORY, so this is the
highest-value remaining item on the trajectory.

Weight-only post-training quantization, symmetric, no calibration set and no retraining: the weights
are quantized and dequantized back to fp32 for the forward. Accuracy is therefore measured EXACTLY
as an int kernel would produce it; the byte saving is analytic and exact because the storage format
is fully determined (N bits per weight + one fp32 scale per row, or one per tensor).

Design: the full 2x4 grid, depth {12 teacher, 6 distilled} x quant {fp32, int8_row, int4_row,
int8_tensor}, every cell on the SAME 3859 images so all pairings hold. `fp32` is the untouched model,
so `d12_fp32` must reproduce step006's 0.711 and `d6_fp32` its 0.713 -- the run's validity check.

`int8_tensor` is a deliberate one-variable control against `int8_row`: identical bit-width, only the
scale granularity differs. It answers whether per-row scales are load-bearing or whether a single
tensor scale (simpler kernel, one less lookup) suffices.

PRE-REGISTERED DECISION RULE (written before the run):
  * A quant setting SHIPS at a depth iff its paired boot95 LOWER bound vs the same-depth fp32 cell
    is >= -2.0pp. KILLED iff the UPPER bound < -2.0pp. INCONCLUSIVE between, reported as such.
  * +-2.0pp is inherited from vlm_step006, where a 5.3pp headline swing turned out to be eval noise.
  * Compounding (the project's Compounding Rule, same statistic as vlm_step009): the double
    difference I(q) = [acc(d6,q) - acc(d6,fp32)] - [acc(d12,q) - acc(d12,fp32)] asks whether
    quantization costs the TRUNCATED tower more than the full one. ORTHOGONAL iff I(q)'s 95% CI is
    contained in +-2.0pp; SUB-ADDITIVE iff upper < -2.0pp; SUPER-ADDITIVE iff lower > +2.0pp.
    Depth acts on the vision tower and quantization on the text tables -- different modules, but they
    meet at the logits, so orthogonality is a HYPOTHESIS this measures rather than assumes.
  * Read `agree` alongside top-1 in every cell. vlm_step009 CONFIRMED agree(d6,d12) = 0.499 at
    matched top-1: these towers are equally accurate and NOT functionally equivalent.
  * Bytes are reported as actual bytes, never as a percentage alone (meditation-005: measure the
    GOAL metric, not a proxy).

Output: results/frontier/vlm_step010_quant_tables_{TAG}__{SLOT}.json  (includes per-image records)
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
from scripts.frontier.vlm_quant import (apply_arm, boot_dd, call_ortho, call_ship, hits,
                                        table_bytes)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=6, help="the distilled arm; 12 is always the ref")
parser.add_argument("--ckpt", default="results/frontier/vlm_step004_tower_distill_d6_25ep9k__mini_mps.pt")
parser.add_argument("--arms", nargs="+", default=["fp32", "int8_row", "int4_row", "int8_tensor"])
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
REF = "fp32"  # the unquantized reference every arm is measured against
TAG = f"d{args.depth}_{'-'.join(args.arms)}"
OUT = ROOT / "results" / "frontier" / f"vlm_step010_quant_tables_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"


def cell(depth, arm):
    return f"d{depth}_{arm}"


def evaluate(ev, tower, pristine, arm, val, act_mod=None):
    """One pass over the images at a fixed (depth, arm). Quantization is applied ONCE per arm, not
    per image -- writing 56.8M weights per image would dominate the run. Image order is fixed by
    sample_images, so index i is the same image in every pass and the pairing survives."""
    apply_arm(pristine, arm, act_mod)
    ev.set_layers(tower)
    preds, vis, pre, lps, t0 = [], 0.0, 0.0, [], time.perf_counter()
    for i, (path, _) in enumerate(val):
        lg, pred, vm, pm = ev.run(Image.open(path).convert("RGB"))
        preds.append(pred); vis += vm; pre += pm
        lps.append(F.log_softmax(lg, -1).cpu())
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"    [{i+1}/{len(val)}] {el:.0f}s eta {el/(i+1)*(len(val)-i-1):.0f}s", flush=True)
    n = len(val)
    return preds, lps, vis / n, pre / n


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"{'='*78}\nvlm_step010 table quant  device={DEVICE}  depth={args.depth}  arms={args.arms}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    m.requires_grad_(False)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    ck = Path(args.ckpt)
    if not ck.is_absolute(): ck = ROOT / ck
    towers = [(12, ev.full), (args.depth, list(load_student(ev, ck, args.depth, DEVICE)))]
    tabs = [ev.m.get_input_embeddings(), ev.m.lm_head]
    pristine = [(t, t.weight.data.clone()) for t in tabs]
    rows, cols = pristine[0][1].shape
    total = sum(p.numel() for p in m.parameters())
    print(f"  tables {rows}x{cols} x{len(tabs)} = {2*rows*cols:,} params "
          f"({200*rows*cols/total:.2f}% of {total:,});  distilled tower <- {ck}")

    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    print(f"  {len(val)} eval images; {len(towers) * len(args.arms)} cells", flush=True)
    out, ref_lp = {}, None
    for d, tower in towers:
        for arm in args.arms:
            print(f"  --- {cell(d, arm)} ---", flush=True)
            out[cell(d, arm)] = evaluate(ev, tower, pristine, arm, val, ev.m.lm_head)
            if ref_lp is None: ref_lp = out[cell(d, arm)][1]
    apply_arm(pristine, REF)  # leave the model as we found it (also drops any act hook)

    n = len(val)
    gold = [LABELS.index(WNID2LABEL[w]) for _, w in val]
    recs = [{"gold": g, "path": p.name, **{c: out[c][0][i] for c in out}}
            for i, ((p, _), g) in enumerate(zip(val, gold))]
    base = total * 4 - 2 * table_bytes(rows, cols, REF)  # everything outside the two tables
    res = {"step": "vlm_step010", "device": str(DEVICE), "depth": args.depth, "ckpt": str(ck),
           "arms": args.arms, "n_eval": n, "seed": args.seed, "cells": {}, "ship": {},
           "interactions": {}, "records": recs}
    print(f"\n  {'cell':<16} {'top1':>6} {'agree':>6} {'KL':>7} {'vis_ms':>8} {'pre_ms':>8} "
          f"{'tables_MB':>10} {'model_MB':>9}")
    for d, _ in towers:
        for arm in args.arms:
            preds, lps, v, p_ = out[cell(d, arm)]
            kl = sum(float(F.kl_div(lp, r, log_target=True, reduction="sum"))
                     for lp, r in zip(lps, ref_lp)) / n
            tb = 2 * table_bytes(rows, cols, arm)
            row = {"depth": d, "arm": arm, "top1": sum(int(a == b) for a, b in zip(preds, gold)) / n,
                   "agree_with_ref": sum(int(a == b) for a, b in
                                         zip(preds, out[cell(12, REF)][0])) / n,
                   "kl_vs_ref": kl, "vision_ms": round(v, 2), "prefill_ms": round(p_, 2),
                   "tables_bytes": tb, "model_bytes": base + tb}
            res["cells"][cell(d, arm)] = row
            print(f"  {cell(d,arm):<16} {row['top1']:>6.3f} {row['agree_with_ref']:>6.3f} {kl:>7.3f} "
                  f"{v:>8.1f} {p_:>8.1f} {tb/1e6:>10.1f} {(base+tb)/1e6:>9.1f}")

    print(f"\n  paired vs same-depth {REF}")
    for d, _ in towers:
        for arm in (a for a in args.arms if a != REF):
            s, t = hits(recs, cell(d, arm)), hits(recs, cell(d, REF))
            b = sum(1 for i in range(n) if s[i] and not t[i])
            c = sum(1 for i in range(n) if t[i] and not s[i])
            lo, hi = boot_ci(s, t, args.boot)
            v = call_ship(lo, hi)
            res["ship"][cell(d, arm)] = {"delta": (sum(s) - sum(t)) / n, "boot95_lo": lo,
                                         "boot95_hi": hi, "b": b, "c": c,
                                         "p_two_sided": mcnemar_exact(b, c), "verdict": v}
            print(f"  {cell(d,arm):<16} d {(sum(s)-sum(t))/n:+.4f}  boot95 [{lo:+.4f}, {hi:+.4f}]  "
                  f"b={b} c={c} p={mcnemar_exact(b,c):.5f}  -> {v}")

    print(f"\n  double difference I(arm) = [d{args.depth} quant loss] - [d12 quant loss]")
    for arm in (a for a in args.arms if a != REF):
        pt, lo, hi = boot_dd(recs, [cell(args.depth, arm), cell(args.depth, REF),
                                    cell(12, arm), cell(12, REF)], args.boot)
        v = call_ortho(lo, hi, lever="quantization")
        res["interactions"][arm] = {"I": pt, "boot95_lo": lo, "boot95_hi": hi, "verdict": v}
        print(f"  {arm:<12} I {pt:+.4f}  boot95 [{lo:+.4f}, {hi:+.4f}]  -> {v}")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
