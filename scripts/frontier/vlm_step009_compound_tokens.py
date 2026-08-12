"""vlm_step009: do depth truncation and token pruning COMPOUND, or do they cancel?

Two levers are CONFIRMED separately. Depth: the d6 distilled tower sits at teacher parity for 16.58%
of the model deleted and 1.60x faster vision (vlm_step004+006, n=3859). Tokens: top-k by
post-connector L2 norm beats an even-stride control at identical budget by +18pp (vlm_step002). The
drone needs BOTH. The project's Compounding Rule says mechanisms on the same signal path tend to
cancel, and these two are adjacent: depth changes what the tower emits, top-k selects among what it
emitted. Plausibly orthogonal -- but that is a HYPOTHESIS until the isolation ablation runs, and
this is the ablation.

Eval-only. Both checkpoints exist, nothing is trained, so the whole cost is forward passes.

Design: the full 2x4 grid, depth {12 teacher, 6 distilled} x k {64, 32, 16, 8}, on ONE eval set with
per-image records so every cell is paired with every other. k=64 is no pruning (prune() is a no-op
at k >= len(feats)), so the top row reproduces step006's two numbers and is the run's validity
check -- if teacher_d12_k64 does not land near 0.711, nothing else is readable.

The quantity of interest is NOT any single cell. It is the DOUBLE difference

    I(k) = [acc(d6,k) - acc(d6,64)] - [acc(d12,k) - acc(d12,64)]

i.e. does pruning cost the truncated tower MORE than it costs the full one. Bootstrapped by
resampling images once per replicate and recomputing all four terms on the shared index, so the
pairing across all four cells survives.

PRE-REGISTERED DECISION RULE (written before the run):
  * ORTHOGONAL at budget k iff I(k)'s 95% CI is contained in +-2.0pp. The levers compound; the drone
    stacks them and the expected loss is the sum of the two solo losses.
  * SUB-ADDITIVE iff the CI's UPPER bound < -2.0pp. They cancel: the tower must keep depth to afford
    pruning, and the stacked config must be re-tuned, not assumed.
  * SUPER-ADDITIVE iff the LOWER bound > +2.0pp. Distillation made the tower MORE prunable -- report
    it, do not build on it without a replication.
  * Anything else is INCONCLUSIVE at this n and is reported as such.
  * +-2.0pp is inherited from step006, where a 5.3pp headline swing turned out to be eval noise.
  * Read `agree` alongside top-1 in every cell: on a closed 10-way readout top-1 can be right for the
    wrong reason, and step004's agree 0.420 vs top-1 0.560 is the standing proof of that.

Output: results/frontier/vlm_step009_compound_tokens_{TAG}__{SLOT}.json  (includes per-image records)
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))
if (_LIBS := ROOT / "vlm_libs").is_dir(): sys.path.insert(0, str(_LIBS))

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.frontier.vlm_distill import load_student
from scripts.frontier.vlm_eval import (LABELS, WNID2LABEL, VLMEval, boot_ci, mcnemar_exact,
                                       sample_images)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=6, help="the distilled arm; 12 is always the ref")
parser.add_argument("--ckpt", default="results/frontier/vlm_step004_tower_distill_d6_25ep9k__mini_mps.pt")
parser.add_argument("--budgets", type=int, nargs="+", default=[64, 32, 16, 8],
                    help="image tokens kept; 64 == no pruning == the step006 reproduction")
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
FULL = max(args.budgets)  # the unpruned reference budget both depths are measured against
TAG = f"d{args.depth}_k{'-'.join(map(str, args.budgets))}"
OUT = ROOT / "results" / "frontier" / f"vlm_step009_compound_tokens_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"


def cell(depth, k):
    return f"d{depth}_k{k}"


def evaluate(ev, towers, val):
    """One pass over the images; every (depth, k) cell measured on the SAME image, so all pairings
    hold. The reference cell (teacher, unpruned) is evaluated first and is the same/agree/KL ref."""
    cells = [(d, k) for d, _ in towers for k in args.budgets]
    acc = {cell(d, k): dict.fromkeys(("correct", "same", "agree", "kl", "vis", "pre"), 0.0)
           for d, k in cells}
    recs, t0 = [], time.perf_counter()
    for i, (path, wnid) in enumerate(val):
        img = Image.open(path).convert("RGB")
        gold = LABELS.index(WNID2LABEL[wnid])
        rec, ref = {"wnid": wnid, "gold": gold, "path": path.name}, None
        for d, mods in towers:
            ev.set_layers(mods)
            for k in args.budgets:
                lg, pred, vm, pm = ev.run(img, k=k)
                lp = F.log_softmax(lg, -1)
                if ref is None: ref = (pred, lp, lg.argmax().item())
                a = acc[cell(d, k)]
                a["correct"] += pred == gold; a["same"] += pred == ref[0]
                a["agree"] += lg.argmax().item() == ref[2]
                a["kl"] += float(F.kl_div(lp, ref[1], log_target=True, reduction="sum"))
                a["vis"] += vm; a["pre"] += pm
                rec[cell(d, k)] = pred
        recs.append(rec)
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  eval [{i+1}/{len(val)}] {el:.0f}s eta {el/(i+1)*(len(val)-i-1):.0f}s",
                  flush=True)
    return acc, recs


def hits(recs, name):
    return [int(r[name] == r["gold"]) for r in recs]


def boot_dd(recs, k, depth, n_boot, seed=0):
    """Percentile bootstrap on the double difference I(k). One shared resample index per replicate
    feeds all four cells, so the four-way pairing is preserved -- the whole point of measuring every
    cell on the same image. Returns (point, lo, hi)."""
    cols = [hits(recs, cell(depth, k)), hits(recs, cell(depth, FULL)),
            hits(recs, cell(12, k)), hits(recs, cell(12, FULL))]
    n, rng, out = len(recs), random.Random(seed), []
    point = (sum(cols[0]) - sum(cols[1]) - sum(cols[2]) + sum(cols[3])) / n
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [sum(c[i] for i in idx) for c in cols]
        out.append((s[0] - s[1] - s[2] + s[3]) / n)
    out.sort()
    return point, out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


def verdict(lo, hi, tol=0.02):
    """The docstring's pre-registered rule, applied to one budget's CI."""
    if lo >= -tol and hi <= tol: return "ORTHOGONAL -- levers compound"
    if hi < -tol: return "SUB-ADDITIVE -- levers cancel, stacked config must be re-tuned"
    if lo > tol: return "SUPER-ADDITIVE -- distillation raised prunability; replicate before use"
    return "INCONCLUSIVE at this n"


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"{'='*78}\nvlm_step009 depth x tokens  device={DEVICE}  depth={args.depth}  "
          f"budgets={args.budgets}  n_eval={args.n_eval}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    m.requires_grad_(False)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    ck = Path(args.ckpt)
    if not ck.is_absolute(): ck = ROOT / ck
    towers = [(12, ev.full), (args.depth, list(load_student(ev, ck, args.depth, DEVICE)))]
    print(f"  distilled tower <- {ck}")

    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    print(f"  {len(val)} eval images; {len(towers) * len(args.budgets)} cells", flush=True)
    acc, recs = evaluate(ev, towers, val)

    n = len(val)
    res = {"step": "vlm_step009", "device": str(DEVICE), "depth": args.depth, "ckpt": str(ck),
           "budgets": args.budgets, "n_eval": n, "seed": args.seed, "cells": {},
           "interactions": {}, "records": recs}
    print(f"\n  {'cell':<10} {'top1':>6} {'same':>6} {'agree':>6} {'KL':>7} {'vis_ms':>8} "
          f"{'pre_ms':>8} {'total':>8}")
    for d, _ in towers:
        for k in args.budgets:
            a = acc[cell(d, k)]; v, p_ = a["vis"] / n, a["pre"] / n
            row = {"depth": d, "tokens": k, "top1": a["correct"] / n,
                   "same_class_as_ref": a["same"] / n, "agree_with_ref": a["agree"] / n,
                   "kl_vs_ref": a["kl"] / n, "vision_ms": round(v, 2), "prefill_ms": round(p_, 2),
                   "total_ms": round(v + p_, 2)}
            res["cells"][cell(d, k)] = row
            print(f"  {cell(d,k):<10} {row['top1']:>6.3f} {row['same_class_as_ref']:>6.3f} "
                  f"{row['agree_with_ref']:>6.3f} {row['kl_vs_ref']:>7.3f} {v:>8.1f} {p_:>8.1f} "
                  f"{v+p_:>8.1f}")

    print(f"\n  double difference I(k) = [d{args.depth} loss from pruning] - [d12 loss from pruning]")
    for k in sorted(b for b in args.budgets if b != FULL):
        pt, lo, hi = boot_dd(recs, k, args.depth, args.boot)
        s, t = hits(recs, cell(args.depth, k)), hits(recs, cell(12, k))
        b = sum(1 for i in range(n) if s[i] and not t[i])
        c = sum(1 for i in range(n) if t[i] and not s[i])
        call = verdict(lo, hi)
        res["interactions"][str(k)] = {"I": pt, "boot95_lo": lo, "boot95_hi": hi, "verdict": call,
                                       "b": b, "c": c, "p_two_sided": mcnemar_exact(b, c),
                                       "vs_teacher_same_k": boot_ci(s, t, args.boot)}
        print(f"  k={k:<4} I {pt:+.4f}  boot95 [{lo:+.4f}, {hi:+.4f}]  "
              f"(d{args.depth} vs d12 at k: b={b} c={c} p={mcnemar_exact(b,c):.5f})  -> {call}")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
