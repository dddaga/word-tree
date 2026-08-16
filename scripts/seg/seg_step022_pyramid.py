"""seg_step022/023: what is the M_FPM pyramid actually worth, and is a TRUE global vector missing?

WHY NOW. seg_step014 is a CONFIRMED NULL: a better teacher buys the student nothing, i.e. the hint
saturates at the STUDENT'S CAPACITY. Init (+4.26pp, step003), logit KD (+1.51pp, step009) and the
feature hint (+3.67pp, step010/011) are all banked and all flat in their own weight. So for the first
time in this line the binding constraint is the student's architecture, not its supervision — which
makes reallocating capacity inside the fixed 1 GMAC drone ceiling the right lever, and makes the
pyramid the first place to look, since it is the only part of the net that is a design choice rather
than a slice of VGG.

TWO QUESTIONS, ONE SCRIPT (they share the teacher, the cached targets and the loss; splitting them
would cost a second teacher load per cell and let the two answers drift apart on data):

  seg_step022 — BRANCH ABLATION. Drop pool/d1/d4/d8/d16 one at a time. At a 16x16 grid a rate-16
    3x3 kernel already spans the whole map, so d16 is a candidate for being pure cost. Every dropped
    branch frees ~20% of the head's input width and its share of the pyramid's 1.32M params, which is
    capacity that seg_step024 can spend on depth or resolution instead. Cost only falls in every
    ablation cell, so GMAC is held as the budget and the cheaper cells are strictly free if neutral.

  seg_step023 — TRUE GLOBAL CONTEXT. M_FPM's `pool` branch is a 1x1 conv on a 3x3 max-pool: local,
    despite the name. There is NO image-level vector anywhere in this net. gap=cat adds one by global
    average pooling and concatenating. gap=se adds the SAME vector as a multiplicative squeeze-excite
    gate. The multiplicative arm is a PRE-REGISTERED CONTROL, not a hopeful second bet: the gate-death
    theorem (steps 873-916) and GLAM's T0 (mul hurts, add neutral) both predict cat >= se. If se wins
    here, both are falsified on a dense spatial task and THAT is the result worth writing up.

HELD FIXED so the pyramid is the only variable: E7 student (0.8757 GMAC / 851,816 params), E2 teacher,
arm H1 (logit KD alpha=2.0 + feature hint beta=1.0), 128px in / 16x16 out, 60 epochs. The 16x16 grid
in particular is NOT moved here — an 8x8 grid changes what "global" means and would confound d16.

PRE-REGISTERED BAR: +1.2pp on mean best_mAP over the `full` cell to call a change a win, ~2 sigma of
the hint arm's own +-0.56pp seed spread (seg_step011). A cell within +-1.2pp at LOWER cost is a WIN on
the Pareto and is reported as such; accuracy alone never decides here. Cells are paired seed-by-seed.

Output: results/seg/seg_step022_{TAG}__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.seg.seg_common import BRANCHES
from scripts.seg.seg_encoders import SegNet
from scripts.seg.seg_eval import cache_targets, count_macs, evaluate
from scripts.seg.seg_train import fit, load_teacher

# name -> (branches, gap). `full` is the incumbent and every delta is measured against it.
CELLS = {"full": (BRANCHES, "none"),
         **{f"no_{b}": (tuple(x for x in BRANCHES if x != b), "none") for b in BRANCHES},
         "gap_cat": (BRANCHES, "cat"),
         "gap_se": (BRANCHES, "se"),
         # t1: the two Pareto wins from the t0 pass taken together. no_pool (+0.22pp, -8,896p) and
         # no_d1 (-0.41pp, -111,296p) both sit on the pyramid's own signal path, so the Compounding
         # Rule forbids assuming the deltas add; this is the paired cell that measures it.
         "lean": (("d4", "d8", "d16"), "none")}
BAR_PP = 1.2

parser = argparse.ArgumentParser()
parser.add_argument("--cells", nargs="+", default=list(CELLS),
                    help=f"subset of {list(CELLS)}; 'full' is always run first as the paired control")
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
parser.add_argument("--teacher_arm", default="E2")
parser.add_argument("--student_arm", default="E7")
parser.add_argument("--device", default="auto")
parser.add_argument("--teacher_device", default="cpu")   # fasterrcnn HANGS on MPS
parser.add_argument("--arm", default="H1", help="H0 logit KD only / H1 + feature hint (default)")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--teacher_epochs", type=int, default=60)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--temp", type=float, default=2.0)
parser.add_argument("--res", type=int, default=128)
parser.add_argument("--grid", type=int, default=16)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--c_mid", type=int, default=64)
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_val", type=int, default=500)
parser.add_argument("--decor_lambda", type=float, default=0.05)
parser.add_argument("--teacher_seed", type=int, default=42)
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="t0")
args = parser.parse_args()
args.shuffle_hint = False       # fit() reads it; the step011 control is not part of this question
args.save_ckpt = False          # no cell here is a candidate teacher

if args.smoke_test:
    args.seeds, args.epochs, args.n_train, args.n_val = [42], 1, 40, 20

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "seg" / f"seg_step022_{args.tag}__{SLOT}.json"


def build(cell, seed):
    branches, gap = CELLS[cell]
    torch.manual_seed(seed)
    return SegNet(args.student_arm, k=args.k, c_mid=args.c_mid,
                  branches=branches, gap=gap).to(DEVICE)


def smoke():
    ok = True
    for cell in args.cells:
        s = build(cell, 42)
        x = torch.zeros(2, 3, args.res, args.res, device=DEVICE)
        with torch.no_grad():
            y = s(x)
        n_p = sum(p.numel() for p in s.parameters() if p.requires_grad)
        good = y.shape[2] == args.grid and s.head.in_channels == s.mfpm.out_ch
        ok &= good
        print(f"  {cell:9s} out={tuple(y.shape)} mfpm_out={s.mfpm.out_ch:4d} "
              f"params={n_p:,} {count_macs(s, args.res)/1e9:.4f}G {'OK' if good else 'FAIL'}")
    return ok


def main():
    if args.smoke_test and not smoke():
        sys.exit(1)
    print(f"\n{'='*70}\nseg_step022 pyramid  cells={args.cells}  seeds={args.seeds}\n"
          f"  {args.teacher_arm} -> {args.student_arm}  arm={args.arm} "
          f"alpha={args.alpha} beta={args.beta}  device={DEVICE}\n{'='*70}", flush=True)
    tx, ty = cache_targets("train", args.n_train, args.res, args.grid, args.teacher_device)
    vx, vy = cache_targets("val", args.n_val, args.res, args.grid, args.teacher_device)
    teacher = load_teacher(args, DEVICE, SLOT)

    order = (["full"] if "full" in args.cells else []) + [c for c in args.cells if c != "full"]
    cells, t_all = {}, time.time()
    for cell in order:
        runs = []
        for seed in args.seeds:
            student = build(cell, seed)
            n_p = sum(p.numel() for p in student.parameters() if p.requires_grad)
            macs = count_macs(student, args.res)
            t0 = time.time()
            best, best_ep, _ = fit(student, teacher, tx, ty, vx, vy, args, DEVICE, seed)
            final = evaluate(student, vx, vy, DEVICE)
            print(f"  {cell:9s} seed{seed}  best_mAP={best:.4f} @ep{best_ep}  "
                  f"{n_p:,}p {macs/1e9:.4f}G  [{time.time()-t0:.0f}s]", flush=True)
            runs.append({"seed": seed, "best_mAP": best, "best_ep": best_ep, "n_params": n_p,
                         "macs": macs, "final": final})
        m = [r["best_mAP"] for r in runs]
        cells[cell] = {"branches": list(CELLS[cell][0]), "gap": CELLS[cell][1], "runs": runs,
                       "mean": statistics.mean(m),
                       "std": statistics.stdev(m) if len(m) > 1 else 0.0,
                       "n_params": runs[0]["n_params"], "gmacs": round(runs[0]["macs"] / 1e9, 4)}
        print(f"  -> {cell:9s} mean={cells[cell]['mean']:.4f} +-{cells[cell]['std']:.4f}", flush=True)

    base = cells.get("full")
    print(f"\n{'cell':10s} {'mean':>8s} {'+-':>7s} {'d_pp':>7s} {'params':>10s} {'GMAC':>7s}  verdict")
    for cell, c in cells.items():
        d = (c["mean"] - base["mean"]) * 100 if base else float("nan")
        cheaper = base is not None and c["n_params"] < base["n_params"]
        v = ("BASE" if cell == "full" else
             "WIN" if d >= BAR_PP else
             "WIN(pareto)" if cheaper and d > -BAR_PP else
             "NEUTRAL" if abs(d) < BAR_PP else "LOSS")
        c["delta_pp"], c["verdict"] = d, v
        print(f"{cell:10s} {c['mean']:8.4f} {c['std']:7.4f} {d:+7.2f} {c['n_params']:10,} "
              f"{c['gmacs']:7.4f}  {v}")

    if args.smoke_test:      # a 1-epoch shakedown must never overwrite a real result file
        print("\n(smoke test — results not written)")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "seg_step022", "bar_pp": BAR_PP, "arm": args.arm,
                               "alpha": args.alpha, "beta": args.beta, "epochs": args.epochs,
                               "seeds": args.seeds, "teacher_arm": args.teacher_arm,
                               "student_arm": args.student_arm, "grid": args.grid,
                               "res": args.res, "device": str(DEVICE),
                               "elapsed_s": round(time.time() - t_all, 1),
                               "cells": cells}, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
