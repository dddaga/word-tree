"""seg_step024: at a FIXED MAC budget, which encoder block family buys the most accuracy?

WHY NOW. seg_step022 closed the head-side question — `lean` (M_FPM minus pool and d1) is the default
student at 731,624p / 0.8450 GMAC, the two drops are additive, and nothing on the head cleared the
+1.2pp bar. Supervision is saturated too (seg_step014 NULL: a better teacher buys nothing). What has
never been varied in this line is the ENCODER BLOCK: every arm E0-E12 is a VGG-shaped stack of dense
3x3 convs, an inherited constraint rather than a measured choice.

THE DESIGN IS DICTATED BY THE INIT CONFOUND. seg_step003 CONFIRMED the sliced ImageNet init is worth
+4.26pp — larger than any architecture lever measured here. No family in this step can inherit that
slice (there is no VGG counterpart to a 7x7 depthwise kernel), so reading `cnx` against `lean` would
report family + init as one number, dominated by init. Every cell is therefore SCRATCH and the paired
base is `e8` (the lean architecture at random init), per seg_encoders.py's standing rule that scratch
arms are read against the scratch control, never a pretrained number. HONEST LIMIT: this answers
"best family at scratch"; a winner still needs an init found for it, which is seg_step025's problem.

ISO-GMAC, NOT ISO-WIDTH. Separable and inverted-residual blocks are several times cheaper per layer,
so comparing them at E8's channel counts would just measure a smaller net (that is E4, already a
LOSS). Each family's interior width is bisected to land within 1% of e8's GMAC, so every cell spends
the same compute and the question is purely how it is spent.

HELD FIXED: 3 stages, reps [2,2,3], stride 8, encoder output PINNED at 128 channels so the
seg_step010 feature hint (which reads t_enc[:, :C_student]) is identical in every cell. E2 teacher,
arm H1 (alpha=2.0 logit KD + beta=1.0 hint), 128px in / 16x16 out, 60 epochs, 5 seeds, `lean` pyramid.

CELLS. e8 = untouched E8 encoder, the paired base and the tie back to the existing line. vgg = the
same dense 3x3 shape in this file's harness, differing from e8 ONLY by a residual on the c->c
repeats, so vgg minus e8 isolates the skip connection. sep, ir, cnx are the three families.

PRE-REGISTERED BAR: +1.2pp on mean best_mAP over `e8`, same bar and rationale as seg_step022 (~2
sigma of the hint arm's seed spread). A cell inside +-1.2pp at lower cost is a WIN on the Pareto;
accuracy alone never decides. Cells are paired seed-by-seed.

Output: results/seg/seg_step024_{TAG}__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.seg.seg_blocks import build_block_stack
from scripts.seg.seg_encoders import SegNet
from scripts.seg.seg_eval import cache_targets, count_macs, evaluate
from scripts.seg.seg_train import fit, load_teacher

LEAN = ("d4", "d8", "d16")      # seg_step022 t1 default pyramid
CELLS = ["e8", "vgg", "sep", "ir", "cnx"]
BAR_PP = 1.2

parser = argparse.ArgumentParser()
parser.add_argument("--cells", nargs="+", default=CELLS,
                    help=f"subset of {CELLS}; 'e8' is always run first as the paired control")
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
parser.add_argument("--teacher_arm", default="E2")
parser.add_argument("--student_arm", default="E8", help="scratch control; the base cell's encoder")
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
OUT = ROOT / "results" / "seg" / f"seg_step024_{args.tag}__{SLOT}.json"

WIDTHS: dict[str, float] = {}   # family -> width solved against e8's GMAC


def build(cell, seed):
    """Every cell is the SAME SegNet with the SAME lean pyramid and head; only .enc is swapped.
    Swapping beats adding arms to seg_encoders.ARMS here: the encoder output is pinned at 128
    channels, so E8's M_FPM and head already fit and no existing module has to change."""
    torch.manual_seed(seed)
    net = SegNet(args.student_arm, k=args.k, c_mid=args.c_mid, branches=LEAN, gap="none")
    if cell != "e8":
        want = next(m.in_channels for m in net.mfpm.modules()
                    if isinstance(m, torch.nn.Conv2d))
        enc, c_out = build_block_stack(cell, WIDTHS.get(cell, 1.0))
        if c_out != want:
            raise ValueError(f"{cell}: encoder out {c_out} != pyramid input {want}; the hint "
                             f"contract requires the final stage stay pinned")
        net.enc = enc
    return net.to(DEVICE)


def solve_width(fam: str, target: float, lo: float = 0.1, hi: float = 12.0, tol: float = 0.01):
    """Bisect the interior-width multiplier onto `target` MACs. MACs are monotone in width (every
    term is non-decreasing), so this is exact rather than a heuristic."""
    for _ in range(40):
        mid = (lo + hi) / 2
        WIDTHS[fam] = mid
        m = count_macs(build(fam, 42), args.res)
        if abs(m - target) / target < tol:
            return mid, m
        lo, hi = (mid, hi) if m < target else (lo, mid)
    return WIDTHS[fam], m


def main():
    order = (["e8"] if "e8" in args.cells else []) + [c for c in args.cells if c != "e8"]
    print(f"\n{'='*70}\nseg_step024 blocks  cells={order}  seeds={args.seeds}\n"
          f"  {args.teacher_arm} -> {args.student_arm}(scratch) arm={args.arm} "
          f"alpha={args.alpha} beta={args.beta}  device={DEVICE}\n{'='*70}", flush=True)

    target = count_macs(build("e8", 42), args.res)
    print(f"  budget = e8 {target/1e9:.4f} GMAC; solving interior widths", flush=True)
    for fam in [c for c in order if c != "e8"]:
        w, m = solve_width(fam, target)
        print(f"    {fam:4s} width={w:.3f} -> {m/1e9:.4f}G ({100*(m-target)/target:+.1f}%)", flush=True)

    for cell in order:      # every cell must still land on the 16x16 grid the head expects
        with torch.no_grad():
            y = build(cell, 42)(torch.zeros(2, 3, args.res, args.res, device=DEVICE))
        if y.shape[2] != args.grid:
            sys.exit(f"{cell}: output grid {y.shape[2]} != {args.grid}")

    tx, ty = cache_targets("train", args.n_train, args.res, args.grid, args.teacher_device)
    vx, vy = cache_targets("val", args.n_val, args.res, args.grid, args.teacher_device)
    teacher = load_teacher(args, DEVICE, SLOT)

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
            print(f"  {cell:4s} seed{seed}  best_mAP={best:.4f} @ep{best_ep}  "
                  f"{n_p:,}p {macs/1e9:.4f}G  [{time.time()-t0:.0f}s]", flush=True)
            runs.append({"seed": seed, "best_mAP": best, "best_ep": best_ep, "n_params": n_p,
                         "macs": macs, "final": final})
        m = [r["best_mAP"] for r in runs]
        cells[cell] = {"width": round(WIDTHS.get(cell, 1.0), 4), "runs": runs,
                       "mean": statistics.mean(m),
                       "std": statistics.stdev(m) if len(m) > 1 else 0.0,
                       "n_params": runs[0]["n_params"], "gmacs": round(runs[0]["macs"] / 1e9, 4)}
        print(f"  -> {cell:4s} mean={cells[cell]['mean']:.4f} +-{cells[cell]['std']:.4f}", flush=True)

    base = cells.get("e8")
    print(f"\n{'cell':6s} {'width':>6s} {'mean':>8s} {'+-':>7s} {'d_pp':>7s} {'params':>10s} "
          f"{'GMAC':>7s}  verdict")
    for cell, c in cells.items():
        d = (c["mean"] - base["mean"]) * 100 if base else float("nan")
        cheaper = base is not None and c["n_params"] < base["n_params"]
        v = ("BASE" if cell == "e8" else
             "WIN" if d >= BAR_PP else
             "WIN(pareto)" if cheaper and d > -BAR_PP else
             "NEUTRAL" if abs(d) < BAR_PP else "LOSS")
        c["delta_pp"], c["verdict"] = d, v
        print(f"{cell:6s} {c['width']:6.3f} {c['mean']:8.4f} {c['std']:7.4f} {d:+7.2f} "
              f"{c['n_params']:10,} {c['gmacs']:7.4f}  {v}")

    if args.smoke_test:      # a 1-epoch shakedown must never overwrite a real result file
        print("\n(smoke test — results not written)")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "seg_step024", "bar_pp": BAR_PP, "arm": args.arm,
                               "alpha": args.alpha, "beta": args.beta, "epochs": args.epochs,
                               "seeds": args.seeds, "teacher_arm": args.teacher_arm,
                               "student_arm": args.student_arm, "grid": args.grid,
                               "res": args.res, "device": str(DEVICE), "pyramid": list(LEAN),
                               "widths": WIDTHS, "mac_budget": target,
                               "elapsed_s": round(time.time() - t_all, 1),
                               "cells": cells}, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
