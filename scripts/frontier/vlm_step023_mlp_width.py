"""vlm_step023: can the vision tower's MLP width be SLICED and distilled back?

WHY THIS AXIS, AND WHY NOW. The tower is 82% of end-to-end latency (step006). Of the tower's
structural axes: DEPTH is done (d6 = 2.0x at parity, step004/006); RESOLUTION is CLOSED -- step022's
training-free oracle proved no res-256 student can exceed 0.4160 vs the teacher's 0.7040, so the best
possible low-res arm is -28.8pp for 2.93x while depth got parity for 2.0x; READOUT/token pruning is
dead (step009). WIDTH is the one axis never touched, and it is also the only one that shrinks
PARAMS -- resolution shrank neither params nor bytes, which the drone goal cares about.

WHICH WIDTH. The tower has two width axes and they are separable:
  (a) residual width (hidden 768) -- changing it needs down/up adapters at the tower boundary,
      because patch embeddings feed 768 in and post_layernorm+connector expect 768 out. Messy, and
      the adapters are a second variable.
  (b) MLP intermediate width (3072) -- slicing it needs NO adapters at all: fc1 loses rows, fc2
      loses the matching columns, and the residual stream stays 768 end to end.
(b) is 2*768*3072 = 4.72M of each layer's 7.08M params, i.e. **67% of the tower**, and it is a
strictly one-variable change. So (b) first. This is also the same question this whole project is
built on -- SGNNET exists because a fully-connected block was replaceable -- asked of a VLM tower.

INIT: MAGNITUDE-RANKED SLICE, NOT RANDOM. Keep the top-m intermediate channels ranked by
||fc1_row_j|| * ||fc2_col_j||, the standard neuron-importance product: a channel matters only if it
both fires and is read. This is the exact analogue of depth truncation -- keep the teacher's own
most-used units, warm-start from its weights -- so step023 sits on the same methodological footing
as step004 rather than introducing a randomly-initialised student as a confound.

WHAT MOVES: only the sliced encoder layers. Patch embeddings, post_layernorm, connector and the
whole text stack stay frozen, as in every prior arm. Loss is relative MSE on post-connector
features, identical to step004/007/008/021, so the numbers stay comparable across the line.

PRE-REGISTERED GATES (fixed BEFORE the run):
  (1) VALIDITY. The teacher arm (d12 stock @512, no student) must land within +-2.0pp of 0.711.
  (2) NAIVE CONTROL per ratio. The sliced-but-UNTRAINED tower is evaluated in THIS run, so the
      distilled number is compared against a control from the same code path and eval sample. The
      trajectory's own history demands this: naive structural cuts to this tower read at chance
      (naive d6 = 0.120, naive res-256 = 0.160) and are uninformative on their own.
VERDICT RULE, fixed in advance, stated as an absolute bar because step022 taught that a
gain-over-naive rule can certify an arm that is still far below the teacher:
  WIN      iff distilled top-1 >= teacher - 2.0pp  (parity, the bar d6 actually cleared).
  PARTIAL  iff within 10pp of teacher but not parity.
  LOSS     iff more than 10pp below teacher -- MLP width carries information distillation does not
           recover at this budget, and the axis is reported closed at this budget.
T0 SCOUT budget. A null is 'at this budget', never a ceiling -- the loss curve is reported so a
still-falling curve cannot be read as convergence. Unlike step022 there is no training-free oracle
available here (a sliced MLP has no oracle counterpart), so that caveat is load-bearing.

Output: results/frontier/vlm_step023_mlp_width_{TAG}__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.frontier.vlm_distill import cache_teacher, park, train_student
from scripts.frontier.vlm_eval import VLMEval, evaluate, sample_images
from scripts.frontier.vlm_width import build_width_student, n_params

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=12, help="TEACHER depth; the baseline params are measured here")
parser.add_argument("--student_depth", type=int, default=None,
                    help="student layer count; defaults to --depth (width-only). Set below --depth "
                         "for the COMPOUND arm (vlm_step025: depth AND width together). Distillation "
                         "targets are unaffected either way -- cache_teacher always runs the FULL "
                         "tower -- so the student is compared against the same d12 teacher.")
parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.25])
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_eval", type=int, default=500)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load", nargs="+", default=None, metavar="CKPT",
                    help="one checkpoint per --ratios entry, in the same order: SKIP training and "
                         "re-score those exact weights (vlm_step031). NOT the same as --epochs 0, "
                         "which trains for zero steps and would silently score a SLICED-INIT "
                         "student. Nothing is written over: TAG carries the new n_eval.")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_train, args.n_eval, args.epochs, args.ratios = 40, 20, 1, [0.5]
if args.load and len(args.load) != len(args.ratios):
    parser.error(f"--load takes one checkpoint per ratio, in order: got {len(args.load)} "
                 f"for {len(args.ratios)} ratios. A silent mismatch would score the wrong width.")

SLOT = os.environ.get("SGN_SLOT", "local")
SD = args.depth if args.student_depth is None else args.student_depth
# The d-suffix appears ONLY for a compound run, so the width-only arms already written up
# (step023 15ep/1k, step024 25ep/9.3k) keep the exact filenames their results are quoted under.
TAG = ("r" + "-".join(str(r) for r in args.ratios) + f"_e{args.epochs}_t{args.n_train}"
       f"_n{args.n_eval}" + ("" if SD == args.depth else f"_d{SD}"))
OUT = ROOT / "results" / "frontier" / f"vlm_step023_mlp_width_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"
ANCHOR, GATE_PP, WIN_PP, PARTIAL_PP = 0.711, 2.0, 2.0, 10.0


def pick_device(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device(args.device)


def main():
    print("=" * 78, flush=True)
    print(f"vlm_step023 MLP width  device={DEVICE}  depth={args.depth}->{SD}  ratios={args.ratios}  "
          f"n_train={args.n_train} epochs={args.epochs}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
    ev = VLMEval(model.eval().to(DEVICE), AutoProcessor.from_pretrained(args.model), DEVICE)
    ev.set_depth(args.depth)

    train = [] if args.load else sample_images(DATA / "train", args.n_train, seed=args.seed)
    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    print(f"  {len(train)} train / {len(val)} eval images", flush=True)

    inter = ev.full[0].mlp.fc1.out_features
    base_params = n_params(nn.ModuleList(list(ev.full[:args.depth])))
    cells = {"teacher": evaluate(ev, val)}
    cells["teacher"]["tower_params"] = base_params
    tt = cells["teacher"]["top1"]
    d = abs(tt - ANCHOR) * 100
    print(f"  GATE validity  teacher d{args.depth} top1 {tt:.4f} vs {ANCHOR} "
          f"-> {'PASS' if d <= GATE_PP else 'FAIL'}   tower params {base_params/1e6:.2f}M "
          f"(mlp intermediate {inter})", flush=True)

    targets = None if args.load else cache_teacher(ev, train, args.bs)

    for r in args.ratios:
        m = int(round(inter * r))
        print(f"\n=== mlp ratio {r} ({inter} -> {m} intermediate) ===", flush=True)
        naive_st = build_width_student(ev, SD, m, DEVICE)
        ev.set_layers(naive_st)
        naive = evaluate(ev, val)
        pr = n_params(naive_st)
        print(f"  naive (sliced, no training)  top1 {naive['top1']:.4f}   "
              f"tower params {pr/1e6:.2f}M ({pr/base_params:.3f}x)", flush=True)

        if args.load:
            student = build_width_student(ev, SD, m, "cpu")
            student.load_state_dict(torch.load(args.load[args.ratios.index(r)], map_location="cpu"))
            student, hist = student.to(DEVICE), []
        else:
            ev.set_depth(args.depth)
            park(ev, "cpu")
            student, hist = train_student(ev, train, targets, SD, args.epochs, args.bs, args.lr,
                                          DEVICE, student=build_width_student(ev, SD, m, "cpu"))
            park(ev, DEVICE)
            # TAG in the name: a rematch at a different epoch/data budget must not overwrite the
            # checkpoint of the run whose number is already written up.
            torch.save(student.state_dict(),
                       ROOT / "results" / "frontier" / f"vlm_step023_mlp_r{r}_{TAG}.pt")

        ev.set_layers(student)
        dist = evaluate(ev, val)
        gap = (tt - dist["top1"]) * 100
        verdict = ("WIN" if gap <= WIN_PP else "PARTIAL" if gap <= PARTIAL_PP else "LOSS")
        print(f"  distilled top1 {dist['top1']:.4f}  ({gap:+.2f}pp below teacher, "
              f"{(dist['top1']-naive['top1'])*100:+.2f}pp over naive) -> {verdict}", flush=True)
        print(f"  tower {cells['teacher']['vision_ms']:.2f} -> {dist['vision_ms']:.2f} ms "
              f"({cells['teacher']['vision_ms']/dist['vision_ms']:.2f}x), params {pr/1e6:.2f}M "
              f"({pr/base_params:.3f}x)", flush=True)
        cells[str(r)] = {"intermediate": m, "naive": naive, "distilled": dist, "gap_pp": gap,
                         "gain_over_naive_pp": (dist["top1"] - naive["top1"]) * 100,
                         "verdict": verdict, "tower_params": pr,
                         "param_ratio": pr / base_params, "hist": hist,
                         "tower_speedup": cells["teacher"]["vision_ms"] / dist["vision_ms"]}
        ev.set_depth(args.depth)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"step": "vlm_step023", "depth": args.depth, "student_depth": SD,
                               "anchor": ANCHOR,
                               "intermediate": inter, "n_train": len(train), "epochs": args.epochs,
                               "lr": args.lr, "seed": args.seed, "device": str(DEVICE),
                               "cells": cells}, indent=2, default=str))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
