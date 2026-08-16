"""seg_step010: feature-level hint on top of logit KD, exploiting the sliced init.

seg_step009 CONFIRMED logit soft-KD is worth +1.51pp (n=10 paired, p~0.03) and seg_step008
CONFIRMED it is FLAT in alpha over 16x — so the logit channel is saturated and more of it buys
nothing. The remaining gap to the teacher is -3.7pp. The next place to look is the feature channel.

The usual FitNets hint needs a learned projection because student and teacher widths differ. Here it
does NOT: E7 is built by SLICING E2's convs (seg_step003), so student channel j IS teacher channel j
for j < 128, by construction. The hint is therefore a plain MSE against the teacher's first 128
encoder channels — zero extra parameters, zero architectural choice, and no projection to confound
the result. If channel correspondence has decayed during training, the hint pulls it back.

  H0  logit KD only (= seg_step008 S1 alpha=2.0)          control, already measured seeds 42-51
  H1  logit KD + beta * MSE(student_enc, teacher_enc[:, :C_s])

Teacher features are recomputed per batch under no_grad rather than cached: 2000x256x16x16 float32
is 0.5 GB, and the teacher forward is only 3.2 GMAC.

Paired against the seg_step007/009 S0 and a2 cells on the SAME seeds (42-51). seg_step008 CONFIRMED
the noise is seed-level, so 10 seeds is the minimum readable comparison and rerunning seeds is waste.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.seg.seg_encoders import SegNet
from scripts.seg.seg_eval import cache_targets, count_macs, evaluate
from scripts.seg.seg_train import fit, load_teacher

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="H1", help="H0 (logit KD only) / H1 (+ feature hint)")
parser.add_argument("--teacher_arm", default="E2")
parser.add_argument("--student_arm", default="E7")
parser.add_argument("--device", default="auto")
parser.add_argument("--teacher_device", default="cpu")   # fasterrcnn HANGS on MPS
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--teacher_epochs", type=int, default=60)
parser.add_argument("--alpha", type=float, default=2.0, help="logit KD weight (flat per step008)")
parser.add_argument("--beta", type=float, default=1.0, help="feature hint weight")
parser.add_argument("--shuffle_hint", action="store_true",
                    help="seg_step011 control: permute the teacher's hint channels with a fixed "
                         "permutation. Same statistics, channel alignment destroyed — separates "
                         "'the sliced init aligns channels' from 'feature-MSE is a regulariser'.")
parser.add_argument("--temp", type=float, default=2.0)
parser.add_argument("--res", type=int, default=128)
parser.add_argument("--grid", type=int, default=16)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_val", type=int, default=500)
parser.add_argument("--decor_lambda", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--teacher_seed", type=int, default=42)
parser.add_argument("--save_ckpt", action="store_true",
                    help="seg_step014: persist the trained student as a teacher checkpoint, in the "
                         "same format load_teacher() expects, so a student that beat its own teacher "
                         "can supervise the next round.")
parser.add_argument("--ckpt_suffix", default="",
                    help="appended to the checkpoint stem. The default name keys only on "
                         "(arm, epochs, seed, slot), so any sweep that varies a LOSS term while "
                         "holding those fixed silently overwrites its own cells. Empty by default, "
                         "so existing checkpoints and load_teacher() are unaffected.")
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="h0")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED = args.seed
SLOT = os.environ.get("SGN_SLOT", "local")


def main():
    if args.smoke_test:
        torch.manual_seed(SEED)
        s = SegNet(args.student_arm, k=args.k).to(DEVICE)
        t = SegNet(args.teacher_arm, k=args.k).to(DEVICE)
        x = torch.zeros(2, 3, args.res, args.res, device=DEVICE)
        se, te = s.enc(x), t.enc(x)
        ok = se.shape[2:] == te.shape[2:] and se.shape[1] <= te.shape[1]
        print(f"  {args.arm} s_enc={tuple(se.shape)} t_enc={tuple(te.shape)} "
              f"student={count_macs(s, args.res)/1e9:.4f}G {'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    print(f"\n{'='*70}\nseg_step010 — {args.arm}  {args.teacher_arm} -> {args.student_arm}  "
          f"alpha={args.alpha} beta={args.beta}\n  device={DEVICE}  seed={SEED}\n{'='*70}", flush=True)
    tx, ty = cache_targets("train", args.n_train, args.res, args.grid, args.teacher_device)
    vx, vy = cache_targets("val", args.n_val, args.res, args.grid, args.teacher_device)
    teacher = load_teacher(args, DEVICE, SLOT)

    torch.manual_seed(SEED)
    student = SegNet(args.student_arm, k=args.k).to(DEVICE)
    n_p = sum(p.numel() for p in student.parameters() if p.requires_grad)
    macs = count_macs(student, args.res)
    print(f"  student trainable={n_p:,}  MACs/frame={macs/1e9:.4f}G  (ceiling 1.0G)", flush=True)

    t0 = time.time()
    best, best_ep, best_state = fit(student, teacher, tx, ty, vx, vy, args, DEVICE, SEED)
    final = evaluate(student, vx, vy, DEVICE)
    if best_state is not None:
        # Written in load_teacher()'s exact naming scheme so the next round can consume it directly.
        cp = ROOT / "results" / "seg" / (f"teacher_{args.student_arm}_e{args.epochs}"
                                         f"_seed{SEED}{args.ckpt_suffix}__{SLOT}.pt")
        torch.save({"state": best_state, "best_mAP": best}, cp)
        print(f"  saved teacher checkpoint {cp.name}  best_mAP={best:.4f}", flush=True)
    print(f"  DONE: best_mAP={best:.4f} @ep{best_ep}  {final['ap_per_class']}  "
          f"{time.time()-t0:.0f}s")
    res = {"step": "seg_step010", "arm": args.arm, "alpha": args.alpha, "beta": args.beta,
           "teacher_arm": args.teacher_arm, "student_arm": args.student_arm,
           "epochs": args.epochs, "n_params": n_p, "macs": macs, "gmacs": round(macs / 1e9, 4),
           "best_mAP": round(best, 4), "best_ep": best_ep, "final": final, "seed": SEED,
           "elapsed_s": round(time.time() - t0, 1)}
    p = ROOT / "results" / "seg" / f"seg_step010_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
