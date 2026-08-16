"""seg_step007: distil the expensive pretrained encoder into the cheap one.

seg_step004 CONFIRMED the E2-E7 gap is real and NOT a training-budget artifact: at 60 epochs
E2 = 0.2597 +-0.0037 vs E7 = 0.2079 +-0.0122, and the gap WIDENS with budget (4.29 -> 5.18pp).
Both arms peak at ep20-29, so neither is boundary-limited. That leaves the student short on
capacity/init, which is exactly the situation soft-KD fixed in the SGNNET line (step605: K=1
student recovered the K=5 teacher at 1/5 the cost).

  teacher  E2  drop block4, pretrained   3.2324 G  2,195,656 par   0.2597 @60ep
  student  E7  E2 + width 0.5x sliced    0.8757 G    851,816 par   0.2079 @60ep  <- under 1 GMAC

The teacher is trained ONCE (seed 42) and cached to disk; every student seed reuses it, so the
teacher cost is amortised and the student seeds stay cheap. Arms:

  S0  student, BCE on the teacher-rasterised targets only        = the seg_step004 E7 control
  S1  student, BCE + alpha * soft-KD on the teacher class logits

S0 is rerun here rather than read from seg_step004 so the control shares this script's exact code
path and seed set. seg_step006 CONFIRMED a ~+-2pp run-to-run floor on this task, so 5 seeds is the
minimum for a readable delta and 3-seed bars from earlier in the line understate noise.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.seg.seg_common import decor_penalty
from scripts.seg.seg_encoders import SegNet
from scripts.seg.seg_eval import BATCH, cache_targets, count_macs, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="S1", help="S0 (no KD control) / S1 (soft-KD)")
parser.add_argument("--teacher_arm", default="E2")
parser.add_argument("--student_arm", default="E7")
parser.add_argument("--device", default="auto")
parser.add_argument("--teacher_device", default="cpu")   # fasterrcnn HANGS on MPS
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--teacher_epochs", type=int, default=60)
parser.add_argument("--alpha", type=float, default=0.5, help="weight on the soft-KD term")
parser.add_argument("--temp", type=float, default=2.0)
parser.add_argument("--res", type=int, default=128)
parser.add_argument("--grid", type=int, default=16)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_val", type=int, default=500)
parser.add_argument("--decor_lambda", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--teacher_seed", type=int, default=42)
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="t0")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED = args.seed
SLOT = os.environ.get("SGN_SLOT", "local")


def fit(model, tx, ty, vx, vy, epochs, seed, teacher_logits=None):
    """One training run. With teacher_logits, adds the soft-KD term. Returns (best, best_ep, state)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    nb = (len(tx) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * nb,
                                                pct_start=0.1, anneal_strategy="cos")
    best, best_ep, best_state, t0 = 0.0, 0, None, time.time()
    g = torch.Generator().manual_seed(seed)
    T = args.temp
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tx), generator=g)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            out = model(tx[b].to(DEVICE))
            cl = model.class_logits(out)
            loss = F.binary_cross_entropy_with_logits(cl, ty[b].to(DEVICE))
            if teacher_logits is not None:
                # Sigmoid-KD: the head is multi-label (a cell can hold several event classes), so
                # the soft target is per-class sigmoid, not a softmax over classes. T^2 keeps the
                # KD gradient scale comparable to the hard term, as in Hinton et al.
                soft = torch.sigmoid(teacher_logits[b].to(DEVICE) / T)
                loss = loss + args.alpha * T * T * F.binary_cross_entropy_with_logits(cl / T, soft)
            loss = loss + args.decor_lambda * decor_penalty(out, args.k)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = evaluate(model, vx, vy, DEVICE)
        if m["mAP"] > best:
            best, best_ep = m["mAP"], ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    e{ep+1:3d}/{epochs}  mAP={m['mAP']:.4f}  best={best:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return best, best_ep, best_state


def get_teacher(tx, ty, vx, vy):
    """Train the teacher once (or load the cached best state) and return it in eval mode."""
    cp = ROOT / "results" / "seg" / (f"teacher_{args.teacher_arm}_e{args.teacher_epochs}"
                                     f"_seed{args.teacher_seed}__{SLOT}.pt")
    torch.manual_seed(args.teacher_seed)
    model = SegNet(args.teacher_arm, k=args.k).to(DEVICE)
    if cp.exists():
        d = torch.load(cp, map_location=DEVICE)
        model.load_state_dict(d["state"])
        print(f"  teacher loaded {cp.name}  best_mAP={d['best_mAP']:.4f}", flush=True)
    else:
        print(f"  training teacher {args.teacher_arm} ({args.teacher_epochs} ep) — once", flush=True)
        best, best_ep, state = fit(model, tx, ty, vx, vy, args.teacher_epochs, args.teacher_seed)
        model.load_state_dict(state)
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": state, "best_mAP": best, "best_ep": best_ep}, cp)
        print(f"  teacher best_mAP={best:.4f} @ep{best_ep} -> {cp.name}", flush=True)
    model.eval()
    return model


def teacher_class_logits(teacher, tx):
    """Precompute the teacher's class logits over the train set — one pass, then it is never used."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(tx), BATCH):
            outs.append(teacher.class_logits(teacher(tx[i:i + BATCH].to(DEVICE))).cpu())
    return torch.cat(outs)


def main():
    if args.smoke_test:
        torch.manual_seed(SEED)
        s = SegNet(args.student_arm, k=args.k).to(DEVICE)
        t = SegNet(args.teacher_arm, k=args.k).to(DEVICE)
        x = torch.zeros(2, 3, args.res, args.res, device=DEVICE)
        ok = s.class_logits(s(x)).shape == t.class_logits(t(x)).shape
        print(f"  {args.arm} student={count_macs(s, args.res)/1e9:.4f}G "
              f"teacher={count_macs(t, args.res)/1e9:.4f}G {'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    print(f"\n{'='*70}\nseg_step007 — {args.arm}  {args.teacher_arm} -> {args.student_arm}  "
          f"alpha={args.alpha} T={args.temp}\n  device={DEVICE}  seed={SEED}\n{'='*70}", flush=True)
    tx, ty = cache_targets("train", args.n_train, args.res, args.grid, args.teacher_device)
    vx, vy = cache_targets("val", args.n_val, args.res, args.grid, args.teacher_device)

    tl = None
    if args.arm == "S1":
        tl = teacher_class_logits(get_teacher(tx, ty, vx, vy), tx)

    torch.manual_seed(SEED)
    student = SegNet(args.student_arm, k=args.k).to(DEVICE)
    n_p = sum(p.numel() for p in student.parameters() if p.requires_grad)
    macs = count_macs(student, args.res)
    print(f"  student trainable={n_p:,}  MACs/frame={macs/1e9:.4f}G  (ceiling 1.0G)", flush=True)

    t0 = time.time()
    best, best_ep, _ = fit(student, tx, ty, vx, vy, args.epochs, SEED, tl)
    final = evaluate(student, vx, vy, DEVICE)
    print(f"  DONE: best_mAP={best:.4f} @ep{best_ep}  {final['ap_per_class']}  "
          f"{time.time()-t0:.0f}s")
    res = {"step": "seg_step007", "arm": args.arm, "teacher_arm": args.teacher_arm,
           "student_arm": args.student_arm, "alpha": args.alpha, "temp": args.temp,
           "epochs": args.epochs, "n_params": n_p, "macs": macs, "gmacs": round(macs / 1e9, 4),
           "best_mAP": round(best, 4), "best_ep": best_ep, "final": final, "seed": SEED,
           "elapsed_s": round(time.time() - t0, 1)}
    p = ROOT / "results" / "seg" / f"seg_step007_{args.arm}_{args.tag}_seed{SEED}__{SLOT}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
