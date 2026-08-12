"""vlm_step007: does the d6 tower's teacher-PARITY survive off the distillation domain?

vlm_step004 distilled a 6-layer student off the 12-layer SigLIP tower (16.58% of the whole model
deleted, 1.60x faster vision) and vlm_step006 CONFIRMED it at teacher parity: top-1 0.713 vs 0.711,
McNemar p=0.813, bootstrap 95% CI [-1.3pp, +1.7pp] over 3859 val images. (An earlier +5.5pp
"overtake" at n=200 was noise and is withdrawn -- so there is no gain left to explain, and the
"capacity-limited regression smooths the teacher" HYPOTHESIS is DROPPED: the discordance split was
b=440 / c=432, balanced, where a denoiser predicts b >> c.)

What is still unknown is the part the drone recipe depends on. That parity was bought with 9352
UNLABELLED IN-DOMAIN images. If parity is a property of the METHOD, one distillation run ships
anywhere. If it is a property of the DOMAIN, every deployment must first collect unlabelled
target-domain frames. Those are very different amounts of fieldwork.

Design: hold out 5 of the 10 classes from the DISTILLATION data only. The readout stays the same
constrained 10-way choice over all 10 labels, so nothing about the task changes -- only which
classes the student's unlabelled images contained. Then eval on both halves separately:

    held-5 delta ~ 0 (and seen ~ 0)  ->  parity is method-level; distil on anything, ship anywhere
    held-5 delta < 0, seen-5 ~ 0     ->  parity is domain-scoped; collect target-domain frames
    both deltas < 0                  ->  half-data budget hurt regardless; read vs the control

The control that makes this tight costs nothing extra: the EXISTING all-10 student (step004's
checkpoint) is evaluated on the same two halves. It saw every class, so its seen-minus-held gap is
the baseline for "no domain restriction" -- without it, a seen/held gap could just be the two
halves differing in difficulty. step006 pins its expected value: the all-10 student sits at +0.2pp
over the full val set, so it should read ~0 on BOTH halves; if it does not, the halves differ in
difficulty and the seen-5 deltas must be read against that offset rather than against zero.

Budget is matched on UPDATES, not epochs: 5 classes give ~4676 train images, so --epochs 50 lands
on the same 233,800 image-updates as step004's 9352 x 25. Two caveats to state with any result:
(1) 50 passes over half the images overfits more than 25 over all of them, which biases TOWARD
looking domain-scoped; (2) n=1000/group has roughly half the resolving power of the n=3859 run that
just overturned a 5.5pp claim, so only interactions well outside +-2pp should be called.

Output: results/frontier/vlm_step007_domain_split_{TAG}__{SLOT}.json  (includes per-image records)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.frontier.vlm_distill import cache_teacher, load_student, train_student
from scripts.frontier.vlm_eval import (LABELS, WNID2LABEL, VLMEval, boot_ci, mcnemar_exact,
                                       sample_images)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--ckpt", default="results/frontier/vlm_step004_tower_distill_d6_25ep9k__mini_mps.pt",
                    help="all-10-class student from step004; the no-restriction control arm")
parser.add_argument("--n_train", type=int, default=5000, help="floors to ~935/class over 5 classes")
parser.add_argument("--n_eval", type=int, default=1000, help="per group, before the //5 floor")
parser.add_argument("--epochs", type=int, default=50, help="50 x 4676 == 25 x 9352 updates")
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_train, args.n_eval, args.epochs, args.batch, args.boot = 40, 20, 1, 4, 200

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
    if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
TAG = f"d{args.depth}_{args.epochs}ep"
OUT = ROOT / "results" / "frontier" / f"vlm_step007_domain_split_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"

# Alternating rather than first-5/last-5: sorted wnid order is arbitrary w.r.t. semantics, but
# alternating keeps any residual ordering effect from landing entirely on one side.
ALL_WNIDS = sorted(WNID2LABEL)
SEEN, HELD = ALL_WNIDS[0::2], ALL_WNIDS[1::2]


def evaluate(ev, arms, val, tag):
    """Per-image records over one group. First arm is the reference for same/agree/KL."""
    recs, acc = [], {a: dict.fromkeys(("correct", "same", "agree", "kl", "vis"), 0.0) for a, _ in arms}
    t0 = time.perf_counter()
    for i, (path, wnid) in enumerate(val):
        img = Image.open(path).convert("RGB")
        gold = LABELS.index(WNID2LABEL[wnid])
        rec, ref = {"group": tag, "wnid": wnid, "gold": gold, "path": path.name}, None
        for name, mods in arms:
            ev.set_layers(mods)
            lg, pred, vm, _ = ev.run(img)
            lp = F.log_softmax(lg, -1)
            if ref is None: ref = (pred, lp, lg.argmax().item())
            a = acc[name]
            a["correct"] += pred == gold; a["same"] += pred == ref[0]
            a["agree"] += lg.argmax().item() == ref[2]
            a["kl"] += float(F.kl_div(lp, ref[1], log_target=True, reduction="sum"))
            a["vis"] += vm
            rec[name] = pred
        recs.append(rec)
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  eval {tag} [{i+1}/{len(val)}] {el:.0f}s "
                  f"eta {el/(i+1)*(len(val)-i-1):.0f}s", flush=True)
    return acc, recs


def compare(recs, name, ref="teacher_d12"):
    """Paired student-vs-teacher stats on one group."""
    s = [int(r[name] == r["gold"]) for r in recs]
    t = [int(r[ref] == r["gold"]) for r in recs]
    b = sum(1 for i in range(len(s)) if s[i] and not t[i])
    c = sum(1 for i in range(len(s)) if t[i] and not s[i])
    lo, hi = boot_ci(s, t, args.boot)
    return {"n": len(s), "top1": sum(s) / len(s), "teacher_top1": sum(t) / len(t),
            "delta": (sum(s) - sum(t)) / len(s), "b": b, "c": c,
            "p_two_sided": mcnemar_exact(b, c), "boot95_lo": lo, "boot95_hi": hi}


def verdict(groups, inter, ctrl, tol=0.02):
    """Map the two deltas onto the docstring's three outcomes.

    `tol` is the call threshold, set to the step006 power caveat: at n=1000/group anything inside
    +-2pp is not resolvable, so it is read as parity rather than as a small effect. The control's
    own interaction must be SMALLER than the student's, else the seen/held halves simply differ in
    difficulty and the student's interaction is measuring that instead.
    """
    seen, held = (groups[g]["seen5_vs_teacher"]["delta"] for g in ("seen", "held"))
    if inter > tol and abs(ctrl) < abs(inter):
        return "domain-scoped parity -- drone MUST distil on target-domain frames"
    if max(abs(seen), abs(held)) <= tol:
        return "method-level parity -- distil on anything, ships off-domain"
    if seen < -tol and held < -tol:
        return "both halves degraded -- half-data budget, not domain; read vs the control"
    return "inconclusive"


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"{'='*78}\nvlm_step007 domain split  device={DEVICE}  depth={args.depth}  "
          f"epochs={args.epochs}  lr={args.lr}\n  SEEN {[WNID2LABEL[w] for w in SEEN]}"
          f"\n  HELD {[WNID2LABEL[w] for w in HELD]}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    ck = Path(args.ckpt)
    if not ck.is_absolute(): ck = ROOT / ck
    all10 = load_student(ev, ck, args.depth, DEVICE)
    print(f"  control student <- {ck}")

    train = sample_images(DATA / "train", args.n_train, seed=7, wnids=SEEN)
    print(f"  {len(train)} train images from {len(SEEN)} classes; "
          f"{len(train) * args.epochs:,} image-updates")
    student, hist = train_student(ev, train, cache_teacher(ev, train, args.batch),
                                  args.depth, args.epochs, args.batch, args.lr, DEVICE)

    seen5, all10n = f"seen5_d{args.depth}", f"all10_d{args.depth}"
    arms = [("teacher_d12", ev.full), (all10n, list(all10)), (seen5, list(student))]
    groups, recs = {}, []
    for tag, wn in (("seen", SEEN), ("held", HELD)):
        val = sample_images(DATA / "val", args.n_eval, seed=42, wnids=wn)
        acc, r = evaluate(ev, arms, val, tag)
        recs += r
        groups[tag] = {"n": len(val), "arms": {a: {k: v / len(val) for k, v in acc[a].items()}
                                               for a, _ in arms},
                       "seen5_vs_teacher": compare(r, seen5), "all10_vs_teacher": compare(r, all10n)}

    inter = groups["seen"]["seen5_vs_teacher"]["delta"] - groups["held"]["seen5_vs_teacher"]["delta"]
    ctrl = groups["seen"]["all10_vs_teacher"]["delta"] - groups["held"]["all10_vs_teacher"]["delta"]
    res = {"step": "vlm_step007", "depth": args.depth, "device": str(DEVICE), "ckpt": str(ck),
           "seen_wnids": SEEN, "held_wnids": HELD, "n_train": len(train), "epochs": args.epochs,
           "lr": args.lr, "batch": args.batch, "train_history": hist, "groups": groups,
           "interaction_seen5": inter, "interaction_all10_control": ctrl, "records": recs}
    for tag in ("seen", "held"):
        for arm, key in ((seen5, "seen5_vs_teacher"), (all10n, "all10_vs_teacher")):
            d = groups[tag][key]
            print(f"  {tag:<5} {arm:<12} n={d['n']:<5} top1 {d['top1']:.3f} vs teacher "
                  f"{d['teacher_top1']:.3f}  delta {d['delta']:+.4f}  b={d['b']} c={d['c']}  "
                  f"p={d['p_two_sided']:.5f}  boot95 [{d['boot95_lo']:+.4f}, {d['boot95_hi']:+.4f}]")
    print(f"\n  interaction (seen delta - held delta): seen5 student {inter:+.4f}   "
          f"all-10 control {ctrl:+.4f}")
    print("  VERDICT:", verdict(groups, inter, ctrl))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
