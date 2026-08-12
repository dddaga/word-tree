"""vlm_step006: is distil_d6 > teacher_d12 REAL, or 200-image noise?

vlm_step004's full-budget run (9352 train imgs x 25 ep) returned distil_d6 top-1 0.740 against
teacher_d12 0.685 on the SAME 200 val images -- the student BEAT the teacher it was distilled
from, at 16.58% fewer params and 1.61x faster vision. That is a large claim resting on 11 net
images, and the unpaired binomial SE at n=200 is +/-3.2pp, so it cannot be asserted from the
aggregate alone.

This step spends no training compute: it loads the saved student checkpoint and re-runs ONLY the
eval, at a larger n and with PER-IMAGE records kept, so the comparison becomes paired. The
deciding statistic is McNemar's exact test on the discordant pairs (student right/teacher wrong
vs teacher right/student wrong) -- the aggregate delta is compatible with anything from p=0.001
to p=0.19 depending on how the 59 disagreements split, which is exactly why they must be counted.

Arms: teacher_d12 and distil_d{depth} only. naive_d6 is omitted deliberately -- it is CONFIRMED
at chance by vlm_step003 and vlm_step004 (0.120/0.135) and would cost a third of the wall-time to
re-confirm. Pass --with_naive to include it anyway.

Output: results/frontier/vlm_step006_eval_scale_{TAG}__{SLOT}.json  (includes per-image records)
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, VLMEval, sample_images

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--ckpt", default="results/frontier/vlm_step004_tower_distill_d6_25ep9k__mini_mps.pt")
parser.add_argument("--n_eval", type=int, default=3900)
parser.add_argument("--seed", type=int, default=42, help="42 reproduces step004's eval set")
parser.add_argument("--split", default="val")
parser.add_argument("--with_naive", action="store_true")
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_eval, args.boot = 20, 200

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
    if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
TAG = f"d{args.depth}_n{args.n_eval}_s{args.seed}_{args.split}"
OUT = ROOT / "results" / "frontier" / f"vlm_step006_eval_scale_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. b = student right & teacher wrong, c = the reverse. Under H0 each
    discordant pair is a fair coin, so p = P(|X - n/2| >= |b - n/2|) for X ~ Bin(n, 1/2)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def boot_ci(a, b, n_boot, seed=0):
    """Paired percentile bootstrap on mean(a) - mean(b); a, b are 0/1 lists over the SAME images."""
    rng, n, out = random.Random(seed), len(a), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    out.sort()
    return out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


def load_student(ev):
    ck = Path(args.ckpt)
    if not ck.is_absolute(): ck = ROOT / ck
    student = nn.ModuleList([copy.deepcopy(l) for l in ev.full[:args.depth]])
    sd = torch.load(ck, map_location=DEVICE)
    student.load_state_dict(sd)
    return student.eval().to(DEVICE), ck


def evaluate(ev, arms, val):
    """Per-image records. The first arm is the reference for same/agree/KL, as in step004."""
    recs, acc = [], {a: dict.fromkeys(("correct", "same", "agree", "kl", "vis", "pre"), 0.0)
                     for a, _ in arms}
    t0 = time.perf_counter()
    for i, (path, wnid) in enumerate(val):
        img = Image.open(path).convert("RGB")
        gold = LABELS.index(WNID2LABEL[wnid])
        rec, ref = {"wnid": wnid, "gold": gold, "path": path.name}, None
        for name, mods in arms:
            ev.set_layers(mods)
            lg, pred, vm, pm = ev.run(img)
            lp = F.log_softmax(lg, -1)
            if ref is None: ref = (pred, lp, lg.argmax().item())
            a = acc[name]
            a["correct"] += pred == gold; a["same"] += pred == ref[0]
            a["agree"] += lg.argmax().item() == ref[2]
            a["kl"] += float(F.kl_div(lp, ref[1], log_target=True, reduction="sum"))
            a["vis"] += vm; a["pre"] += pm
            rec[name] = pred
        recs.append(rec)
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  eval [{i+1}/{len(val)}]  {el:.0f}s  eta {el/(i+1)*(len(val)-i-1):.0f}s",
                  flush=True)
    return acc, recs


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"{'='*78}\nvlm_step006 eval scale  device={DEVICE}  depth={args.depth}  "
          f"n_eval={args.n_eval}  seed={args.seed}  split={args.split}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    m.requires_grad_(False)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    total_p = sum(p.numel() for p in m.parameters())
    per_layer = sum(p.numel() for p in ev.full[0].parameters())
    student, ck = load_student(ev)
    print(f"  loaded student <- {ck}")

    val = sample_images(DATA / args.split, args.n_eval, seed=args.seed)
    print(f"  {len(val)} eval images")
    arms = [("teacher_d12", ev.full), (f"distil_d{args.depth}", list(student))]
    if args.with_naive:
        arms.insert(1, (f"naive_d{args.depth}", ev.full[:args.depth]))
    acc, recs = evaluate(ev, arms, val)

    n, cut = len(val), total_p - (len(ev.full) - args.depth) * per_layer
    sname = f"distil_d{args.depth}"
    t_ok = [int(r["teacher_d12"] == r["gold"]) for r in recs]
    s_ok = [int(r[sname] == r["gold"]) for r in recs]
    b = sum(1 for i in range(n) if s_ok[i] and not t_ok[i])   # student right, teacher wrong
    c = sum(1 for i in range(n) if t_ok[i] and not s_ok[i])
    p_val = mcnemar_exact(b, c)
    lo, hi = boot_ci(s_ok, t_ok, args.boot)
    ref_tot = (acc["teacher_d12"]["vis"] + acc["teacher_d12"]["pre"]) / n

    res = {"step": "vlm_step006", "model": args.model, "device": str(DEVICE),
           "depth": args.depth, "ckpt": str(ck), "n_eval": n, "seed": args.seed,
           "split": args.split, "total_params": total_p, "vision_layer_params": per_layer,
           "mcnemar": {"student_right_teacher_wrong": b, "teacher_right_student_wrong": c,
                       "p_two_sided": p_val, "delta_top1": (sum(s_ok) - sum(t_ok)) / n,
                       "boot95_lo": lo, "boot95_hi": hi, "n_boot": args.boot},
           "arms": {}, "records": recs}
    print(f"\n  {'arm':<14} {'params':>12} {'%save':>6} {'top1':>6} {'same':>6} {'agree':>6} "
          f"{'KL':>7} {'vis_ms':>8} {'pre_ms':>7} {'total':>8} {'speed':>7}")
    for name, _ in arms:
        d = acc[name]; v, p_ = d["vis"] / n, d["pre"] / n
        prm = total_p if name == "teacher_d12" else cut
        row = {"params": prm, "params_saved_pct": round(100 * (total_p - prm) / total_p, 2),
               "top1": d["correct"] / n, "same_class_as_teacher": d["same"] / n,
               "agree_with_teacher": d["agree"] / n, "kl_vs_teacher": d["kl"] / n,
               "vision_ms": round(v, 2), "prefill_ms": round(p_, 2),
               "total_ms": round(v + p_, 2), "speedup": round(ref_tot / (v + p_), 3)}
        res["arms"][name] = row
        print(f"  {name:<14} {prm:>12,} {row['params_saved_pct']:>6.2f} {row['top1']:>6.3f} "
              f"{row['same_class_as_teacher']:>6.3f} {row['agree_with_teacher']:>6.3f} "
              f"{row['kl_vs_teacher']:>7.3f} {v:>8.1f} {p_:>7.1f} {v+p_:>8.1f} "
              f"{row['speedup']:>6.2f}x")
    print(f"\n  McNemar: student-only-right b={b}  teacher-only-right c={c}  "
          f"delta {res['mcnemar']['delta_top1']:+.4f}  p={p_val:.5f}  "
          f"boot95 [{lo:+.4f}, {hi:+.4f}]")
    print("  VERDICT:", "student > teacher CONFIRMED" if (p_val < 0.05 and b > c) else
          ("teacher > student CONFIRMED" if (p_val < 0.05 and c > b) else "NOT significant"))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
