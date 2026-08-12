"""vlm_step008: is the 16.58% cut a floor or a ceiling? d3 and d9 at the budget that made d6 work.

vlm_step004+006 CONFIRMED a 6-layer distilled tower at teacher parity (0.713 vs 0.711, McNemar
p=0.813, boot95 [-1.3pp, +1.7pp], n=3859) for 16.58% of the model deleted and 1.60x faster vision.
That is the shipped lever. The drone wants more; d3 is the only untested route -- 24.87%, 2.33x.

d3 has been measured ONCE, at Tier-0 budget, where it read -28pp. Not evidence against d3: the same
budget read -18pp for d6, which recovered ALL of it when the budget grew, monotone and never flat
-- rel_mse 0.2573 (2000x10) -> 0.2011 (6000x15) -> 0.1639 (9352x25, the parity checkpoint). So the
binding constraint is CONFIRMED to be budget, not depth, over the range tested. This gives d3 the
exact budget d6 got; d9 interpolates, turning two numbers into a curve that can show a knee.

Teacher features are cached ONCE, and a depth whose checkpoint exists is reloaded rather than
retrained, so a crashed sweep restarts where it stopped. The d6 anchor is loaded the same way,
tying this to step006 on the SAME 3859 images (seed 42) -- if d6 misses 0.713, nothing is readable.

PRE-REGISTERED DECISION RULE (written before the run):
  * A depth SHIPS IN PLACE OF d6 iff its paired bootstrap 95% lower bound is >= -2.0pp vs teacher.
    The drone trades accuracy for params, and a worst-case 2pp buys d3 an extra 8.3pp of the model
    and 1.46x more vision speed over d6. Point estimates are NOT the test -- step006 showed a 5.3pp
    headline swing from eval noise alone.
  * KILLED iff the bootstrap UPPER bound is < -2.0pp (degradation established, not just unmeasured).
  * Anything between is INCONCLUSIVE at this n, and is reported as such, not rounded to a verdict.
  * Independent of ship/kill: if rel_mse is still FALLING over the last 5 epochs, that depth's
    accuracy is a FLOOR under this budget, not its ceiling -- a negative result with a falling curve
    licenses a longer run; a flat curve does not.

Output: results/frontier/vlm_step008_depth_budget_{TAG}__{SLOT}.json  (includes per-image records)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))
# The 5060ti's shared venv is pinned to transformers 4.44 (no Idefics3) with a mismatched
# torchvision. `vlm_libs` is a private --target install that shadows both for THIS process only.
if (_LIBS := ROOT / "vlm_libs").is_dir(): sys.path.insert(0, str(_LIBS))

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.frontier.vlm_distill import cache_teacher, load_student, park, train_or_resume
from scripts.frontier.vlm_eval import (LABELS, WNID2LABEL, VLMEval, boot_ci, mcnemar_exact,
                                       sample_images)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depths", type=int, nargs="+", default=[3, 9], help="trained ascending")
parser.add_argument("--anchor_ckpt", default="results/frontier/vlm_step004_tower_distill_d6_25ep9k__mini_mps.pt",
                    help="step004's d6 student, loaded not retrained; reproduces step006 or invalidates the run")
parser.add_argument("--anchor_depth", type=int, default=6)
parser.add_argument("--n_train", type=int, default=10000,
                    help="sample_images floors to 10*(n//10) -> 9469, NOT step004's 9352 (arg 9469)")
parser.add_argument("--n_eval", type=int, default=3900, help="seed 42 -> step006's 3859 images")
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_train, args.n_eval, args.epochs, args.batch, args.boot = 40, 20, 1, 4, 200

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
    if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local") + ("_smoke" if args.smoke_test else "")  # never resumable as real
TAG = f"d{'-'.join(map(str, args.depths))}_{args.epochs}ep"
OUT = ROOT / "results" / "frontier" / f"vlm_step008_depth_budget_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"


def evaluate(ev, arms, val):
    """Per-image records over every arm. First arm is the reference for same/agree/KL."""
    keys = ("correct", "same", "agree", "kl", "vis", "pre")
    recs, acc = [], {a: dict.fromkeys(keys, 0.0) for a, _ in arms}
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
            print(f"  eval [{i+1}/{len(val)}] {el:.0f}s eta {el/(i+1)*(len(val)-i-1):.0f}s",
                  flush=True)
    return acc, recs


def compare(recs, name, ref="teacher_d12"):
    """Paired student-vs-teacher stats. The bootstrap bounds ARE the pre-registered decision rule."""
    s = [int(r[name] == r["gold"]) for r in recs]
    t = [int(r[ref] == r["gold"]) for r in recs]
    b = sum(1 for i in range(len(s)) if s[i] and not t[i])
    c = sum(1 for i in range(len(s)) if t[i] and not s[i])
    lo, hi = boot_ci(s, t, args.boot)
    return {"n": len(s), "top1": sum(s) / len(s), "teacher_top1": sum(t) / len(t),
            "delta": (sum(s) - sum(t)) / len(s), "b": b, "c": c,
            "p_two_sided": mcnemar_exact(b, c), "boot95_lo": lo, "boot95_hi": hi}


def verdict(cmp, hist, tol=0.02):
    """Apply the docstring's pre-registered rule to one depth. `hist` may be None (loaded anchor)."""
    call = ("SHIP" if cmp["boot95_lo"] >= -tol else
            "KILLED" if cmp["boot95_hi"] < -tol else "INCONCLUSIVE")
    if hist is None or len(hist) < 6: return call, None
    # Falling loss over the last 5 epochs => this depth is budget-limited, so the number is a floor.
    slope = hist[-1]["rel_mse"] - hist[-6]["rel_mse"]
    return call, ("floor -- rel_mse still falling, longer run licensed" if slope < -1e-3
                  else "converged -- rel_mse flat, this is the depth's ceiling")


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"{'='*78}\nvlm_step008 depth x budget  device={DEVICE}  depths={args.depths}  "
          f"epochs={args.epochs}  lr={args.lr}  n_eval={args.n_eval}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    m.requires_grad_(False)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    total_p = sum(p.numel() for p in m.parameters())
    per_layer = sum(p.numel() for p in ev.full[0].parameters())

    train = sample_images(DATA / "train", args.n_train, seed=7)
    print(f"  {len(train)} train images; {len(train) * args.epochs:,} image-updates per depth")
    cks = {d: OUT.with_name(f"vlm_step008_d{d}_{args.epochs}ep__{SLOT}.pt") for d in sorted(args.depths)}
    targets = cache_teacher(ev, train, args.batch) if any(not c.exists() for c in cks.values()) else None

    arms, hists = [("teacher_d12", ev.full)], {}
    park(ev, "cpu")  # nothing below this line touches the text stack or the frozen teacher layers
    for d, ck_d in cks.items():
        st, hists[d] = train_or_resume(ev, ck_d, train, targets, d, args.epochs, args.batch,
                                       args.lr, DEVICE)
        arms.append((f"distil_d{d}", list(st)))
    park(ev, DEVICE)

    ck = Path(args.anchor_ckpt)  # loaded AFTER training so it costs nothing during the backward pass
    if not ck.is_absolute(): ck = ROOT / ck
    if ck.exists():
        arms.insert(1, (f"distil_d{args.anchor_depth}",
                        list(load_student(ev, ck, args.anchor_depth, DEVICE))))
        print(f"  anchor d{args.anchor_depth} <- {ck}")
    else:
        print(f"  WARNING: anchor checkpoint missing ({ck}); run is NOT anchored to step006")

    val = sample_images(DATA / "val", args.n_eval, seed=args.seed)
    print(f"\n  {len(val)} eval images; {len(arms)} arms", flush=True)
    acc, recs = evaluate(ev, arms, val)

    n = len(val)
    ref_tot = (acc["teacher_d12"]["vis"] + acc["teacher_d12"]["pre"]) / n
    res = {"step": "vlm_step008", "device": str(DEVICE), "depths": args.depths,
           "anchor_ckpt": str(ck), "n_train": len(train), "epochs": args.epochs, "lr": args.lr,
           "batch": args.batch, "n_eval": n, "seed": args.seed, "total_params": total_p,
           "vision_layer_params": per_layer,
           "train_history": {str(k): v for k, v in hists.items()}, "arms": {}, "records": recs}
    print(f"\n  {'arm':<14} {'params':>12} {'%save':>6} {'top1':>6} {'same':>6} {'agree':>6} "
          f"{'KL':>7} {'vis_ms':>8} {'total':>8} {'speed':>7}")
    for name, mods in arms:
        d_ = acc[name]; v, p_ = d_["vis"] / n, d_["pre"] / n
        prm = total_p - (len(ev.full) - len(mods)) * per_layer
        row = {"depth": len(mods), "params": prm,
               "params_saved_pct": round(100 * (total_p - prm) / total_p, 2),
               "top1": d_["correct"] / n, "same_class_as_teacher": d_["same"] / n,
               "agree_with_teacher": d_["agree"] / n, "kl_vs_teacher": d_["kl"] / n,
               "vision_ms": round(v, 2), "prefill_ms": round(p_, 2),
               "total_ms": round(v + p_, 2), "speedup": round(ref_tot / (v + p_), 3),
               "vision_speedup": round((acc["teacher_d12"]["vis"] / n) / v, 3)}
        res["arms"][name] = row
        print(f"  {name:<14} {prm:>12,} {row['params_saved_pct']:>6.2f} {row['top1']:>6.3f} "
              f"{row['same_class_as_teacher']:>6.3f} {row['agree_with_teacher']:>6.3f} "
              f"{row['kl_vs_teacher']:>7.3f} {v:>8.1f} {v+p_:>8.1f} {row['speedup']:>6.2f}x")
    for name, mods in arms[1:]:
        cmp = compare(recs, name)
        call, budget = verdict(cmp, hists.get(len(mods)))
        res["arms"][name]["vs_teacher"] = cmp
        res["arms"][name]["verdict"] = call
        res["arms"][name]["budget_state"] = budget
        print(f"  {name:<14} delta {cmp['delta']:+.4f}  b={cmp['b']} c={cmp['c']}  "
              f"p={cmp['p_two_sided']:.5f}  boot95 [{cmp['boot95_lo']:+.4f}, {cmp['boot95_hi']:+.4f}]"
              f"  -> {call}" + (f"  [{budget}]" if budget else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
