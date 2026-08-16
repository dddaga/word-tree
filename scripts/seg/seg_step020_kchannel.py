"""seg_step020: does anti-Hebbian decorrelation earn its keep on EXPLAINABILITY?

Decorrelation is a CONFIRMED NULL on mAP (7 paired seeds, seg_coarse_heatmap.md). It was kept on
one explicit condition: that the K-channel explainability it was built for gets measured with its
own metric, not mAP. This is that metric, and it closes the last open claim for the mechanism.

The claim being tested: with K=2 channels per event class, the operator reads confidence off
max_k and reads *which sub-pattern fired* off argmax_k. That story only works if the two channels
are (i) both alive and (ii) encoding different things. So, over the val cells the teacher marks
positive for class c, using the PRE-max channel logits:

    balance_c        = 2 * min(p, 1-p),  p = fraction of those cells with argmax_k == 0
                       1.0 = both channels used, 0.0 = one channel dead (the gate-death mode)
    specialisation_c = 1 - |pearson(ch0, ch1)| over those cells
                       this is the actual claim decorrelation makes

Reported as the mean over the 4 classes. An untrained random-init E7 is measured as a FLOOR, so a
high specialisation is not mistaken for structure when it is only uncorrelated noise.

Analysis is offline on saved checkpoints so the metric can be revised and re-run without paying
for training again. Decision rule pre-registered in EXPERIMENT_QUEUE.md before any checkpoint
existed: decor wins only on mean dS >= +0.05 AND t >= 2.78 (df=4) with balance not reduced.

Usage:
    d_env/bin/python3 scripts/seg/seg_step020_kchannel.py --device mps --seeds 42,43,44,45,46
"""
from __future__ import annotations
import argparse, json, os, statistics, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.seg.seg_common import C, EVENT_CLASSES
from scripts.seg.seg_encoders import SegNet
from scripts.seg.seg_eval import cache_targets

parser = argparse.ArgumentParser()
parser.add_argument("--student_arm", default="E7")
parser.add_argument("--epochs", type=int, default=60, help="matches the checkpoint stem")
parser.add_argument("--seeds", default="42,43,44,45,46")
parser.add_argument("--suffixes", default="_dec05,_dec00", help="treatment first, control second")
parser.add_argument("--device", default="auto")
parser.add_argument("--teacher_device", default="cpu")
parser.add_argument("--res", type=int, default=128)
parser.add_argument("--grid", type=int, default=16)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--n_val", type=int, default=500)
parser.add_argument("--pos_thresh", type=float, default=0.5,
                    help="a cell counts as positive for class c when its soft target exceeds this")
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="k")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")
SEEDS = [int(s) for s in args.seeds.split(",")]
SUFFIXES = [s.strip() for s in args.suffixes.split(",")]


def ckpt_path(seed: int, suffix: str) -> Path:
    return ROOT / "results" / "seg" / (f"teacher_{args.student_arm}_e{args.epochs}"
                                       f"_seed{seed}{suffix}__{SLOT}.pt")


def channel_stats(model, vx, vy) -> dict:
    """Balance and specialisation per class, over the cells the teacher calls positive.

    Restricting to positive cells is the whole point: on background cells both channels are
    driven to the same floor, so their correlation there measures the background, not the
    sub-pattern structure the mechanism claims to create.
    """
    outs = []
    with torch.no_grad():
        for i in range(0, vx.shape[0], 32):
            outs.append(model(vx[i:i + 32].to(DEVICE)).cpu())
    out = torch.cat(outs)                                   # [N, C*k, g, g]
    n, _, g, _ = out.shape
    # SegNet.class_logits does .amax(2) on exactly this view -- we keep the K axis instead.
    ch = out.view(n, C, args.k, g * g)
    tgt = vy.reshape(n, C, g * g)

    per_class = {}
    for c in range(C):
        mask = tgt[:, c] > args.pos_thresh                  # [N, g*g]
        if mask.sum() < 2:
            per_class[EVENT_CLASSES[c]] = {"n_pos": int(mask.sum()), "balance": float("nan"),
                                           "spec": float("nan")}
            continue
        a = ch[:, c, 0][mask]
        b = ch[:, c, 1][mask]
        p = (a > b).float().mean().item()
        av, bv = a - a.mean(), b - b.mean()
        denom = av.norm() * bv.norm()
        r = (av @ bv / denom).item() if denom > 0 else 0.0
        per_class[EVENT_CLASSES[c]] = {"n_pos": int(mask.sum()),
                                       "balance": round(2 * min(p, 1 - p), 4),
                                       "spec": round(1 - abs(r), 4)}
    vals = [v for v in per_class.values() if v["n_pos"] >= 2]
    return {"per_class": per_class,
            "balance": round(statistics.fmean(v["balance"] for v in vals), 4),
            "spec": round(statistics.fmean(v["spec"] for v in vals), 4)}


def load_model(seed: int, suffix: str):
    p = ckpt_path(seed, suffix)
    if not p.exists():
        raise SystemExit(f"missing checkpoint {p}")
    m = SegNet(args.student_arm, k=args.k).to(DEVICE).eval()
    m.load_state_dict(torch.load(p, map_location="cpu")["state"])
    return m


def main():
    vx, vy = cache_targets("val", args.n_val, args.res, args.grid, args.teacher_device)

    if args.smoke_test:
        torch.manual_seed(0)
        m = SegNet(args.student_arm, k=args.k).to(DEVICE).eval()
        s = channel_stats(m, vx[:32], vy[:32])
        print(f"  random-init on 32 val images: balance={s['balance']} spec={s['spec']}  OK")
        sys.exit(0)

    torch.manual_seed(0)
    floor = channel_stats(SegNet(args.student_arm, k=args.k).to(DEVICE).eval(), vx, vy)
    print(f"\n  FLOOR (random init)   balance={floor['balance']:.4f}  spec={floor['spec']:.4f}",
          flush=True)

    rows, d_spec, d_bal = {s: [] for s in SUFFIXES}, [], []
    for seed in SEEDS:
        st = {s: channel_stats(load_model(seed, s), vx, vy) for s in SUFFIXES}
        for s in SUFFIXES:
            rows[s].append(st[s])
        d_spec.append(st[SUFFIXES[0]]["spec"] - st[SUFFIXES[1]]["spec"])
        d_bal.append(st[SUFFIXES[0]]["balance"] - st[SUFFIXES[1]]["balance"])
        print(f"  seed {seed}  " + "  ".join(
            f"{s}: bal={st[s]['balance']:.4f} spec={st[s]['spec']:.4f}" for s in SUFFIXES)
            + f"   dspec={d_spec[-1]:+.4f} dbal={d_bal[-1]:+.4f}", flush=True)

    def paired(d):
        mean = statistics.fmean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        t = mean / (sd / len(d) ** 0.5) if sd > 0 else float("inf") if mean else 0.0
        return {"mean": round(mean, 4), "sd": round(sd, 4), "t": round(t, 3),
                "n": len(d), "pos": sum(1 for x in d if x > 0)}

    ps, pb = paired(d_spec), paired(d_bal)
    win = ps["mean"] >= 0.05 and ps["t"] >= 2.78 and pb["mean"] >= 0
    if len(d_spec) < 5:
        # The rule was pre-registered at df=4. With fewer pairs sd is underdetermined (at n=1 it is
        # 0, so t is +inf and ANY positive mean would "win"), so refuse to name a branch at all
        # rather than print a verdict the design cannot support.
        branch = f"NO VERDICT — {len(d_spec)} pair(s), rule needs 5"
    else:
        branch = ("a decorrelation EARNS its keep on explainability" if win else
                  "b CONFIRMED NULL on explainability too — drop the mechanism")
    print(f"\n  dspec mean {ps['mean']:+.4f} sd {ps['sd']:.4f} t {ps['t']:+.2f} "
          f"{ps['pos']}/{ps['n']} positive")
    print(f"  dbal  mean {pb['mean']:+.4f} sd {pb['sd']:.4f} t {pb['t']:+.2f} "
          f"{pb['pos']}/{pb['n']} positive")
    print(f"  -> branch {branch}", flush=True)

    res = {"step": "seg_step020", "device": str(DEVICE), "slot": SLOT, "seeds": SEEDS,
           "suffixes": SUFFIXES, "pos_thresh": args.pos_thresh, "floor": floor,
           "rows": rows, "d_spec": ps, "d_balance": pb, "branch": branch}
    p = ROOT / "results" / "seg" / f"seg_step020_{args.tag}__{SLOT}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
