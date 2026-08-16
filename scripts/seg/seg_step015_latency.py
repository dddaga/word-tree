"""seg_step015: the seg line's missing goal metric — wall-time and memory, not MACs.

Every seg result so far (steps 001-014) is reported in mAP and MACs. MACs are a PROXY. The project
charter requires the full Pareto (accuracy + params + FLOPs + wall-time + memory + Joules), and
bench_step608 already CONFIRMED this codebase can be dispatch-bound rather than FLOP-bound — the
K=1 student's 5.26x wall-time win did not track its FLOP ratio. So "E7 is 0.8757 GMAC, inside the
1 GMAC drone ceiling" does NOT establish that E7 is fast.

This measures latency directly across the arm ladder at batch 1 (a drone processes one frame at a
time — the deployment case, and the one where dispatch overhead dominates) and batch 32.

Weights are random: architecture alone determines latency, so no training or checkpoint is needed.
That also makes this cheap enough to re-run on any slot.

Decision rule pre-registered in EXPERIMENT_QUEUE.md before this file was written, keyed on the
measured latency(E2)/latency(E7) against their MAC ratio of 3.6917x:
    ratio >= 3.0  -> MACs predict latency; the 1 GMAC ceiling is a sound objective
    1.5-3.0       -> partially compute-bound; ceiling directionally right but overstated
    ratio <= 1.5  -> MACs do NOT predict latency here; the drone objective was the wrong one

Joules are deliberately NOT measured here: that needs nvidia-smi power telemetry (see
bench_step997_energy_joules.py), so it is 5060ti-only and currently blocked by a teammate job.

Usage:
    d_env/bin/python3 scripts/seg/seg_step015_latency.py --device mps
"""
from __future__ import annotations
import argparse, json, os, platform, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.seg.seg_encoders import ARMS, SegNet
from scripts.seg.seg_eval import count_macs

parser = argparse.ArgumentParser()
parser.add_argument("--arms", default="E9,E7,E10,E2",
                    help="ladder to time, cheapest first (E8 optional: same shape as E7)")
parser.add_argument("--device", default="auto")
parser.add_argument("--res", type=int, default=128)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--batches", default="1,32", help="batch 1 IS the drone case")
parser.add_argument("--repeats", type=int, default=200)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--no_pretrained", action="store_true",
                    help="build sliced/pretrained arms with random weights of the IDENTICAL shape. "
                         "Latency and peak memory are set by architecture alone, so this changes "
                         "nothing measured here — it only lets the bench run on a box without the "
                         "553 MB cached vgg16 checkpoint. Never use it for a training script.")
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--tag", default="lat")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SLOT = os.environ.get("SGN_SLOT", "local")


def sync():
    """Async backends queue kernels; without this we would time the dispatch, not the work."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def peak_mem_mb() -> float:
    """Peak allocator bytes since the last reset. CPU has no allocator counter -> nan."""
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if DEVICE.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    return float("nan")


def reset_mem():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def time_arm(model, batch: int) -> dict:
    """Per-repeat timing so we can report the MEDIAN and the spread, not just a mean.

    The median is the honest number for a latency claim: one scheduler hiccup or a background
    process moves a mean far more than it moves a median, and this box is shared.
    """
    x = torch.zeros(batch, 3, args.res, args.res, device=DEVICE)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        sync()
        reset_mem()
        samples = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            model(x)
            sync()
            samples.append((time.perf_counter() - t0) * 1e3)   # ms per forward
    samples.sort()
    med = statistics.median(samples)
    return {"batch": batch, "ms_per_forward": round(med, 4),
            "ms_per_frame": round(med / batch, 4),
            "p10": round(samples[len(samples) // 10], 4),
            "p90": round(samples[9 * len(samples) // 10], 4),
            "peak_mem_mb": round(peak_mem_mb(), 1)}


def main():
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    batches = [int(b) for b in args.batches.split(",")]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {sorted(ARMS)}")

    if args.no_pretrained:
        # 'pre' kwargs (n_blocks) and 'sliced' kwargs (width/n_blocks/reps) are both accepted by
        # build_vgg_shaped and produce the SAME layer shapes -- only the initial values differ.
        for a in arms:
            kind, kw, fb, _ = ARMS[a]
            ARMS[a] = ("shaped", kw, fb, False)
        print("  [--no_pretrained] arms built with random weights of identical shape", flush=True)

    if args.smoke_test:
        m = SegNet(arms[0], k=args.k).to(DEVICE).eval()
        x = torch.zeros(1, 3, args.res, args.res, device=DEVICE)
        with torch.no_grad():
            out = m(x)
        print(f"  {arms[0]} out={tuple(out.shape)} dev={DEVICE} OK")
        sys.exit(0)

    print(f"\n{'='*78}\nseg_step015 latency Pareto — {DEVICE}  res={args.res}  "
          f"repeats={args.repeats}\n  {platform.platform()}\n{'='*78}", flush=True)

    rows = []
    for arm in arms:
        torch.manual_seed(42)
        model = SegNet(arm, k=args.k).to(DEVICE).eval()
        n_p = sum(p.numel() for p in model.parameters())
        macs = count_macs(model, args.res)
        r = {"arm": arm, "params": n_p, "gmacs": round(macs / 1e9, 4),
             "timings": [time_arm(model, b) for b in batches]}
        rows.append(r)
        for t in r["timings"]:
            print(f"  {arm:<4} b{t['batch']:<3} {t['ms_per_forward']:8.3f} ms/fwd  "
                  f"{t['ms_per_frame']:8.3f} ms/frame  [p10 {t['p10']:.3f} p90 {t['p90']:.3f}]  "
                  f"mem {t['peak_mem_mb']:.0f} MB  {r['gmacs']:.4f}G", flush=True)
        del model

    # The pre-registered comparison: does the measured latency ratio track the MAC ratio?
    by_arm = {r["arm"]: r for r in rows}
    verdict = {}
    if "E2" in by_arm and "E7" in by_arm:
        mac_ratio = by_arm["E2"]["gmacs"] / by_arm["E7"]["gmacs"]
        print(f"\n  MAC ratio E2/E7 = {mac_ratio:.4f}x", flush=True)
        for i, b in enumerate(batches):
            lat = (by_arm["E2"]["timings"][i]["ms_per_frame"]
                   / by_arm["E7"]["timings"][i]["ms_per_frame"])
            branch = ("a MACs predict latency" if lat >= 3.0 else
                      "b partially compute-bound" if lat > 1.5 else
                      "c MACs do NOT predict latency")
            verdict[f"b{b}"] = {"latency_ratio": round(lat, 4),
                                "mac_ratio": round(mac_ratio, 4),
                                "efficiency": round(lat / mac_ratio, 4), "branch": branch}
            print(f"  b{b:<3} latency ratio {lat:.4f}x  -> {lat/mac_ratio*100:5.1f}% of the "
                  f"MAC ratio  -> branch {branch}", flush=True)

    res = {"step": "seg_step015", "device": str(DEVICE), "slot": SLOT, "res": args.res,
           "repeats": args.repeats, "warmup": args.warmup, "platform": platform.platform(),
           "rows": rows, "verdict": verdict}
    p = ROOT / "results" / "seg" / f"seg_step015_{args.tag}__{SLOT}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
