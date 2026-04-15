"""Step 830: K=4 vs K=5 direct wall-clock measurement (5060ti CUDA).

MOTIVATION
==========
V3 Gap Analysis Blocker-7: our paper currently says "K=4 saves 20% wall-clock"
projected from 0.280ms × 0.8 = 0.224ms. That is an ESTIMATE not a MEASUREMENT.
Paper review will demand a real number.

bench_step811 measured K=5 at 0.280ms (max-autotune fp32). This script runs
the same variants with K_iter=4 and compares directly.

VARIANTS (N=2048, D=16, K_hh=2, bs=32 — same as step811)
  K5_eager        : eager fp32 K=5                        [baseline ref]
  K5_ro           : compile reduce-overhead fp32 K=5
  K5_ma           : compile max-autotune fp32 K=5         [step811 best: 0.280ms]
  K4_eager        : eager fp32 K=4
  K4_ro           : compile reduce-overhead fp32 K=4
  K4_ma           : compile max-autotune fp32 K=4         [expected ~0.224ms]

PAPER CLAIM TO VALIDATE
  "K_iter=4 gives 20% wall-clock reduction vs K=5 at the same accuracy."
Acceptance criterion: K4_ma / K5_ma ≤ 0.85. Otherwise paper claim must be
revised to the measured ratio.

To run:
    python -u scripts/bench_step830_k4_wallclock.py --device cuda
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
parser.add_argument("--iters",  type=int, default=300)
parser.add_argument("--warmup", type=int, default=80)
parser.add_argument("--bs",     type=int, default=32)
args = parser.parse_args()

DEVICE = torch.device(args.device)
assert DEVICE.type == "cuda", "step830 is CUDA-only"
print(f"GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")

N_IN = 25088; N_OUT = 10; SEED = 42
N = 2048; D = 16; K_HH = 2; K_IN = 25

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"bench_step830_k4_wallclock__{SLOT}.json"


def make_sgnnet(K_iter: int):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
                              n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
                          beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                          resonance_threshold=0.0)
    return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


def sync(): torch.cuda.synchronize()


def bench(fn, x, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup): fn(x)
        sync()
        ts = []
        for _ in range(iters):
            sync(); t0 = time.perf_counter()
            fn(x)
            sync(); ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return {
        "median_ms":      float(np.median(ts)),
        "p95_ms":         float(np.percentile(ts, 95)),
        "p99_ms":         float(np.percentile(ts, 99)),
        "mean_ms":        float(np.mean(ts)),
        "throughput_sps": float(args.bs / (np.median(ts) / 1000.0)),
    }


def run_variant(K_iter: int, compile_mode: str | None, desc: str):
    m = make_sgnnet(K_iter=K_iter).to(DEVICE).eval()
    if compile_mode:
        m = torch.compile(m, mode=compile_mode, dynamic=False)
    x = torch.randn(args.bs, N_IN, device=DEVICE)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    r = bench(m, x, args.warmup, args.iters)
    r["peak_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    r["desc"] = desc
    del m; torch.cuda.empty_cache()
    return r


def main():
    print(f"\nStep 830 — K=4 vs K=5 direct wall-clock, N={N} D={D} K_hh={K_HH} bs={args.bs}")
    variants = [
        ("K5_eager", 5, None,             "K=5 eager fp32"),
        ("K5_ro",    5, "reduce-overhead","K=5 compile reduce-overhead"),
        ("K5_ma",    5, "max-autotune",   "K=5 compile max-autotune"),
        ("K4_eager", 4, None,             "K=4 eager fp32"),
        ("K4_ro",    4, "reduce-overhead","K=4 compile reduce-overhead"),
        ("K4_ma",    4, "max-autotune",   "K=4 compile max-autotune"),
    ]

    results = {"device": str(DEVICE), "bs": args.bs, "variants": {}}
    for key, K_iter, mode, desc in variants:
        print(f"\n── {key}: {desc} ──")
        try:
            results["variants"][key] = run_variant(K_iter, mode, desc)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results["variants"][key] = {"error": str(exc)}

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Variant':<12} {'median_ms':>10} {'p95_ms':>8} {'sps':>9} {'mib':>7}")
    for k, r in results["variants"].items():
        if "error" in r:
            print(f"{k:<12}  ERROR: {r['error']}")
            continue
        print(f"{k:<12} {r['median_ms']:>10.3f} {r['p95_ms']:>8.3f} "
              f"{r['throughput_sps']:>9.0f} {r.get('peak_mib', -1):>7.1f}")

    # Paper-claim check
    k5_ma = results["variants"].get("K5_ma", {}).get("median_ms")
    k4_ma = results["variants"].get("K4_ma", {}).get("median_ms")
    if k5_ma and k4_ma:
        ratio = k4_ma / k5_ma
        pct = (1 - ratio) * 100
        print(f"\nK4_ma / K5_ma = {ratio:.3f} → K=4 is {pct:+.1f}% {'faster' if ratio<1 else 'slower'}")
        print(f"Paper claim is '20% wall-clock reduction' → {'VALIDATED' if ratio <= 0.85 else 'REVISE to ' + f'{pct:.1f}%'}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
