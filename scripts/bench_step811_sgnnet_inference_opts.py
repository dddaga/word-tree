"""Step 811: SGNNET inference optimization experiments.

MOTIVATION
==========
bench_step810 baseline (5060ti, torch.compile reduce-overhead):
  SGNNET_AH @ N=2048: inference 0.299ms (5.24× faster than VGG_FC, but 3.4× slower than Linear).

Goal: close the SGNNET vs MLP inference gap via inference-only alterations.
Training optimizations deferred (step801 showed bf16+GradScaler HURTS).

VARIANTS PROFILED (N=2048, D=16, K_hh=2, K_iter=5, bs=32)
  Baseline: eager  fp32
  V1: torch.compile(reduce-overhead) fp32         [step810 winner]
  V2: torch.compile(max-autotune)    fp32         [longer compile, may win for inference]
  V3: torch.compile(reduce-overhead) fp16 (half)  [inference-only; no GradScaler]
  V4: torch.compile(reduce-overhead) bf16         [native Ada support; no GradScaler]
  V5: CUDA Graph capture of K_iter loop (if feasible — fixed-shape inference)

Outputs per-variant: median/p95/p99 latency, throughput, peak memory.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--iters",  type=int, default=300)
parser.add_argument("--warmup", type=int, default=80)
parser.add_argument("--bs",     type=int, default=32)
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
assert DEVICE.type == "cuda", "step811 is CUDA-only"

print(f"GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")
N_IN = 25088; N_OUT = 10; SEED = 42

OUT_PATH = ROOT / "results" / "bench_step811_sgnnet_inference_opts.json"


def make_sgnnet():
    torch.manual_seed(SEED)
    N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
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
        "median_ms": float(np.median(ts)),
        "p95_ms":    float(np.percentile(ts, 95)),
        "p99_ms":    float(np.percentile(ts, 99)),
        "mean_ms":   float(np.mean(ts)),
        "throughput_sps": float(args.bs / (np.median(ts) / 1000.0)),
    }


def peak_mem():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Variants
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline():
    m = make_sgnnet().to(DEVICE).eval()
    x = torch.randn(args.bs, N_IN, device=DEVICE)
    reset_mem()
    r = bench(m, x, args.warmup, args.iters)
    r["peak_mib"] = peak_mem()
    del m; torch.cuda.empty_cache()
    return r


def run_compile(mode: str, dtype=torch.float32, desc: str = ""):
    print(f"  compiling: mode={mode} dtype={dtype}")
    m = make_sgnnet().to(DEVICE).eval()
    if dtype != torch.float32:
        m = m.to(dtype=dtype)
    m = torch.compile(m, mode=mode, dynamic=False)
    x = torch.randn(args.bs, N_IN, device=DEVICE, dtype=dtype)
    reset_mem()
    r = bench(m, x, args.warmup, args.iters)
    r["peak_mib"] = peak_mem()
    r["desc"] = desc
    del m; torch.cuda.empty_cache()
    return r


def run_cuda_graph():
    """CUDA Graph capture of a fixed-shape fp32 forward pass."""
    m = make_sgnnet().to(DEVICE).eval()
    x = torch.randn(args.bs, N_IN, device=DEVICE)

    # Warm up stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(3):
                _ = m(x)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    static_x = torch.empty_like(x)
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_out = m(static_x)

    # Warmup runs
    for _ in range(args.warmup):
        static_x.copy_(x); g.replay()
    sync()

    # Timed
    ts = []
    for _ in range(args.iters):
        sync(); t0 = time.perf_counter()
        static_x.copy_(x); g.replay()
        sync(); ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    mem = peak_mem()
    del m; torch.cuda.empty_cache()
    return {
        "median_ms": float(np.median(ts)),
        "p95_ms":    float(np.percentile(ts, 95)),
        "p99_ms":    float(np.percentile(ts, 99)),
        "mean_ms":   float(np.mean(ts)),
        "throughput_sps": float(args.bs / (np.median(ts) / 1000.0)),
        "peak_mib": mem,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {"device": str(DEVICE), "bs": args.bs, "variants": {}}

    print("\n── V0: eager fp32 ──")
    results["variants"]["V0_eager"] = run_baseline()

    print("\n── V1: compile reduce-overhead fp32 ──")
    try:
        results["variants"]["V1_reduce_overhead"] = run_compile("reduce-overhead", torch.float32, "reduce-overhead fp32")
    except Exception as e:
        print(f"  V1 failed: {e}")

    print("\n── V2: compile max-autotune fp32 ──")
    try:
        results["variants"]["V2_max_autotune"] = run_compile("max-autotune", torch.float32, "max-autotune fp32")
    except Exception as e:
        print(f"  V2 failed: {e}")

    print("\n── V3: compile reduce-overhead fp16 (inference only) ──")
    try:
        results["variants"]["V3_fp16"] = run_compile("reduce-overhead", torch.float16, "reduce-overhead fp16")
    except Exception as e:
        print(f"  V3 failed: {e}")

    print("\n── V4: compile reduce-overhead bf16 (inference only) ──")
    try:
        results["variants"]["V4_bf16"] = run_compile("reduce-overhead", torch.bfloat16, "reduce-overhead bf16")
    except Exception as e:
        print(f"  V4 failed: {e}")

    print("\n── V5: CUDA Graph capture (fp32) ──")
    try:
        results["variants"]["V5_cuda_graph"] = run_cuda_graph()
    except Exception as e:
        print(f"  V5 failed: {e}")

    # Summary
    print("\n\n========== SUMMARY ==========")
    print(f"{'Variant':<22} {'median_ms':>10} {'p95_ms':>8} {'sps':>9} {'mem_MiB':>9}")
    base_ms = results["variants"]["V0_eager"]["median_ms"]
    for k, r in results["variants"].items():
        if not r: continue
        print(f"{k:<22} {r['median_ms']:>10.3f} {r['p95_ms']:>8.3f} "
              f"{r['throughput_sps']:>9.0f} {r.get('peak_mib', -1):>9.1f}")
    print(f"\nSpeedups vs V0_eager ({base_ms:.3f}ms):")
    for k, r in results["variants"].items():
        if not r: continue
        print(f"  {k:<22} {base_ms / r['median_ms']:>6.2f}×")

    # Best variant
    best = min((k for k in results["variants"] if results["variants"][k]),
               key=lambda k: results["variants"][k]["median_ms"])
    print(f"\nBest variant: {best} @ {results['variants'][best]['median_ms']:.3f}ms")
    # VGG_FC reference: 1.570ms (compiled, step810)
    print(f"vs VGG_FC (1.570ms from step810): {1.570 / results['variants'][best]['median_ms']:.2f}× faster")
    # vs Linear reference: 0.089ms (step810)
    print(f"vs Linear (0.089ms from step810): {results['variants'][best]['median_ms'] / 0.089:.2f}× slower")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
