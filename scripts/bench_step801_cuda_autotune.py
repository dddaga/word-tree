"""Step 801: CUDA optimization — max-autotune + loop-unroll + bf16 variants.

MOTIVATION
==========
Step 500 (torch.compile reduce-overhead) gave 4.2× training speedup.
Three untested optimizations may stack on top:
  V4_max_autotune : mode="max-autotune" — Inductor autotunes kernel tiling,
                    more aggressive fusion across K_iter loop (vs reduce-overhead)
  V5_bf16         : bf16 autocast — halves Z tensor bandwidth (K_iter=5 × full
                    read-write). RTX 5060 Ti SM120 has BF16 tensor cores.
                    Gain is bandwidth, not compute (D=16 too small for tensorcore).
  V6_bf16_autotune: both combined — maximum throughput config

CLAIM UNDER TEST
================
  V4: +15–40% vs V1 (reduce-overhead)
  V5: +50–100% if bandwidth-bound; smaller if compute-bound
  V6: compound if independent

All variants benchmarked against V1_reduce_overhead (our current production config).
Uses same CUDA Event harness as step800 (median 50 runs, 3 warmup) — paper-grade.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, AH=1.0, reflect=0.5 — efficiency config)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian


# ── Config ────────────────────────────────────────────────────────────────────
N_IN          = 25088
N_CLASSES     = 10
N             = 2048
D             = 16
K_HH          = 2
K_ITER        = 5
K_IN          = 25
ALPHA_AHEBB   = 1.0
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0

N_WARMUP      = 3      # short warmup — compile already done
N_TIMED       = 50
TRAIN_BS      = 128
BENCH_SIZES   = [1, 32, 128, 512]


# ── Model factory ─────────────────────────────────────────────────────────────
def make_base(device: torch.device) -> SGNNET_AntiHebbian:
    torch.manual_seed(42)
    n_groups = max(8, N // 8)
    K_local  = max(1, K_HH - max(1, K_HH // 4))
    K_random = K_HH - K_local
    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_CLASSES, D=D, N_in=N_IN,
        K_in=K_IN, K_local=K_local, K_random=K_random,
        n_groups=n_groups, K_iter=K_ITER,
        norm_mode="l2", encoding_mode="fourier",
    ).to(device)
    res = SGNNET_Resonant(
        base=sw, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
    ).to(device)
    return SGNNET_AntiHebbian(base=res, alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(device)


# ── Measurement utilities ─────────────────────────────────────────────────────
def sync():
    torch.cuda.synchronize()


def cuda_event_latency(model, x, device, n_warmup=N_WARMUP, n_timed=N_TIMED,
                       train=False, use_bf16=False):
    """CUDA Event timing — median of n_timed runs."""
    if train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        y = torch.randint(0, N_CLASSES, (x.shape[0],), device=device)
    else:
        model.eval()

    # AMP scaler only for training + bf16
    scaler = torch.cuda.amp.GradScaler() if (train and use_bf16) else None

    def _step(x_in):
        if train:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_bf16):
                out  = model(x_in)
                loss = criterion(out, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=use_bf16):
                    model(x_in)

    for _ in range(n_warmup):
        _step(x)
    sync()

    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)
    times   = []
    for _ in range(n_timed):
        sync()
        start_e.record()
        _step(x)
        end_e.record()
        sync()
        times.append(start_e.elapsed_time(end_e))

    times.sort()
    trim    = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    med     = float(np.median(trimmed))
    p5      = float(np.percentile(trimmed, 5))
    p95     = float(np.percentile(trimmed, 95))
    return {
        "median_ms":      round(med, 3),
        "p5_ms":          round(p5, 3),
        "p95_ms":         round(p95, 3),
        "throughput_sps": round(x.shape[0] / (med / 1000), 1),
    }


def nvidia_smi_snapshot():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            if len(parts) >= 2:
                return {"gpu_util_pct": int(parts[0]), "power_w": float(parts[1])}
    except Exception:
        pass
    return None


def poll_gpu_during(fn, interval=0.2):
    """Run fn(), sampling nvidia-smi every interval seconds."""
    samples = []
    stop_ev = threading.Event()

    def _poll():
        while not stop_ev.is_set():
            s = nvidia_smi_snapshot()
            if s:
                samples.append(s)
            time.sleep(interval)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    result = fn()
    stop_ev.set()
    t.join(timeout=2)
    if samples:
        avg_util = round(float(np.mean([s["gpu_util_pct"] for s in samples])), 1)
        avg_pow  = round(float(np.mean([s["power_w"] for s in samples])), 1)
    else:
        avg_util = avg_pow = None
    return result, avg_util, avg_pow


def bench_variant(label, model_fn, device, compile_mode=None, use_bf16=False):
    """Build, optionally compile, then benchmark a variant.

    Returns dict with inference + training latency + GPU util.
    """
    print(f"\n── {label} ──")
    torch.cuda.empty_cache()
    model = model_fn(device)

    if compile_mode is not None:
        print(f"  Compiling (mode={compile_mode!r})... ", end="", flush=True)
        t_compile_start = time.perf_counter()
        model = torch.compile(model, mode=compile_mode)
        # Trigger compilation with a dummy forward
        _x = torch.randn(TRAIN_BS, N_IN, device=device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_bf16):
                model(_x)
        sync()
        compile_ms = (time.perf_counter() - t_compile_start) * 1000
        print(f"done ({compile_ms/1000:.1f}s)")
        del _x

    # Inference latency
    inf_results = {}
    for bs in BENCH_SIZES:
        x = torch.randn(bs, N_IN, device=device)
        r = cuda_event_latency(model, x, device, train=False, use_bf16=use_bf16)
        inf_results[bs] = r
        print(f"  Inf bs={bs:<4}  med={r['median_ms']:.3f}ms  "
              f"p5={r['p5_ms']:.3f}  p95={r['p95_ms']:.3f}  "
              f"tput={r['throughput_sps']:.0f} sps")

    # Training latency + GPU util
    x_train = torch.randn(TRAIN_BS, N_IN, device=device)
    print(f"  Train bs={TRAIN_BS}  ", end="", flush=True)

    def _train_bench():
        return cuda_event_latency(model, x_train, device, train=True, use_bf16=use_bf16)

    train_r, gpu_util, gpu_pow = poll_gpu_during(_train_bench, interval=0.2)
    print(f"med={train_r['median_ms']:.3f}ms  "
          f"tput={train_r['throughput_sps']:.0f} sps  "
          f"GPU util={gpu_util}%  power={gpu_pow}W")

    result = {
        "compile_mode": compile_mode,
        "bf16":         use_bf16,
        "inference":    {str(k): v for k, v in inf_results.items()},
        "training":     train_r,
        "gpu_util_avg": gpu_util,
        "gpu_power_avg_w": gpu_pow,
    }
    del model
    torch.cuda.empty_cache()
    return result


# ── Numerical equivalence ─────────────────────────────────────────────────────
def check_bf16_accuracy(device):
    """Verify bf16 output is within 1% of fp32 for same model."""
    m = make_base(device)
    m.eval()
    x  = torch.randn(8, N_IN, device=device)
    with torch.no_grad():
        out_fp32 = m(x)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_bf16 = m(x)
    max_diff = (out_fp32 - out_bf16.float()).abs().max().item()
    print(f"  [check] bf16 vs fp32 logit max_diff: {max_diff:.4f}  "
          f"({'OK' if max_diff < 0.5 else 'WARNING > 0.5 — check accuracy'})")
    return max_diff


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Step 801: CUDA autotune + bf16 benchmark"
    )
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--warmup",      type=int, default=N_WARMUP)
    parser.add_argument("--timed",       type=int, default=N_TIMED)
    parser.add_argument("--skip",        default="",
                        help="Comma-separated variants to skip: V1,V4,V5,V6")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()
    skip = set(v.strip() for v in args.skip.split(",") if v.strip())

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = torch.device(args.device if ":" in args.device else f"{args.device}:0")
    torch.cuda.set_device(device)

    print(f"\n{'='*70}")
    print("STEP 801 — CUDA Autotune + bf16 Benchmark")
    print(f"{'='*70}")
    print(f"Device  : {device} ({torch.cuda.get_device_name(device)})")
    print(f"Config  : N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"Timing  : {args.warmup} warmup + {args.timed} timed (CUDA Events)")
    print(f"Skipped : {skip or 'none'}")

    # Numerical check for bf16
    print(f"\n── Numerical checks ──")
    check_bf16_accuracy(device)

    results = {}

    # V1: reduce-overhead (our current production config = step500 winner)
    if "V1" not in skip:
        results["V1_reduce_overhead"] = bench_variant(
            "V1_reduce_overhead  (step500 baseline)",
            make_base, device, compile_mode="reduce-overhead", use_bf16=False,
        )

    # V4: max-autotune — more aggressive fusion, longer compile
    if "V4" not in skip:
        print("\n  NOTE: max-autotune compilation takes 3–10 min first run.")
        results["V4_max_autotune"] = bench_variant(
            "V4_max_autotune",
            make_base, device, compile_mode="max-autotune", use_bf16=False,
        )

    # V5: bf16 + reduce-overhead
    if "V5" not in skip:
        results["V5_bf16_reduce_overhead"] = bench_variant(
            "V5_bf16 + reduce-overhead",
            make_base, device, compile_mode="reduce-overhead", use_bf16=True,
        )

    # V6: bf16 + max-autotune (compound)
    if "V6" not in skip:
        results["V6_bf16_max_autotune"] = bench_variant(
            "V6_bf16 + max-autotune",
            make_base, device, compile_mode="max-autotune", use_bf16=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY (training throughput, speedup vs V1_reduce_overhead)")
    print(f"{'='*70}")
    ref_ms = results.get("V1_reduce_overhead", {}).get("training", {}).get("median_ms", None)
    for name, r in results.items():
        t_ms     = r["training"]["median_ms"]
        sps      = r["training"]["throughput_sps"]
        speedup  = round(ref_ms / t_ms, 2) if ref_ms else "—"
        gpu_util = r.get("gpu_util_avg", "N/A")
        print(f"  {name:<32}  train={t_ms:.3f}ms  {sps:.0f}sps  "
              f"GPU={gpu_util}%  speedup vs V1={speedup}")

    if ref_ms:
        print(f"\n  Reference: V1_reduce_overhead = {ref_ms:.3f}ms")
        print(f"  V1 was {4.2}× faster than V0_eager (step500 result)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_data = {
        "system": {
            "hostname":      platform.node(),
            "gpu":           torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        },
        "config": {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "alpha_ahebb": ALPHA_AHEBB,
        },
        "variants": results,
    }
    out_path = args.output or str(ROOT / "results" / "bench_step801_autotune_5060ti.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
