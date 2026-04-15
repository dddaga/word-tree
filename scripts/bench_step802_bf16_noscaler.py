"""Step 802: bf16 without GradScaler — isolate whether bf16 slowdown was scaler overhead.

MOTIVATION
==========
Step 801 showed bf16+reduce_overhead is 4.4× SLOWER than fp32 V1 (23ms vs 5.3ms).
The hypothesis: GradScaler overhead caused it. GradScaler is designed for fp16
overflow prevention — bf16 has more exponent bits and doesn't need scaling.
The scaler adds per-step overhead: scale() wraps backward, unscale_(), inf checks,
update(). At small model size (67K params, bs=128), this overhead dominates.

THIS EXPERIMENT: drop GradScaler entirely for bf16 training. Use direct
loss.backward() with bf16 autocast (same as CUDA fp16 training without scaling).

HYPOTHESIS: bf16 WITHOUT scaler should be ~same or faster than V1 (fp32).
If bf16 inference is faster at bs=32+ (confirmed 0.311 vs 0.316ms in step801),
then bf16 training without overhead should also be faster.

CONFIGS: same as step801. V1 (fp32 reduce-overhead) vs V7 (bf16 reduce-overhead
no scaler) vs V8 (bf16 max-autotune no scaler).
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

N_WARMUP      = 3
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
    """CUDA Event timing — median of n_timed runs. NO GradScaler for bf16."""
    if train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        y = torch.randint(0, N_CLASSES, (x.shape[0],), device=device)
    else:
        model.eval()

    # KEY DIFFERENCE FROM step801: NO GradScaler at all.
    # bf16 doesn't need gradient scaling — only fp16 can overflow.
    def _step(x_in):
        if train:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_bf16):
                out  = model(x_in)
                loss = criterion(out, y)
            # Direct backward — no scaler overhead
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
    print(f"\n── {label} ──")
    torch.cuda.empty_cache()
    model = model_fn(device)

    if compile_mode is not None:
        print(f"  Compiling (mode={compile_mode!r})... ", end="", flush=True)
        t0 = time.perf_counter()
        model = torch.compile(model, mode=compile_mode)
        _x = torch.randn(TRAIN_BS, N_IN, device=device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_bf16):
                model(_x)
        sync()
        print(f"done ({time.perf_counter()-t0:.1f}s)")
        del _x

    inf_results = {}
    for bs in BENCH_SIZES:
        x = torch.randn(bs, N_IN, device=device)
        r = cuda_event_latency(model, x, device, train=False, use_bf16=use_bf16)
        inf_results[bs] = r
        print(f"  Inf bs={bs:<4}  med={r['median_ms']:.3f}ms  "
              f"p5={r['p5_ms']:.3f}  p95={r['p95_ms']:.3f}  "
              f"tput={r['throughput_sps']:.0f} sps")

    x_train = torch.randn(TRAIN_BS, N_IN, device=device)
    print(f"  Train bs={TRAIN_BS}  ", end="", flush=True)

    def _train_bench():
        return cuda_event_latency(model, x_train, device, train=True, use_bf16=use_bf16)

    train_r, gpu_util, gpu_pow = poll_gpu_during(_train_bench, interval=0.2)
    print(f"med={train_r['median_ms']:.3f}ms  "
          f"tput={train_r['throughput_sps']:.0f} sps  "
          f"GPU util={gpu_util}%  power={gpu_pow}W")

    result = {
        "compile_mode":    compile_mode,
        "bf16":            use_bf16,
        "grad_scaler":     False,  # key: no scaler
        "inference":       {str(k): v for k, v in inf_results.items()},
        "training":        train_r,
        "gpu_util_avg":    gpu_util,
        "gpu_power_avg_w": gpu_pow,
    }
    del model
    torch.cuda.empty_cache()
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Step 802: bf16 without GradScaler — isolate scaler overhead"
    )
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--warmup",  type=int, default=N_WARMUP)
    parser.add_argument("--timed",   type=int, default=N_TIMED)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = torch.device(args.device if ":" in args.device else f"{args.device}:0")
    torch.cuda.set_device(device)

    print(f"\n{'='*70}")
    print("STEP 802 — bf16 WITHOUT GradScaler (scaler overhead isolation)")
    print(f"{'='*70}")
    print(f"Device  : {device} ({torch.cuda.get_device_name(device)})")
    print(f"Config  : N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"NOTE    : No GradScaler. bf16 doesn't need fp16 overflow scaling.")
    print(f"          Step 801 showed V5 (bf16+scaler) was 4.4× SLOWER than V1.")
    print(f"          This isolates: is the slowdown scaler or bf16 itself?")

    results = {}

    # V1: reduce-overhead fp32 (baseline from step801)
    results["V1_reduce_overhead_fp32"] = bench_variant(
        "V1_reduce_overhead fp32 (step801 baseline)",
        make_base, device, compile_mode="reduce-overhead", use_bf16=False,
    )

    # V7: bf16 + reduce-overhead, NO scaler
    results["V7_bf16_noscaler"] = bench_variant(
        "V7_bf16 + reduce-overhead, NO GradScaler",
        make_base, device, compile_mode="reduce-overhead", use_bf16=True,
    )

    # V8: bf16 + max-autotune, NO scaler
    results["V8_bf16_autotune_noscaler"] = bench_variant(
        "V8_bf16 + max-autotune, NO GradScaler",
        make_base, device, compile_mode="max-autotune", use_bf16=True,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — training speedup vs V1 (step801 comparison column)")
    print(f"{'='*70}")
    # V1 from step801 for reference
    print(f"  {'V1 step801 (ref)':35}  train=5.304ms  24133sps  speedup=1.00x")
    ref_ms = results["V1_reduce_overhead_fp32"]["training"]["median_ms"]
    for name, r in results.items():
        t_ms    = r["training"]["median_ms"]
        sps     = r["training"]["throughput_sps"]
        speedup = round(ref_ms / t_ms, 2)
        print(f"  {name:<35}  train={t_ms:.3f}ms  {sps:.0f}sps  speedup vs V1={speedup}x")

    out_data = {
        "system": {
            "hostname":      platform.node(),
            "gpu":           torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        },
        "config": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER},
        "hypothesis": "bf16 slowdown in step801 was GradScaler overhead, not bf16 itself",
        "variants": results,
    }
    out_path = args.output or str(ROOT / "results" / "bench_step802_bf16_noscaler_5060ti.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
