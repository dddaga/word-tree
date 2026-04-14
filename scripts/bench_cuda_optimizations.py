"""CUDA optimization benchmark: 4 SGNNET variants on RTX 5060 Ti.

Tests whether torch.compile / CUDA graphs / model restructuring can close the
gap between 116x fewer FLOPs and the measured 1.21x training speedup (2.6% GPU util).

Variants (N=2048, D=16, K_hh=2, K_iter=5, AH=1.0 — efficiency config):
  V0_eager        : Standard SGNNET_AntiHebbian(SGNNET_Resonant(...)) — baseline
  V1_compile      : Same model wrapped with torch.compile(mode="reduce-overhead")
  V2_cuda_graph   : Manual CUDA graph capture of the forward pass
  V3_cuda_optimized: SGNNET_AntiHebbian_CUDA from model_resonant_cuda.py

For each variant:
  - Inference latency at bs=1, 32, 128, 512 (50 warmup + 200 timed iters)
  - Training step time at bs=128 (50 warmup + 200 timed iters)
  - GPU memory reserved
  - GPU utilization via nvidia-smi (polled every 100ms during training)

Output: results/bench_cuda_optimizations_5060ti.json

Usage:
  python scripts/bench_cuda_optimizations.py
  python scripts/bench_cuda_optimizations.py --device cuda:1
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

# Repo root on PATH
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_resonant_cuda import SGNNET_AntiHebbian_CUDA, SGNNET_Resonant_CUDA


# ── Config ────────────────────────────────────────────────────────────────────

N_IN         = 25088
N_CLASSES    = 10
N            = 2048
D            = 16
K_HH         = 2
K_ITER       = 5
ALPHA_AHEBB  = 1.0
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0

WARMUP_ITERS = 50
BENCH_ITERS  = 200
TRAIN_BS     = 128
BATCH_SIZES  = [1, 32, 128, 512]


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_base(device: torch.device, seed: int = 42) -> SGNNET_AntiHebbian:
    """Standard V0 model (SGNNET_AntiHebbian wrapping SGNNET_Resonant)."""
    torch.manual_seed(seed)
    n_groups = max(8, N // 8)
    K_local  = max(1, K_HH - max(1, K_HH // 4))
    K_random = K_HH - K_local

    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_CLASSES, D=D, N_in=N_IN,
        K_local=K_local, K_random=K_random,
        n_groups=n_groups, K_iter=K_ITER,
        norm_mode="l2", encoding_mode="fourier",
    ).to(device)

    res = SGNNET_Resonant(
        base=sw,
        alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING,
        mode="dynamic_z_geo",
    ).to(device)

    model = SGNNET_AntiHebbian(
        base=res,
        alpha_ahebb=ALPHA_AHEBB,
        variant="wpos",
    ).to(device)

    return model


def _make_v3_cuda(device: torch.device, seed: int = 42) -> SGNNET_AntiHebbian_CUDA:
    """V3: SGNNET_AntiHebbian_CUDA (compiled AH + routing step)."""
    torch.manual_seed(seed)
    n_groups = max(8, N // 8)
    K_local  = max(1, K_HH - max(1, K_HH // 4))
    K_random = K_HH - K_local

    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_CLASSES, D=D, N_in=N_IN,
        K_local=K_local, K_random=K_random,
        n_groups=n_groups, K_iter=K_ITER,
        norm_mode="l2", encoding_mode="fourier",
    ).to(device)

    res = SGNNET_Resonant_CUDA(
        base=sw,
        alpha_reflect=ALPHA_REFLECT,
        alpha_turing=0.0,
        mode="dynamic_z_geo",
        compile=True,
    ).to(device)

    model = SGNNET_AntiHebbian_CUDA(
        base=res,
        alpha_ahebb=ALPHA_AHEBB,
        variant="wpos",
        compile=True,
    ).to(device)

    return model


# ── Numerical equivalence check ───────────────────────────────────────────────

def verify_v0_output(device: torch.device) -> float:
    """Assert V0_eager forward matches original model (within 1e-4).

    We compare two models with the same seed — they should produce identical
    logits since V0 is just the standard stacking pattern.
    """
    m1 = _make_base(device, seed=7)
    m2 = _make_base(device, seed=7)
    m1.eval(); m2.eval()
    x = torch.randn(4, N_IN, device=device)
    with torch.no_grad():
        out1 = m1(x)
        out2 = m2(x)
    max_diff = (out1 - out2).abs().max().item()
    assert max_diff < 1e-4, f"V0 equivalence failed: max_diff={max_diff}"
    print(f"  [check] V0 equivalence: max_diff={max_diff:.2e}  PASSED")
    return max_diff


def verify_v3_vs_v0(device: torch.device) -> float:
    """Check that V3 (uncompiled eager) produces logits close to V0.

    Note: V3 uses the AH wpos branch which is numerically equivalent to
    the SGNNET_AntiHebbian wpos branch, so outputs should match to fp32
    precision when starting from the same weights.
    """
    torch.manual_seed(42)
    n_groups = max(8, N // 8)
    K_local  = max(1, K_HH - max(1, K_HH // 4))
    K_random = K_HH - K_local

    # Shared SmallWorld backbone
    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_CLASSES, D=D, N_in=N_IN,
        K_local=K_local, K_random=K_random,
        n_groups=n_groups, K_iter=K_ITER,
        norm_mode="l2", encoding_mode="fourier",
    ).to(device)

    res_orig = SGNNET_Resonant(
        base=sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0, mode="dynamic_z_geo",
    ).to(device)
    v0_model = SGNNET_AntiHebbian(
        base=res_orig, alpha_ahebb=ALPHA_AHEBB, variant="wpos"
    ).to(device)
    v0_model.eval()

    # V3 uncompiled (same backbone)
    res_cuda = SGNNET_Resonant_CUDA(
        base=sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        mode="dynamic_z_geo", compile=False,
    ).to(device)
    # Copy theta from original resonant
    with torch.no_grad():
        res_cuda.theta.copy_(res_orig.theta)

    v3_model = SGNNET_AntiHebbian_CUDA(
        base=res_cuda, alpha_ahebb=ALPHA_AHEBB, variant="wpos", compile=False
    ).to(device)
    v3_model.eval()

    x = torch.randn(4, N_IN, device=device)
    with torch.no_grad():
        out_v0 = v0_model(x)
        out_v3 = v3_model(x)

    max_diff = (out_v0 - out_v3).abs().max().item()
    print(f"  [check] V0 vs V3 (eager): max_diff={max_diff:.2e}  "
          f"{'PASSED' if max_diff < 1e-3 else 'WARNING: diff > 1e-3'}")
    return max_diff


# ── nvidia-smi utilities ──────────────────────────────────────────────────────

def _smi_snapshot() -> dict | None:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            if len(parts) >= 5:
                return {
                    "gpu_util_pct": int(parts[0]),
                    "mem_util_pct": int(parts[1]),
                    "mem_used_mb":  int(parts[2]),
                    "mem_total_mb": int(parts[3]),
                    "power_w":      float(parts[4]),
                }
    except Exception:
        pass
    return None


def _smi_monitor(fn, interval: float = 0.1):
    """Run fn() while polling nvidia-smi every interval seconds."""
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            s = _smi_snapshot()
            if s:
                samples.append(s)
            time.sleep(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    result = fn()
    stop.set()
    t.join(timeout=2)
    return result, samples


def _smi_summary(samples: list) -> dict:
    if not samples:
        return {}
    return {
        "gpu_util_avg":    round(float(np.mean([s["gpu_util_pct"] for s in samples])), 1),
        "gpu_util_max":    max(s["gpu_util_pct"] for s in samples),
        "mem_used_avg_mb": round(float(np.mean([s["mem_used_mb"] for s in samples])), 0),
        "power_avg_w":     round(float(np.mean([s["power_w"] for s in samples])), 1),
        "n_samples":       len(samples),
    }


# ── Benchmark utilities ───────────────────────────────────────────────────────

def sync_cuda():
    torch.cuda.synchronize()


def gpu_mem_reserved_mb() -> float:
    return round(torch.cuda.memory_reserved() / 1e6, 1)


def bench_inference(
    model: nn.Module,
    device: torch.device,
    batch_sizes: list[int] = BATCH_SIZES,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
) -> dict:
    model.eval()
    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, N_IN, device=device)

        with torch.no_grad():
            for _ in range(warmup):
                model(x)
        sync_cuda()

        times = []
        with torch.no_grad():
            for _ in range(iters):
                sync_cuda()
                t0 = time.perf_counter()
                model(x)
                sync_cuda()
                times.append((time.perf_counter() - t0) * 1000)

        times = sorted(times)
        trim = max(1, len(times) // 10)
        trimmed = times[trim:-trim]
        lat = float(np.median(trimmed))

        results[bs] = {
            "latency_ms":     round(lat, 3),
            "throughput_sps": round(bs / (lat / 1000), 1),
            "p5_ms":          round(float(np.percentile(trimmed, 5)), 3),
            "p95_ms":         round(float(np.percentile(trimmed, 95)), 3),
            "gpu_mem_reserved_mb": gpu_mem_reserved_mb(),
        }

    return results


def bench_training(
    model: nn.Module,
    device: torch.device,
    batch_size: int = TRAIN_BS,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
) -> tuple[dict, list]:
    """Returns (timing_dict, smi_samples)."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, N_IN, device=device)
    y = torch.randint(0, N_CLASSES, (batch_size,), device=device)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    sync_cuda()

    def _run_timed():
        times = []
        for _ in range(iters):
            sync_cuda()
            t0 = time.perf_counter()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            sync_cuda()
            times.append((time.perf_counter() - t0) * 1000)
        return times

    times, smi_samples = _smi_monitor(_run_timed, interval=0.1)

    times = sorted(times)
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    step_ms = float(np.median(trimmed))

    result = {
        "batch_size":         batch_size,
        "step_ms":            round(step_ms, 3),
        "throughput_sps":     round(batch_size / (step_ms / 1000), 1),
        "p5_ms":              round(float(np.percentile(trimmed, 5)), 3),
        "p95_ms":             round(float(np.percentile(trimmed, 95)), 3),
        "gpu_mem_reserved_mb": gpu_mem_reserved_mb(),
    }

    return result, smi_samples


# ── CUDA Graph capture (V2) ───────────────────────────────────────────────────

class CUDAGraphWrapper(nn.Module):
    """Wraps a model's forward in a CUDA graph for replay-based inference.

    Captures a static graph at bs=32 (the benchmark default for speedup).
    Other batch sizes fall back to eager.
    """

    def __init__(self, model: nn.Module, static_bs: int, device: torch.device):
        super().__init__()
        self.model     = model
        self.static_bs = static_bs
        self.device    = device
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_x:   torch.Tensor | None = None
        self._static_out: torch.Tensor | None = None
        self._capture(static_bs)

    def _capture(self, bs: int):
        """Capture the CUDA graph."""
        self._static_x   = torch.randn(bs, N_IN, device=self.device)
        self._static_out = torch.zeros(bs, N_CLASSES, device=self.device)

        # Warmup outside graph
        self.model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(self._static_x)
        sync_cuda()

        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self._graph):
            out = self.model(self._static_x)
            self._static_out.copy_(out)
        sync_cuda()
        print(f"    [cuda_graph] captured at bs={bs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == self.static_bs and self._graph is not None:
            self._static_x.copy_(x)
            self._graph.replay()
            return self._static_out.clone()
        # Fallback: eager
        with torch.no_grad():
            return self.model(x)


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(device: torch.device, args) -> dict:
    results = {}

    # ── Numerical checks ──────────────────────────────────────────────────────
    print("\n── Numerical equivalence checks ──")
    verify_v0_output(device)
    v0_vs_v3_diff = verify_v3_vs_v0(device)
    results["numerical"] = {"v0_vs_v3_max_diff": v0_vs_v3_diff}

    # ── V0: eager baseline ────────────────────────────────────────────────────
    print("\n── V0_eager  (baseline) ──")
    torch.cuda.empty_cache()
    v0 = _make_base(device)

    print(f"  Inference:")
    v0_inf = bench_inference(v0, device, BATCH_SIZES, args.warmup, args.iters)
    for bs, r in v0_inf.items():
        print(f"    bs={bs:4d}  lat={r['latency_ms']:8.3f}ms  "
              f"tput={r['throughput_sps']:10.1f} sps  mem={r['gpu_mem_reserved_mb']}MB")

    print(f"  Training (bs={TRAIN_BS}):")
    v0_train, v0_smi = bench_training(v0, device, TRAIN_BS, args.warmup, args.train_iters)
    v0_smi_sum = _smi_summary(v0_smi)
    print(f"    step={v0_train['step_ms']:.3f}ms  "
          f"tput={v0_train['throughput_sps']:.1f} sps  "
          f"GPU util avg={v0_smi_sum.get('gpu_util_avg', 'N/A')}%")

    results["V0_eager"] = {
        "inference": {str(k): v for k, v in v0_inf.items()},
        "training":  v0_train,
        "nvidia_smi": v0_smi_sum,
    }
    del v0; torch.cuda.empty_cache()

    # ── V1: module-level torch.compile ───────────────────────────────────────
    print("\n── V1_compile  (torch.compile module-level) ──")
    torch.cuda.empty_cache()
    v1_base = _make_base(device)
    v1 = torch.compile(v1_base, mode="reduce-overhead")

    # Trigger compile with a warmup forward pass
    print("  Triggering compile...")
    _dummy = torch.randn(TRAIN_BS, N_IN, device=device)
    with torch.no_grad():
        v1(_dummy)
    sync_cuda()
    del _dummy

    print(f"  Inference:")
    v1_inf = bench_inference(v1, device, BATCH_SIZES, args.warmup, args.iters)
    for bs, r in v1_inf.items():
        print(f"    bs={bs:4d}  lat={r['latency_ms']:8.3f}ms  "
              f"tput={r['throughput_sps']:10.1f} sps  mem={r['gpu_mem_reserved_mb']}MB")

    print(f"  Training (bs={TRAIN_BS}):")
    v1_train, v1_smi = bench_training(v1, device, TRAIN_BS, args.warmup, args.train_iters)
    v1_smi_sum = _smi_summary(v1_smi)
    print(f"    step={v1_train['step_ms']:.3f}ms  "
          f"tput={v1_train['throughput_sps']:.1f} sps  "
          f"GPU util avg={v1_smi_sum.get('gpu_util_avg', 'N/A')}%")

    results["V1_compile"] = {
        "inference": {str(k): v for k, v in v1_inf.items()},
        "training":  v1_train,
        "nvidia_smi": v1_smi_sum,
    }
    del v1, v1_base; torch.cuda.empty_cache()

    # ── V2: manual CUDA graph (inference only — training graphs are tricky) ──
    print("\n── V2_cuda_graph  (manual CUDA graph, inference @ static bs=32) ──")
    torch.cuda.empty_cache()
    v2_inner = _make_base(device)
    v2 = CUDAGraphWrapper(v2_inner, static_bs=32, device=device)

    # For non-captured batch sizes, it falls back to eager.
    print(f"  Inference (bs=32 uses CUDA graph, others eager fallback):")
    v2_inf = bench_inference(v2, device, BATCH_SIZES, args.warmup, args.iters)
    for bs, r in v2_inf.items():
        tag = "[graph]" if bs == 32 else "[eager]"
        print(f"    bs={bs:4d} {tag}  lat={r['latency_ms']:8.3f}ms  "
              f"tput={r['throughput_sps']:10.1f} sps  mem={r['gpu_mem_reserved_mb']}MB")

    # Training: fall back to eager (CUDA graph training is tricky with dynamic shapes)
    print(f"  Training (bs={TRAIN_BS}, eager — graph capture only for inference):")
    v2_train, v2_smi = bench_training(v2_inner, device, TRAIN_BS, args.warmup, args.train_iters)
    v2_smi_sum = _smi_summary(v2_smi)
    print(f"    step={v2_train['step_ms']:.3f}ms  "
          f"tput={v2_train['throughput_sps']:.1f} sps  "
          f"GPU util avg={v2_smi_sum.get('gpu_util_avg', 'N/A')}%")

    results["V2_cuda_graph"] = {
        "inference": {str(k): v for k, v in v2_inf.items()},
        "training":  v2_train,
        "nvidia_smi": v2_smi_sum,
        "note": "CUDA graph used at bs=32 inference; training uses eager (same as V0)",
    }
    del v2, v2_inner; torch.cuda.empty_cache()

    # ── V3: SGNNET_AntiHebbian_CUDA (compiled routing step) ──────────────────
    print("\n── V3_cuda_optimized  (SGNNET_AntiHebbian_CUDA) ──")
    torch.cuda.empty_cache()
    v3 = _make_v3_cuda(device)

    # Trigger compile
    print("  Triggering compile...")
    _dummy = torch.randn(TRAIN_BS, N_IN, device=device)
    with torch.no_grad():
        v3(_dummy)
    sync_cuda()
    del _dummy

    print(f"  Inference:")
    v3_inf = bench_inference(v3, device, BATCH_SIZES, args.warmup, args.iters)
    for bs, r in v3_inf.items():
        print(f"    bs={bs:4d}  lat={r['latency_ms']:8.3f}ms  "
              f"tput={r['throughput_sps']:10.1f} sps  mem={r['gpu_mem_reserved_mb']}MB")

    print(f"  Training (bs={TRAIN_BS}):")
    v3_train, v3_smi = bench_training(v3, device, TRAIN_BS, args.warmup, args.train_iters)
    v3_smi_sum = _smi_summary(v3_smi)
    print(f"    step={v3_train['step_ms']:.3f}ms  "
          f"tput={v3_train['throughput_sps']:.1f} sps  "
          f"GPU util avg={v3_smi_sum.get('gpu_util_avg', 'N/A')}%")

    results["V3_cuda_optimized"] = {
        "inference": {str(k): v for k, v in v3_inf.items()},
        "training":  v3_train,
        "nvidia_smi": v3_smi_sum,
    }
    del v3; torch.cuda.empty_cache()

    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: dict):
    v0_train_ms  = results["V0_eager"]["training"]["step_ms"]
    v0_inf_bs32  = results["V0_eager"]["inference"]["32"]["latency_ms"]
    v0_gpu_util  = results["V0_eager"]["nvidia_smi"].get("gpu_util_avg", "N/A")

    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    header = f"  {'Variant':<22} {'train_ms/step':>14} {'inf_ms bs=32':>14} {'GPU util%':>10} {'speedup vs V0':>14}"
    print(header)
    print(f"  {'─'*74}")

    variants = ["V0_eager", "V1_compile", "V2_cuda_graph", "V3_cuda_optimized"]
    for name in variants:
        if name not in results:
            continue
        r        = results[name]
        t_ms     = r["training"]["step_ms"]
        i_ms     = r["inference"]["32"]["latency_ms"]
        gpu_util = r["nvidia_smi"].get("gpu_util_avg", "N/A")
        speedup  = round(v0_train_ms / t_ms, 2) if t_ms > 0 else "—"
        print(f"  {name:<22} {t_ms:>14.3f} {i_ms:>14.3f} {str(gpu_util):>10} {str(speedup):>14}")

    print(f"\n  Baseline V0: train={v0_train_ms:.3f}ms/step  "
          f"inf(bs=32)={v0_inf_bs32:.3f}ms  GPU_util={v0_gpu_util}%")
    print(f"  (GPU util% is from nvidia-smi polling every 100ms during training)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CUDA optimization benchmark for SGNNET")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--warmup",       type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",        type=int, default=BENCH_ITERS)
    parser.add_argument("--train-iters",  type=int, default=BENCH_ITERS, dest="train_iters")
    parser.add_argument("--output",       default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False. This benchmark is CUDA-only.")
        sys.exit(1)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print(f"\n{'='*78}")
    print("SGNNET CUDA OPTIMIZATION BENCHMARK")
    print(f"{'='*78}")
    print(f"Device  : {device} ({torch.cuda.get_device_name(device)})")
    print(f"Host    : {platform.node()}")
    print(f"PyTorch : {torch.__version__}")
    print(f"Config  : N={N} D={D} K_hh={K_HH} K_iter={K_ITER} "
          f"AH={ALPHA_AHEBB} reflect={ALPHA_REFLECT} turing={ALPHA_TURING}")
    print(f"Iters   : warmup={args.warmup}  bench={args.iters}  train={args.train_iters}")

    sys_info = {
        "hostname":    platform.node(),
        "platform":    platform.platform(),
        "gpu_name":    torch.cuda.get_device_name(device),
        "gpu_mem_total_mb": torch.cuda.get_device_properties(device).total_memory / 1e6,
        "torch_version": torch.__version__,
        "device":      str(device),
        "config": {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "alpha_ahebb": ALPHA_AHEBB, "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing": ALPHA_TURING,
        },
    }

    bench_results = run_benchmark(device, args)
    print_summary(bench_results)

    all_results = {"system": sys_info, "variants": bench_results}

    out_path = args.output or str(
        _repo_root / "results" / "bench_cuda_optimizations_5060ti.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
