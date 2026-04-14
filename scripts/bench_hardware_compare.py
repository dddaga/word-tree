"""Hardware benchmark: VGG16 FC vs SGNNET across Mac Studio / Mac Mini / 5060ti.

Measures training speed and inference speed for:
  1. VGG16 FC layers (25088 → 4096 → 4096 → 1000) — dense matmul baseline
  2. SGNNET efficiency config (N=2048, D=16, K_hh=2, K_iter=5) — sparse gather-scatter

Self-contained — no external SGNNET imports needed. Deploys to any machine with PyTorch.

Usage:
  python3 bench_hardware_compare.py --device cuda   # 5060ti
  python3 bench_hardware_compare.py --device mps    # Mac Studio / Mac Mini
  python3 bench_hardware_compare.py --device cpu    # CPU baseline
  python3 bench_hardware_compare.py --device auto   # auto-detect best
"""

from __future__ import annotations
import argparse, json, time, sys, os, platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ───────────────────────────────────────────────────────────────────

N_IN = 25088        # VGG16 feature dim
N_CLASSES = 10      # Imagenette
BATCH_SIZES = [1, 32, 128, 512]
WARMUP_ITERS = 20
BENCH_ITERS = 100
TRAIN_ITERS = 50    # training steps to benchmark

# SGNNET efficiency config (step199 / step235 ΔW proj)
SGNNET_N = 2048
SGNNET_D = 16
SGNNET_K_HH = 2
SGNNET_K_IN = 25
SGNNET_K_ITER = 5
SGNNET_ALPHA_REFLECT = 0.5


# ── VGG16 FC ─────────────────────────────────────────────────────────────────

class VGG16_FC(nn.Module):
    """VGG16 fully-connected classifier head. 123.6M FLOPs."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(N_IN, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, N_CLASSES),
        )

    def forward(self, x):
        return self.fc(x)

    @staticmethod
    def flops():
        # 25088×4096 + 4096×4096 + 4096×1000 (multiply-add = 2 ops each)
        return 2 * (25088 * 4096 + 4096 * 4096 + 4096 * 1000)


# ── SGNNET (self-contained) ──────────────────────────────────────────────────

def _build_smallworld_conn(N, K_local, K_random, n_groups, seed=42):
    """Build small-world connectivity [N, K_local+K_random]."""
    rng = np.random.RandomState(seed)
    K = K_local + K_random
    conn = np.zeros((N, K), dtype=np.int64)
    group_size = N // n_groups

    for i in range(N):
        g = i // group_size
        g_start = g * group_size
        g_end = min(g_start + group_size, N)

        # Local connections within group
        candidates = [j for j in range(g_start, g_end) if j != i]
        if len(candidates) < K_local:
            candidates = [j for j in range(N) if j != i]
        local = rng.choice(candidates, size=min(K_local, len(candidates)), replace=False)
        conn[i, :len(local)] = local

        # Random long-range connections
        all_others = [j for j in range(N) if j != i and j not in local]
        n_rand = min(K_random, len(all_others))
        if n_rand > 0:
            remote = rng.choice(all_others, size=n_rand, replace=False)
            conn[i, K_local:K_local + n_rand] = remote

    return torch.from_numpy(conn)


def _build_input_conn(N_in, N_hidden, K_in, seed=42):
    """Build input fan-in [N_hidden, K_in]."""
    rng = np.random.RandomState(seed)
    conn = np.zeros((N_hidden, K_in), dtype=np.int64)
    for i in range(N_hidden):
        conn[i] = rng.choice(N_in, size=K_in, replace=False)
    return torch.from_numpy(conn)


class SGNNET_Bench(nn.Module):
    """Minimal SGNNET for benchmarking — matches step235 ΔW projection forward pass.

    Includes: Fourier encoding, sparse gather routing, ΔW projection, reflection,
    thresholding, L2 normalize, sparse readout. No AH (per step235 finding).
    """

    def __init__(self, N=SGNNET_N, D=SGNNET_D, K_hh=SGNNET_K_HH,
                 K_in=SGNNET_K_IN, K_iter=SGNNET_K_ITER,
                 alpha_reflect=SGNNET_ALPHA_REFLECT, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.N = N
        self.D = D
        self.K_iter = K_iter
        self.alpha_reflect = alpha_reflect

        K_local = max(1, K_hh - max(1, K_hh // 4))
        K_random = K_hh - K_local
        n_groups = max(8, N // 8)

        # Connectivity tables (not parameters, just index buffers)
        self.register_buffer('conn_in', _build_input_conn(N_IN, N, K_in, seed))
        self.register_buffer('conn_hh', _build_smallworld_conn(N, K_local, K_random, n_groups, seed))

        # Sparse readout: C_ho [N, N_classes] — which hidden neurons vote for which class
        C_ho = torch.zeros(N, N_CLASSES)
        per_class = N // N_CLASSES
        for c in range(N_CLASSES):
            C_ho[c * per_class:(c + 1) * per_class, c] = 1.0
        self.register_buffer('C_ho', C_ho)

        # Learnable parameters
        self.W_pos = nn.Parameter(torch.randn(N + N_CLASSES, D) * 0.1)
        self.theta = nn.Parameter(torch.full((N,), 0.01))

        # Fourier encoding projection
        self.fc_encode = nn.Linear(1, D, bias=False)

        # Output scoring
        self.fc_out = nn.Linear(D, 1, bias=False)

    @staticmethod
    def flops():
        # 3 × N × K_hh × D × K_iter (main routing)
        # + N × K_in × D (input seeding)
        # + N × N_classes × D (readout)
        route = 3 * SGNNET_N * SGNNET_K_HH * SGNNET_D * SGNNET_K_ITER
        seed = SGNNET_N * SGNNET_K_IN * SGNNET_D
        readout = SGNNET_N * N_CLASSES * SGNNET_D
        return route + seed + readout

    def forward(self, x):
        B = x.shape[0]

        # ── Seed: input → hidden activations ──
        # x: [B, N_IN] → gather via conn_in → [B, N, K_in] → encode → [B, N, D]
        x_flat = x.unsqueeze(-1)                          # [B, N_IN, 1]
        x_gathered = x_flat[:, self.conn_in, :]           # [B, N, K_in, 1]
        x_summed = x_gathered.sum(dim=2)                  # [B, N, 1]
        Z = self.fc_encode(x_summed)                      # [B, N, D]
        Z = F.normalize(Z, dim=-1)

        # ── Route: K_iter iterations of sparse message passing ──
        conn_hh = self.conn_hh
        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        W_pos_hidden = F.normalize(self.W_pos[:self.N], dim=-1)  # [N, D]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.K_iter):
            Z_fwd = F.relu(Z - theta_pos)                 # [B, N, D]
            Z_nb = Z_fwd[:, conn_hh, :]                   # [B, N, K_hh, D]

            # ΔW projection: project onto delta vector (receiver - sender)
            W_recv = W_pos_hidden.unsqueeze(1).expand(-1, conn_hh.shape[1], -1)  # [N, K, D]
            W_send = W_pos_hidden[conn_hh]                                        # [N, K, D]
            delta_w = F.normalize(W_recv - W_send, dim=-1)                        # [N, K, D]
            proj = (Z_nb * delta_w.unsqueeze(0)).sum(dim=-1, keepdim=True)        # [B, N, K, 1]
            Z_struct = (Z_nb * proj.abs()).sum(dim=2)                             # [B, N, D]

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        # ── Readout: hidden → class scores ──
        # Sparse: C_ho selects which neurons vote for each class
        Z_out = torch.einsum('bnd,nc->bcd', Z, self.C_ho)  # [B, N_CLASSES, D]
        W_out = F.normalize(self.W_pos[self.N:], dim=-1)    # [N_CLASSES, D]
        scores = (Z_out * W_out.unsqueeze(0)).sum(dim=-1)   # [B, N_CLASSES]

        return scores


# ── Benchmark utilities ──────────────────────────────────────────────────────

def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def sync_device(device):
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_gpu_info(device):
    """Get GPU memory and utilization."""
    info = {}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1e6
        info["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        info["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
    elif device.type == "mps":
        info["gpu_name"] = "Apple MPS"
        try:
            info["gpu_mem_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1e6, 1)
        except:
            pass
    else:
        info["gpu_name"] = "CPU"
    return info


def get_system_info(device):
    """Collect system identification."""
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "device_type": device.type,
    }
    info.update(get_gpu_info(device))
    return info


def bench_inference(model, device, batch_sizes=BATCH_SIZES,
                    warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark inference: latency (ms) and throughput (samples/sec) per batch size."""
    model.eval()
    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, N_IN, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
        sync_device(device)

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(iters):
                sync_device(device)
                t0 = time.perf_counter()
                _ = model(x)
                sync_device(device)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

        times = sorted(times)
        # Drop top/bottom 10% for stable median
        trim = max(1, len(times) // 10)
        trimmed = times[trim:-trim]

        latency_ms = np.median(trimmed)
        throughput = bs / (latency_ms / 1000)

        # GPU memory snapshot
        mem_info = get_gpu_info(device)

        results[bs] = {
            "latency_ms": round(latency_ms, 3),
            "throughput_sps": round(throughput, 1),
            "latency_p5_ms": round(np.percentile(trimmed, 5), 3),
            "latency_p95_ms": round(np.percentile(trimmed, 95), 3),
            "gpu_mem_mb": mem_info.get("gpu_mem_allocated_mb", None),
        }
        print(f"    bs={bs:4d}  latency={latency_ms:8.3f}ms  "
              f"throughput={throughput:10.1f} samp/s  "
              f"p5={np.percentile(trimmed, 5):.3f}  p95={np.percentile(trimmed, 95):.3f}")

    return results


def bench_training(model, device, batch_size=128,
                   warmup=10, iters=TRAIN_ITERS):
    """Benchmark training: time per step (ms) and throughput (samples/sec)."""
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
    sync_device(device)

    # Timed runs
    times = []
    for _ in range(iters):
        sync_device(device)
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        sync_device(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = sorted(times)
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]

    step_ms = np.median(trimmed)
    throughput = batch_size / (step_ms / 1000)
    mem_info = get_gpu_info(device)

    result = {
        "batch_size": batch_size,
        "step_ms": round(step_ms, 3),
        "throughput_sps": round(throughput, 1),
        "step_p5_ms": round(np.percentile(trimmed, 5), 3),
        "step_p95_ms": round(np.percentile(trimmed, 95), 3),
        "gpu_mem_mb": mem_info.get("gpu_mem_allocated_mb", None),
    }
    print(f"    bs={batch_size}  step={step_ms:.3f}ms  "
          f"throughput={throughput:.1f} samp/s  "
          f"p5={np.percentile(trimmed, 5):.3f}  p95={np.percentile(trimmed, 95):.3f}")

    return result


# ── nvidia-smi monitor (CUDA only) ──────────────────────────────────────────

def nvidia_smi_snapshot():
    """Take a single nvidia-smi snapshot."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "gpu_util_pct": int(parts[0]),
                "mem_util_pct": int(parts[1]),
                "mem_used_mb": int(parts[2]),
                "mem_total_mb": int(parts[3]),
                "power_w": float(parts[4]),
            }
    except:
        pass
    return None


def nvidia_smi_during(fn, interval=0.1):
    """Run fn() while sampling nvidia-smi in a thread. Return (fn_result, smi_samples)."""
    import threading

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            s = nvidia_smi_snapshot()
            if s:
                samples.append(s)
            time.sleep(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    result = fn()
    stop.set()
    t.join(timeout=2)

    return result, samples


def summarize_smi(samples):
    """Summarize nvidia-smi samples."""
    if not samples:
        return {}
    return {
        "gpu_util_avg": round(np.mean([s["gpu_util_pct"] for s in samples]), 1),
        "gpu_util_max": max(s["gpu_util_pct"] for s in samples),
        "mem_used_avg_mb": round(np.mean([s["mem_used_mb"] for s in samples]), 0),
        "mem_used_max_mb": max(s["mem_used_mb"] for s in samples),
        "power_avg_w": round(np.mean([s["power_w"] for s in samples]), 1),
        "power_max_w": round(max(s["power_w"] for s in samples), 1),
        "n_samples": len(samples),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-sizes", default="1,32,128,512",
                        help="Comma-separated batch sizes for inference bench")
    parser.add_argument("--train-bs", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--train-iters", type=int, default=TRAIN_ITERS)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    device = get_device(args.device)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"\n{'='*70}")
    print(f"HARDWARE BENCHMARK: VGG16 FC vs SGNNET")
    print(f"{'='*70}")

    sys_info = get_system_info(device)
    print(f"Host:   {sys_info['hostname']}")
    print(f"Device: {device} ({sys_info.get('gpu_name', 'CPU')})")
    print(f"Platform: {sys_info['platform']}")

    all_results = {"system": sys_info, "models": {}}

    for model_name, ModelClass in [("VGG16_FC", VGG16_FC), ("SGNNET", SGNNET_Bench)]:
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}")
        flops = ModelClass.flops()
        n_params = sum(p.numel() for p in ModelClass().parameters())
        print(f"  FLOPs: {flops:,} ({flops/1e6:.2f}M)")
        print(f"  Params: {n_params:,} ({n_params/1e3:.1f}K)")
        print(f"{'─'*60}")

        model = ModelClass().to(device)

        # ── Inference benchmark ──
        print(f"\n  INFERENCE:")
        if device.type == "cuda":
            def run_infer():
                return bench_inference(model, device, batch_sizes,
                                       args.warmup, args.iters)
            infer_results, smi_samples = nvidia_smi_during(run_infer, interval=0.2)
            smi_infer = summarize_smi(smi_samples)
            if smi_infer:
                print(f"  nvidia-smi: GPU util avg={smi_infer['gpu_util_avg']}% "
                      f"max={smi_infer['gpu_util_max']}%  "
                      f"mem={smi_infer['mem_used_avg_mb']:.0f}MB  "
                      f"power={smi_infer['power_avg_w']}W")
        else:
            infer_results = bench_inference(model, device, batch_sizes,
                                            args.warmup, args.iters)
            smi_infer = {}

        # ── Training benchmark ──
        print(f"\n  TRAINING (bs={args.train_bs}):")
        # Re-init model for clean training benchmark
        model = ModelClass().to(device)
        if device.type == "cuda":
            def run_train():
                return bench_training(model, device, args.train_bs,
                                      warmup=10, iters=args.train_iters)
            train_results, smi_samples = nvidia_smi_during(run_train, interval=0.2)
            smi_train = summarize_smi(smi_samples)
            if smi_train:
                print(f"  nvidia-smi: GPU util avg={smi_train['gpu_util_avg']}% "
                      f"max={smi_train['gpu_util_max']}%  "
                      f"mem={smi_train['mem_used_avg_mb']:.0f}MB  "
                      f"power={smi_train['power_avg_w']}W")
        else:
            train_results = bench_training(model, device, args.train_bs,
                                           warmup=10, iters=args.train_iters)
            smi_train = {}

        all_results["models"][model_name] = {
            "flops": flops,
            "params": n_params,
            "inference": infer_results,
            "training": train_results,
            "nvidia_smi_inference": smi_infer,
            "nvidia_smi_training": smi_train,
        }

        # Cleanup
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Summary comparison ──
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    vgg = all_results["models"]["VGG16_FC"]
    sgn = all_results["models"]["SGNNET"]

    print(f"\n  FLOPs ratio:  SGNNET/VGG = {sgn['flops']/vgg['flops']*100:.2f}%")
    print(f"  Param ratio:  SGNNET/VGG = {sgn['params']/vgg['params']*100:.3f}%")

    print(f"\n  {'Metric':<30} {'VGG16 FC':>12} {'SGNNET':>12} {'Ratio':>10}")
    print(f"  {'─'*64}")

    # Inference at bs=128
    if 128 in batch_sizes:
        v_lat = vgg["inference"][128]["latency_ms"]
        s_lat = sgn["inference"][128]["latency_ms"]
        print(f"  {'Infer latency (bs=128)':30} {v_lat:>10.3f}ms {s_lat:>10.3f}ms {v_lat/s_lat:>9.2f}x")

        v_thr = vgg["inference"][128]["throughput_sps"]
        s_thr = sgn["inference"][128]["throughput_sps"]
        print(f"  {'Infer throughput (bs=128)':30} {v_thr:>9.1f}/s {s_thr:>9.1f}/s {s_thr/v_thr:>9.2f}x")

    # Inference at bs=1 (latency-sensitive)
    if 1 in batch_sizes:
        v_lat = vgg["inference"][1]["latency_ms"]
        s_lat = sgn["inference"][1]["latency_ms"]
        print(f"  {'Infer latency (bs=1)':30} {v_lat:>10.3f}ms {s_lat:>10.3f}ms {v_lat/s_lat:>9.2f}x")

    # Training
    v_step = vgg["training"]["step_ms"]
    s_step = sgn["training"]["step_ms"]
    print(f"  {'Train step (bs={})'.format(args.train_bs):30} {v_step:>10.3f}ms {s_step:>10.3f}ms {v_step/s_step:>9.2f}x")

    v_thr = vgg["training"]["throughput_sps"]
    s_thr = sgn["training"]["throughput_sps"]
    print(f"  {'Train throughput':30} {v_thr:>9.1f}/s {s_thr:>9.1f}/s {s_thr/v_thr:>9.2f}x")

    # nvidia-smi summary (CUDA only)
    if vgg.get("nvidia_smi_training"):
        print(f"\n  nvidia-smi during training:")
        v_smi = vgg["nvidia_smi_training"]
        s_smi = sgn["nvidia_smi_training"]
        print(f"  {'GPU util avg':30} {v_smi.get('gpu_util_avg',0):>10.1f}% {s_smi.get('gpu_util_avg',0):>10.1f}%")
        print(f"  {'GPU mem used avg':30} {v_smi.get('mem_used_avg_mb',0):>9.0f}MB {s_smi.get('mem_used_avg_mb',0):>9.0f}MB")
        print(f"  {'Power draw avg':30} {v_smi.get('power_avg_w',0):>10.1f}W {s_smi.get('power_avg_w',0):>10.1f}W")

    print(f"\n  Theoretical speedup (FLOPs): {vgg['flops']/sgn['flops']:.1f}x")
    if 128 in batch_sizes:
        actual = vgg["inference"][128]["latency_ms"] / sgn["inference"][128]["latency_ms"]
        print(f"  Actual speedup (infer bs=128): {actual:.1f}x")
        print(f"  Efficiency gap: {actual / (vgg['flops']/sgn['flops']) * 100:.1f}% of theoretical")

    # Save results
    out_path = args.output or f"bench_hardware_{sys_info['hostname']}_{device.type}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Convert int keys to strings for JSON
    for mname in all_results["models"]:
        infer = all_results["models"][mname]["inference"]
        all_results["models"][mname]["inference"] = {str(k): v for k, v in infer.items()}
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
