"""Step 800: Paper-grade FLOPs audit + CUDA Event latency harness.

MOTIVATION
==========
The reported "0.98M FLOPs" is routing-only message-passing MACs: N×K_iter×K_hh×D×2.
A 2026-04-14 audit identified major missing ops: seed gather (K_in=50, ~3.3M MACs),
AH suppression multiplies, F.normalize, ReLU, reflection. True total ≈ 6.5M per sample.

This script:
  1. Computes analytical FLOPs for every op category (paper-grade table).
  2. Measures wall-clock latency via CUDA Events (median 50 runs, 3 warmup).
  3. Runs torch.profiler for op-level memory + kernel breakdown.
  4. Prints ncu command for hardware FLOPs cross-check on this machine.

Outputs:
  results/bench_step800_flops_5060ti.json
  results/bench_step800_torch_profile/ (Chrome trace)

Usage:
  python scripts/bench_step800_flops_profiler.py --device cuda
  python scripts/bench_step800_flops_profiler.py --device cuda --no-profiler
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.profiler

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian


# ── Config (step199 efficiency config) ────────────────────────────────────────
N_IN          = 25088
N_CLASSES     = 10
N             = 2048
D             = 16
K_HH          = 2
K_ITER        = 5
K_IN          = 25     # fan-in per hidden neuron (step199 config)
ALPHA_AHEBB   = 1.0
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
BENCH_BS      = 128
N_WARMUP      = 3
N_TIMED       = 50


# ── Model factory ─────────────────────────────────────────────────────────────
def make_model(device: torch.device) -> SGNNET_AntiHebbian:
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


# ── Analytical FLOPs ──────────────────────────────────────────────────────────
def compute_analytical_flops(n: int = N, d: int = D, k_hh: int = K_HH,
                              k_iter: int = K_ITER, k_in: int = K_IN,
                              n_out: int = N_CLASSES) -> dict:
    """Compute per-sample FLOPs for every op category.

    Conventions:
      - 1 MAC = 2 FLOPs (1 multiply + 1 add)
      - Element-wise add/sub/relu = 1 FLOP per element
      - L2 normalize: N*D (sq) + N (sum+sqrt) + N*D (div) ≈ 3*N*D FLOPs
    """
    # ── 1. Seed: input fan-in gather ──────────────────────────────────────────
    # Z_in = A_input[:, conn_in, :].sum(dim=2)
    # A_input has shape [N_in, D]. Gathering K_in neighbors per hidden neuron,
    # then summing K_in vectors of dim D → (K_in-1) adds per (n, d) element.
    # We count K_in multiplied by D as the gather-sum ops (conservative: K_in adds ≈ K_in MACs).
    seed_gather_flops = n * k_in * d        # K_in adds per (neuron, dim) = N*K_in*D
    seed_normalize_flops = 3 * n * d        # L2 norm after seed

    # ── 2. Routing loop (K_iter iters) ────────────────────────────────────────
    # Per iter:
    #   a. relu(Z - theta)                : N*D (subtract) + N*D (relu) = 2*N*D
    #   b. Z_nb = Z_fwd[:, conn_hh, :]   : gather, 0 FLOPs (memory)
    #   c. AH suppress: Z_nb * supp_w    : N*K_hh*D mults (the MISSING ops in old formula)
    #   d. Z_struct = (Z_nb*supp).sum(2) : N*K_hh*D adds
    #   e. Z_remainder = Z_fwd - Z       : N*D
    #   f. Z_reflected = alpha*Z_reflected + Z_remainder : 2*N*D (mul + add)
    #   g. Z_new = Z_struct + Z_reflected : N*D
    #   h. F.normalize(Z_new)            : 3*N*D

    per_iter_relu_theta   = 2 * n * d                  # a
    per_iter_ah_multiply  = n * k_hh * d               # c (multiplications)
    per_iter_ah_sum       = n * k_hh * d               # d (additions)
    per_iter_reflect      = 3 * n * d                  # e + f = 1 + 2
    per_iter_combine      = n * d                      # g
    per_iter_normalize    = 3 * n * d                  # h

    per_iter = (per_iter_relu_theta + per_iter_ah_multiply + per_iter_ah_sum
                + per_iter_reflect + per_iter_combine + per_iter_normalize)
    routing_flops = k_iter * per_iter

    # Old formula (routing-only gather+sum, no AH multiply, no normalize):
    routing_old_formula = k_iter * n * k_hh * d * 2   # the published "0.98M" formula

    # ── 3. Readout: C_ho einsum + dot-product ─────────────────────────────────
    # A_out = einsum("bhd,ho->bod", Z, C_ho)  : N * N_out * D mults + adds = 2*N*n_out*D
    # dot product score: N_out * D (elwise mul + sum) = 2*N_out*D
    readout_flops = 2 * n * n_out * d + 2 * n_out * d

    total_true   = seed_gather_flops + seed_normalize_flops + routing_flops + readout_flops
    total_macs   = total_true // 2   # FLOPs → MACs (÷2, since most are MAC pairs)

    return {
        "seed_gather_flops":       seed_gather_flops,
        "seed_normalize_flops":    seed_normalize_flops,
        "routing_flops_true":      routing_flops,
        "routing_flops_old_formula": routing_old_formula,
        "readout_flops":           readout_flops,
        "total_true_flops":        total_true,
        "total_true_macs":         total_macs,
        "per_iter_breakdown": {
            "relu_theta":      per_iter_relu_theta,
            "ah_multiply":     per_iter_ah_multiply,
            "ah_sum":          per_iter_ah_sum,
            "reflect":         per_iter_reflect,
            "combine":         per_iter_combine,
            "normalize":       per_iter_normalize,
            "total_per_iter":  per_iter,
        },
        "undercounting_ratio": round(total_true / routing_old_formula, 2),
    }


# ── CUDA Event latency ────────────────────────────────────────────────────────
def cuda_event_latency(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    n_warmup: int = N_WARMUP,
    n_timed: int = N_TIMED,
    train: bool = False,
) -> dict:
    """Paper-grade latency: CUDA Events, median of n_timed runs."""
    if train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        y = torch.randint(0, N_CLASSES, (x.shape[0],), device=device)
    else:
        model.eval()

    # Warmup (no timing)
    for _ in range(n_warmup):
        if train:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(x)
    torch.cuda.synchronize(device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    times_ms  = []

    for _ in range(n_timed):
        torch.cuda.synchronize(device)
        start_evt.record()
        if train:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(x)
        end_evt.record()
        torch.cuda.synchronize(device)
        times_ms.append(start_evt.elapsed_time(end_evt))

    times_ms.sort()
    trim  = max(1, len(times_ms) // 10)
    trimmed = times_ms[trim:-trim]
    import statistics
    med  = statistics.median(trimmed)
    p5   = trimmed[int(len(trimmed) * 0.05)]
    p95  = trimmed[int(len(trimmed) * 0.95)]
    bs   = x.shape[0]

    return {
        "median_ms":      round(med, 3),
        "p5_ms":          round(p5, 3),
        "p95_ms":         round(p95, 3),
        "throughput_sps": round(bs / (med / 1000), 1),
        "batch_size":     bs,
        "n_timed":        n_timed,
    }


# ── torch.profiler breakdown ──────────────────────────────────────────────────
def run_torch_profiler(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    trace_dir: str,
) -> list[dict]:
    """Collect per-op table from torch.profiler (kernel time + memory).

    Note: with_flops=True only counts aten::mm / aten::conv — returns 0 for
    gather/scatter. We collect it anyway for op timing and memory transfer info.
    """
    model.eval()
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]

    # 1 warmup step inside profiler
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,   # only meaningful for matmul/conv — 0 for gather
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(3):
            with torch.no_grad():
                model(x)

    table = []
    for evt in prof.key_averages():
        if evt.cuda_time_total > 0 or evt.self_cpu_time_total > 0:
            table.append({
                "op":            evt.key,
                "cuda_us":       round(evt.cuda_time_total, 1),
                "cpu_us":        round(evt.self_cpu_time_total, 1),
                "mem_bytes":     evt.self_cuda_memory_usage,
                "flops_counted": evt.flops,
                "count":         evt.count,
            })
    table.sort(key=lambda r: r["cuda_us"], reverse=True)
    return table[:30]   # top-30 by CUDA time


# ── VGG16 FC reference ────────────────────────────────────────────────────────
def vgg16_fc_flops() -> dict:
    """VGG16 classifier FLOPs: 3 Linear layers."""
    # L1: 25088 → 4096, L2: 4096 → 4096, L3: 4096 → 1000
    l1 = 2 * 25088 * 4096
    l2 = 2 * 4096  * 4096
    l3 = 2 * 4096  * 1000
    total = l1 + l2 + l3
    return {"l1": l1, "l2": l2, "l3": l3, "total": total, "total_m": round(total / 1e6, 2)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Step 800: FLOPs audit + latency profiler")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--batch-size",  type=int, default=BENCH_BS)
    parser.add_argument("--warmup",      type=int, default=N_WARMUP)
    parser.add_argument("--timed",       type=int, default=N_TIMED)
    parser.add_argument("--no-profiler", action="store_true",
                        help="Skip torch.profiler (saves time if only need latency)")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script is CUDA-only.")
        sys.exit(1)

    device = torch.device(args.device if ":" in args.device else f"{args.device}:0")
    torch.cuda.set_device(device)

    print(f"\n{'='*70}")
    print("STEP 800 — SGNNET FLOPs Audit + Paper-Grade Latency Profiler")
    print(f"{'='*70}")
    print(f"Device : {device} ({torch.cuda.get_device_name(device)})")
    print(f"Config : N={N} D={D} K_hh={K_HH} K_iter={K_ITER} K_in={K_IN}")

    # ── 1. Analytical FLOPs ──────────────────────────────────────────────────
    print(f"\n── Analytical FLOPs audit ──")
    flops = compute_analytical_flops()
    vgg   = vgg16_fc_flops()

    print(f"  Seed gather      (N×K_in×D)     : {flops['seed_gather_flops']/1e6:.3f} M FLOPs")
    print(f"  Seed normalize                  : {flops['seed_normalize_flops']/1e6:.3f} M FLOPs")
    print(f"  Routing (true, K_iter iters)    : {flops['routing_flops_true']/1e6:.3f} M FLOPs")
    print(f"    └ Old formula (gather-only)   : {flops['routing_flops_old_formula']/1e6:.3f} M FLOPs  (← was the '0.98M' claim)")
    print(f"  Readout (C_ho einsum + dot)     : {flops['readout_flops']/1e6:.3f} M FLOPs")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  TOTAL true FLOPs (per sample)   : {flops['total_true_flops']/1e6:.3f} M")
    print(f"  TOTAL MACs (per sample)         : {flops['total_true_macs']/1e6:.3f} M")
    print(f"  Old formula undercounting ratio : {flops['undercounting_ratio']}×")
    print(f"\n  VGG16 FC total FLOPs            : {vgg['total']/1e6:.1f} M")
    print(f"  Routing MACs ratio vs VGG FC    : {vgg['total']/flops['routing_flops_old_formula']:.0f}× fewer (old claim)")
    print(f"  True FLOPs ratio vs VGG FC      : {vgg['total']/flops['total_true_flops']:.0f}× fewer")

    print(f"\n  Per-iter routing breakdown:")
    bd = flops["per_iter_breakdown"]
    for k, v in bd.items():
        print(f"    {k:<20}: {v/1e3:.1f} K FLOPs")

    # ── 2. CUDA Event latency ────────────────────────────────────────────────
    print(f"\n── CUDA Event latency (median of {args.timed} runs, {args.warmup} warmup) ──")
    model = make_model(device)
    model.eval()
    torch.cuda.empty_cache()

    latency_results = {}
    for bs in [1, 32, 128, args.batch_size]:
        bs = int(bs)
        x  = torch.randn(bs, N_IN, device=device)
        r  = cuda_event_latency(model, x, device, args.warmup, args.timed, train=False)
        latency_results[bs] = r
        print(f"  Inference bs={bs:<4}  median={r['median_ms']:.3f}ms  "
              f"p5={r['p5_ms']:.3f}  p95={r['p95_ms']:.3f}  "
              f"tput={r['throughput_sps']:.0f} sps")

    # Training step latency
    print(f"  Training   bs={BENCH_BS:<4}  ", end="", flush=True)
    x_train = torch.randn(BENCH_BS, N_IN, device=device)
    train_r = cuda_event_latency(model, x_train, device, args.warmup, args.timed, train=True)
    latency_results["train"] = train_r
    print(f"median={train_r['median_ms']:.3f}ms  "
          f"tput={train_r['throughput_sps']:.0f} sps")

    # ── 3. Efficiency vs peak ─────────────────────────────────────────────────
    print(f"\n── Efficiency vs peak ──")
    try:
        props = torch.cuda.get_device_properties(device)
        # Peak FP32 TFLOPS (approx from SM count × FP32 cores × clock)
        # Better: use reported from nvidia-smi or hardcoded for RTX 5060 Ti
        sm_count = props.multi_processor_count
        clock_ghz = props.clock_rate / 1e6
        # RTX 5060 Ti: SM120 — 2 FP32 units per clock per CUDA core, ~4352 CUDA cores
        # Theoretical: ~22.1 TFLOPS FP32. Report as measured.
        peak_tflops_note = "RTX 5060 Ti spec: ~22.1 TFLOPS FP32 (reported)"
        peak_tflops = 22.1

        lat_s    = train_r["median_ms"] / 1000
        batch_flops = flops["total_true_flops"] * BENCH_BS
        achieved_tflops = batch_flops / lat_s / 1e12

        print(f"  {peak_tflops_note}")
        print(f"  SMs: {sm_count}  Clock: {clock_ghz:.2f} GHz")
        print(f"  Batch FLOPs (bs={BENCH_BS}): {batch_flops/1e9:.3f} GFLOPs")
        print(f"  Training step latency: {train_r['median_ms']:.3f} ms")
        print(f"  Achieved TFLOPS: {achieved_tflops:.4f}")
        print(f"  Utilization vs peak: {achieved_tflops/peak_tflops*100:.2f}%")
        efficiency = {
            "peak_tflops_fp32": peak_tflops,
            "achieved_tflops":  round(achieved_tflops, 6),
            "utilization_pct":  round(achieved_tflops / peak_tflops * 100, 2),
        }
    except Exception as e:
        print(f"  Could not compute efficiency: {e}")
        efficiency = {}

    # ── 4. torch.profiler op breakdown ───────────────────────────────────────
    profiler_table = []
    if not args.no_profiler:
        print(f"\n── torch.profiler op breakdown (bs=128) ──")
        trace_dir = str(ROOT / "results" / "bench_step800_torch_profile")
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        x_p = torch.randn(BENCH_BS, N_IN, device=device)
        try:
            profiler_table = run_torch_profiler(model, x_p, device, trace_dir)
            print(f"  Top ops by CUDA time:")
            for r in profiler_table[:15]:
                print(f"    {r['op']:<45} cuda={r['cuda_us']:>9.1f}µs  "
                      f"count={r['count']:>4}  mem={r['mem_bytes']/1024:.1f}KB")
            print(f"  Chrome trace: {trace_dir}")
        except Exception as e:
            print(f"  torch.profiler failed: {e}")
    else:
        print(f"\n── torch.profiler skipped (--no-profiler) ──")

    # ── 5. ncu instructions ───────────────────────────────────────────────────
    print(f"\n── ncu command for hardware FLOPs cross-check ──")
    script_path = Path(__file__).resolve()
    ncu_cmd = (
        f"ncu --metrics "
        "sm__sass_thread_inst_executed_op_fadd.sum,"
        "sm__sass_thread_inst_executed_op_fmul.sum,"
        "sm__sass_thread_inst_executed_op_ffma.sum,"
        "l2__read_bytes.sum,dram__bytes.sum "
        f"--target-processes all "
        f"python3 {script_path} --device cuda --no-profiler --timed 5"
    )
    print(f"  {ncu_cmd}")
    print(f"  FP ops total = fadd + fmul + 2*ffma (FMA counts as 2 ops)")
    print(f"  If within 20% of {flops['total_true_flops']/1e6:.2f}M, formula validated.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_data = {
        "system": {
            "hostname":      platform.node(),
            "gpu":           torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        },
        "config": {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "K_in": K_IN,
            "alpha_ahebb": ALPHA_AHEBB, "alpha_reflect": ALPHA_REFLECT,
        },
        "analytical_flops": flops,
        "vgg16_fc_flops":   vgg,
        "cuda_event_latency": {str(k): v for k, v in latency_results.items()},
        "efficiency_vs_peak": efficiency,
        "profiler_top30":   profiler_table,
    }
    out_path = args.output or str(ROOT / "results" / "bench_step800_flops_5060ti.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
