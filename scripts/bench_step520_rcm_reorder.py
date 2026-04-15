"""Step 520: Reverse Cuthill-McKee (RCM) index reordering for L2 cache locality.

MOTIVATION
==========
The gather op `Z[:, conn_hh, :]` is the hot path in every K_iter step.
conn_hh indices are random Watts-Strogatz — neighbors have no spatial
coherence in memory. RCM permutes node IDs so that frequently co-accessed
nodes are contiguous in memory, improving L2 cache hit rate.

This is a MATHEMATICALLY IDENTICAL transformation — only memory layout
changes. Expected speedup: 1.3–2.4× on memory-bandwidth-bound kernels.

VARIANTS
========
  V1_baseline     : original random conn_hh layout (step500 winner)
  V_rcm_inf       : RCM-reordered, inference only
  V_rcm_train     : RCM-reordered, full training step

Benchmark uses same CUDA Event harness as step800/801/802.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

# ── Config ────────────────────────────────────────────────────────────────────
N_IN, N_CLASSES = 25088, 10
N, D, K_HH, K_ITER, K_IN = 2048, 16, 2, 5, 25
ALPHA_AHEBB, ALPHA_REFLECT, ALPHA_TURING = 1.0, 0.5, 0.0
N_WARMUP, N_TIMED, TRAIN_BS = 3, 50, 128
BENCH_SIZES = [1, 32, 128, 512]


def make_base(device):
    torch.manual_seed(42)
    n_groups = max(8, N // 8)
    K_local = max(1, K_HH - max(1, K_HH // 4))
    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_CLASSES, D=D, N_in=N_IN,
        K_in=K_IN, K_local=K_local, K_random=K_HH - K_local,
        n_groups=n_groups, K_iter=K_ITER,
        norm_mode="l2", encoding_mode="fourier",
    ).to(device)
    res = SGNNET_Resonant(
        base=sw, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
    ).to(device)
    return SGNNET_AntiHebbian(base=res, alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(device)


def apply_rcm(model):
    """Apply RCM permutation to conn_hh and W_pos in-place.

    Builds an undirected adjacency from conn_hh, runs RCM, permutes
    both the row order (which neuron is which index) and the neighbor
    values (neighbor indices must reflect the new numbering).
    Returns the permutation array for reference.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
    except ImportError:
        print("  WARNING: scipy not available — skipping RCM. pip install scipy")
        return None

    # Access conn_hh from the wrapped model hierarchy
    base = model.m.base  # SGNNET_SmallWorld
    conn_hh = base.conn_hh.cpu().numpy()  # [N, K_hh]
    N_h = conn_hh.shape[0]

    # Build symmetric adjacency (add reverse edges)
    rows = np.repeat(np.arange(N_h), K_HH)
    cols = conn_hh.ravel()
    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(N_h, N_h))
    A = (A + A.T)  # symmetrize

    # RCM permutation (reverse=True → bandwidth-minimizing ordering)
    # .copy() ensures contiguous array — RCM output may have negative strides
    perm = np.ascontiguousarray(reverse_cuthill_mckee(A, symmetric_mode=True))
    inv_perm = np.ascontiguousarray(np.argsort(perm))

    # Reorder conn_hh: new_conn_hh[inv_perm[i], :] = inv_perm[conn_hh[i, :]]
    new_conn = inv_perm[conn_hh[perm]]  # rows permuted, values remapped
    device = base.conn_hh.device
    base.conn_hh.data = torch.from_numpy(new_conn).to(device)

    # Reorder W_pos (first N_h rows = hidden neurons)
    with torch.no_grad():
        model.m.W_pos.data[:N_h] = model.m.W_pos.data[:N_h][perm]

    # Also reorder conn_in rows
    base.conn_in.data = base.conn_in.data[perm]

    # NOTE: spatial_coords has shape [N_in, D-1] — it encodes INPUT neuron positions,
    # not hidden neuron positions. Do NOT reorder by perm (size N_h).
    # Reorder C_ho_mask (shape [N_h, N_classes]) — hidden-neuron → class readout
    base.C_ho_mask.data = base.C_ho_mask.data[perm]

    return perm


# ── Timing harness ─────────────────────────────────────────────────────────
def sync(): torch.cuda.synchronize()


def cuda_event_latency(model, x, device, n_warmup=N_WARMUP, n_timed=N_TIMED,
                       train=False):
    if train:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        y = torch.randint(0, N_CLASSES, (x.shape[0],), device=device)
    else:
        model.eval()

    def _step(xi):
        if train:
            opt.zero_grad()
            loss = crit(model(xi), y)
            loss.backward(); opt.step()
        else:
            with torch.no_grad(): model(xi)

    for _ in range(n_warmup): _step(x)
    sync()

    start_e, end_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_timed):
        sync(); start_e.record(); _step(x); end_e.record(); sync()
        times.append(start_e.elapsed_time(end_e))

    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    med = float(np.median(trimmed))
    return {"median_ms": round(med, 3), "p5_ms": round(float(np.percentile(trimmed, 5)), 3),
            "p95_ms": round(float(np.percentile(trimmed, 95)), 3),
            "throughput_sps": round(x.shape[0] / (med / 1000), 1)}


def nvidia_smi_snapshot():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            if len(parts) >= 2:
                return {"gpu_util_pct": int(parts[0]), "power_w": float(parts[1])}
    except Exception: pass
    return None


def poll_gpu_during(fn, interval=0.2):
    samples, stop_ev = [], threading.Event()
    def _poll():
        while not stop_ev.is_set():
            s = nvidia_smi_snapshot()
            if s: samples.append(s)
            time.sleep(interval)
    t = threading.Thread(target=_poll, daemon=True)
    t.start(); result = fn(); stop_ev.set(); t.join(timeout=2)
    if samples:
        return result, round(float(np.mean([s["gpu_util_pct"] for s in samples])), 1), \
               round(float(np.mean([s["power_w"] for s in samples])), 1)
    return result, None, None


def bench_variant(label, model_fn, device, apply_reorder=False, compile_mode="reduce-overhead"):
    print(f"\n── {label} ──")
    torch.cuda.empty_cache()
    model = model_fn(device)

    if apply_reorder:
        print("  Applying RCM permutation... ", end="", flush=True)
        perm = apply_rcm(model)
        if perm is None:
            print("SKIPPED (scipy unavailable)")
            return None
        print(f"done (bandwidth reduction applied)")

    print(f"  Compiling (mode={compile_mode!r})... ", end="", flush=True)
    t0 = time.perf_counter()
    model = torch.compile(model, mode=compile_mode)
    with torch.no_grad():
        model(torch.randn(TRAIN_BS, N_IN, device=device))
    sync()
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    inf_results = {}
    for bs in BENCH_SIZES:
        x = torch.randn(bs, N_IN, device=device)
        r = cuda_event_latency(model, x, device, train=False)
        inf_results[bs] = r
        print(f"  Inf bs={bs:<4}  med={r['median_ms']:.3f}ms  tput={r['throughput_sps']:.0f}sps")

    x_train = torch.randn(TRAIN_BS, N_IN, device=device)
    print(f"  Train bs={TRAIN_BS}  ", end="", flush=True)
    train_r, gpu_util, gpu_pow = poll_gpu_during(
        lambda: cuda_event_latency(model, x_train, device, train=True))
    print(f"med={train_r['median_ms']:.3f}ms  tput={train_r['throughput_sps']:.0f}sps  "
          f"GPU={gpu_util}%  power={gpu_pow}W")

    del model; torch.cuda.empty_cache()
    return {"compile_mode": compile_mode, "rcm_applied": apply_reorder,
            "inference": {str(k): v for k, v in inf_results.items()},
            "training": train_r, "gpu_util_avg": gpu_util, "gpu_power_avg_w": gpu_pow}


def main():
    parser = argparse.ArgumentParser(description="Step 520: RCM index reordering benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available"); sys.exit(1)

    device = torch.device(args.device if ":" in args.device else f"{args.device}:0")
    torch.cuda.set_device(device)

    print(f"\n{'='*70}")
    print("STEP 520 — RCM Index Reordering Benchmark")
    print(f"{'='*70}")
    print(f"Device : {device} ({torch.cuda.get_device_name(device)})")
    print(f"Config : N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"Hypothesis: RCM reorders random conn_hh indices for L2 cache locality")
    print(f"Expected  : +30-140% if memory-BW-bound (step802 showed GPU util=12.7% with bf16)")

    results = {}
    results["V1_baseline"] = bench_variant(
        "V1_baseline (random conn_hh, reduce-overhead)", make_base, device,
        apply_reorder=False)
    results["V_rcm"] = bench_variant(
        "V_rcm (RCM-reordered conn_hh, reduce-overhead)", make_base, device,
        apply_reorder=True)

    print(f"\n{'='*70}")
    print("SUMMARY — speedup vs V1_baseline")
    print(f"{'='*70}")
    ref_ms = results["V1_baseline"]["training"]["median_ms"] if results["V1_baseline"] else None
    for name, r in results.items():
        if r is None: print(f"  {name}: SKIPPED"); continue
        t_ms = r["training"]["median_ms"]
        sp = round(ref_ms / t_ms, 3) if ref_ms else "—"
        print(f"  {name:<30}  train={t_ms:.3f}ms  speedup={sp}x  GPU={r['gpu_util_avg']}%")
    if ref_ms:
        print(f"\n  V1 training reference: {ref_ms:.3f}ms (step802 V1 was 5.299ms)")

    out_data = {
        "system": {"hostname": platform.node(), "gpu": torch.cuda.get_device_name(device),
                   "torch_version": torch.__version__},
        "config": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER},
        "hypothesis": "RCM reordering improves L2 cache hit rate for gather op",
        "variants": results,
    }
    out_path = args.output or str(ROOT / "results" / "bench_step520_rcm_5060ti.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
