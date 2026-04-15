"""Step 803: torch.cuda.make_graphed_callables on the inner K_iter routing loop.

MOTIVATION
==========
Step 802 established that fp32 reduce-overhead is the production baseline at 5.3ms/step.
The next kernel-launch ceiling: each of the 5 K_iter routing iterations dispatches its
own sequence of CPU-launched CUDA kernels (relu, gather, mul, sum, normalize).
torch.cuda.make_graphed_callables captures those kernels into a CUDA Graph — a single
replay command replaces N separate kernel launches, eliminating CPU-GPU round-trips
inside the loop.

HYPOTHESIS: graphed routing step removes per-iteration CPU overhead, giving measurable
speedup at small batch sizes where kernel-launch cost dominates compute.

DESIGN
======
1. routing_step(Z, supp_w, theta_pos, conn_hh_idx) — one K_iter iteration (pure CUDA ops):
     relu(Z - theta_pos) → gather conn_hh → AH-suppress (multiply supp_w) → sum
     → add reflection accumulator → normalize
   supp_w is static (precomputed before the loop) so no data-dependent branching.
   NOTE: reflection accumulator is NOT graphed because it is stateful across iterations
   (CUDA Graphs require fixed input/output shapes and no cross-iteration state writes).
   Only the per-step Z→Z_new transform is graphed.

2. V1: torch.compile reduce-overhead (current production — step802 baseline)
3. V9: torch.compile + make_graphed_callables on routing_step
   The graphed callable is used inside a wrapper model that calls it K_ITER times.

CUDA Graph constraints honoured:
  - All inputs/outputs are CUDA tensors with fixed shapes (B locked at bench time)
  - No Python control flow inside graphed function
  - 3 warmup runs before capture (PyTorch requirement)
  - No in-place ops on tensors outside the function's scope
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
from typing import Optional

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


# ── Model factory (identical to step802) ──────────────────────────────────────
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


# ── Standalone routing step (for CUDA Graph capture) ─────────────────────────
def routing_step(
    Z: torch.Tensor,           # [B, N, D] — current state
    supp_w: torch.Tensor,      # [1, N, K_hh, 1] — static suppression weights
    theta_pos: torch.Tensor,   # [1, N, 1] — per-neuron threshold
    conn_hh_idx: torch.Tensor, # [N, K_hh] — neighbour indices (int64)
) -> torch.Tensor:             # [B, N, D] — Z_new (does NOT include reflection; see note)
    """One K_iter routing iteration: relu → gather → AH-suppress → sum → normalize.

    NOTE: reflection is stateful across iterations (Z_reflected accumulates).
    CUDA Graphs capture a fixed kernel sequence with fixed tensor addresses — you
    cannot update an external accumulator inside the graph without capturing its
    mutation too, which would require the exact same tensor object every call.
    Reflection is therefore computed outside the graphed callable (in the wrapper)
    and added to Z before the next graphed call. The overhead is two tensor ops
    per iteration outside the graph, which is negligible compared to the gather/sum.
    """
    Z_fwd    = F.relu(Z - theta_pos)                     # [B, N, D]
    Z_nb     = Z_fwd[:, conn_hh_idx, :]                  # [B, N, K_hh, D]
    Z_struct = (Z_nb * supp_w).sum(dim=2)                # [B, N, D]
    return F.normalize(Z_struct.clamp(-10, 10), dim=-1)  # [B, N, D]


class GraphedRoutingModel(nn.Module):
    """Wraps SGNNET_AntiHebbian forward using a graphed routing_step callable.

    The CUDA Graph is batch-size specific — one graph per (B,) shape.
    At inference we typically use a fixed batch size; training uses TRAIN_BS.
    For shape flexibility we fall back to eager routing for unseen batch sizes.
    """

    def __init__(self, base_model: SGNNET_AntiHebbian, graphed_fn, captured_bs: int):
        super().__init__()
        self.m           = base_model
        self.graphed_fn  = graphed_fn
        self.captured_bs = captured_bs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # ── Seed (unchanged) ──────────────────────────────────────────────────
        Z = self.m.m.base._seed(x)                       # [B, N, D]

        # ── Precompute static per-forward quantities ───────────────────────────
        theta_pos   = self.m.m.theta.abs().unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        conn_hh_idx = self.m.m.base.conn_hh                              # [N, K_hh]
        N_h         = self.m.m.base.N_hidden
        W_n         = F.normalize(self.m.m.W_pos[:N_h], dim=-1)         # [N, D]
        pos_sim     = (W_n.unsqueeze(1) * W_n[conn_hh_idx]).sum(-1)     # [N, K_hh]
        supp_w      = (
            1.0 - self.m.alpha_ahebb * pos_sim.clamp(min=0)
        ).unsqueeze(0).unsqueeze(-1)                                     # [1, N, K_hh, 1]

        # ── K_iter routing loop ───────────────────────────────────────────────
        Z_reflected = torch.zeros_like(Z)

        if B == self.captured_bs and self.graphed_fn is not None:
            # Fast path: use CUDA-graphed routing step
            for _ in range(K_ITER):
                # Graphed step: relu → gather → suppress → normalize
                Z_new = self.graphed_fn(Z, supp_w, theta_pos, conn_hh_idx)

                # Reflection (stateful accumulator — computed outside graph)
                Z_fwd_outside = F.relu(Z - theta_pos)
                Z_remainder   = Z_fwd_outside - Z
                Z_reflected   = ALPHA_REFLECT * Z_reflected + Z_remainder
                Z             = F.normalize((Z_new + Z_reflected).clamp(-10, 10), dim=-1)
        else:
            # Fallback: eager routing (batch size not captured)
            for _ in range(K_ITER):
                Z_fwd    = F.relu(Z - theta_pos)
                Z_nb     = Z_fwd[:, conn_hh_idx, :]
                Z_struct = (Z_nb * supp_w).sum(dim=2)
                Z_remainder = Z_fwd - Z
                Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
                Z_new       = Z_struct + Z_reflected
                Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.m.base._readout(Z)


def make_graphed_model(
    base_model: SGNNET_AntiHebbian,
    capture_bs: int,
    device: torch.device,
) -> tuple[GraphedRoutingModel, bool, str]:
    """Attempt to capture routing_step into a CUDA Graph.

    Returns (model, success, message).
    """
    try:
        # Build sample inputs for capture — shapes must exactly match runtime shapes.
        # conn_hh_idx is int64 and must be passed as a plain buffer (not captured as
        # a graph input since make_graphed_callables only supports float tensors as
        # capturable inputs; int tensors must be embedded in the closure or passed
        # via a wrapper that converts them into the graph as constants).
        #
        # Strategy: wrap routing_step in a closure that closes over conn_hh_idx
        # and supp_w_sample, exposing only (Z, supp_w, theta_pos) as graphed inputs.
        # This matches the CUDA Graph requirement that all graphed inputs be
        # floating-point CUDA tensors.

        N_h         = base_model.m.base.N_hidden
        conn_hh_idx = base_model.m.base.conn_hh                   # [N, K_hh] — closed over
        W_n         = F.normalize(base_model.m.W_pos[:N_h], dim=-1)
        pos_sim     = (W_n.unsqueeze(1) * W_n[conn_hh_idx]).sum(-1)
        supp_w_ref  = (
            1.0 - base_model.alpha_ahebb * pos_sim.clamp(min=0)
        ).unsqueeze(0).unsqueeze(-1).contiguous()                  # [1, N, K_hh, 1]

        theta_ref   = base_model.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

        # Sample inputs — must be contiguous CUDA float tensors
        Z_sample        = torch.randn(capture_bs, N, D, device=device, requires_grad=False).contiguous()
        supp_w_sample   = supp_w_ref.clone().contiguous()
        theta_sample    = theta_ref.clone().contiguous()

        # Closure: conn_hh_idx is int64 and is baked into the graph as a constant
        # (it never changes — it's a registered buffer). This avoids exposing
        # non-float tensors as graph inputs.
        _conn = conn_hh_idx  # captured in closure

        def _routing_step_closed(Z_in, supp_w_in, theta_in):
            Z_fwd    = F.relu(Z_in - theta_in)
            Z_nb     = Z_fwd[:, _conn, :]
            Z_struct = (Z_nb * supp_w_in).sum(dim=2)
            return F.normalize(Z_struct.clamp(-10, 10), dim=-1)

        # Warmup (mandatory before capture)
        for _ in range(3):
            _ = _routing_step_closed(Z_sample, supp_w_sample, theta_sample)
        torch.cuda.synchronize()

        graphed = torch.cuda.make_graphed_callables(
            _routing_step_closed,
            (Z_sample, supp_w_sample, theta_sample),
        )
        msg = f"make_graphed_callables succeeded (capture_bs={capture_bs})"
        print(f"  [V9] {msg}")

        # Build wrapper model that uses the graphed callable
        class _GraphedModel(nn.Module):
            def __init__(self, base, gfn, cbs):
                super().__init__()
                self.m           = base
                self._graphed    = gfn
                self.captured_bs = cbs
                # store conn_hh_idx as buffer-like attribute for the fallback path
                self._conn       = _conn

            def forward(self, x):
                B = x.shape[0]
                Z = self.m.m.base._seed(x)

                theta_pos   = self.m.m.theta.abs().unsqueeze(0).unsqueeze(-1)
                conn_hh_idx = self._conn
                N_h         = self.m.m.base.N_hidden
                W_n_        = F.normalize(self.m.m.W_pos[:N_h], dim=-1)
                pos_sim_    = (W_n_.unsqueeze(1) * W_n_[conn_hh_idx]).sum(-1)
                supp_w_     = (
                    1.0 - self.m.alpha_ahebb * pos_sim_.clamp(min=0)
                ).unsqueeze(0).unsqueeze(-1)

                Z_reflected = torch.zeros_like(Z)

                if B == self.captured_bs:
                    for _ in range(K_ITER):
                        Z_new       = self._graphed(Z, supp_w_, theta_pos)
                        Z_fwd_out   = F.relu(Z - theta_pos)
                        Z_remainder = Z_fwd_out - Z
                        Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
                        Z           = F.normalize((Z_new + Z_reflected).clamp(-10, 10), dim=-1)
                else:
                    for _ in range(K_ITER):
                        Z_fwd       = F.relu(Z - theta_pos)
                        Z_nb        = Z_fwd[:, conn_hh_idx, :]
                        Z_struct    = (Z_nb * supp_w_).sum(dim=2)
                        Z_remainder = Z_fwd - Z
                        Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
                        Z           = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)

                return self.m.m.base._readout(Z)

        model = _GraphedModel(base_model, graphed, capture_bs).to(device)
        return model, True, msg

    except Exception as e:
        msg = f"make_graphed_callables FAILED: {type(e).__name__}: {e}"
        print(f"  [V9] {msg}")
        return None, False, msg


# ── Measurement utilities (identical to step802) ──────────────────────────────
def sync():
    torch.cuda.synchronize()


def cuda_event_latency(model, x, device, n_warmup=N_WARMUP, n_timed=N_TIMED,
                       train=False):
    """CUDA Event timing — median of n_timed runs (fp32, no scaler)."""
    if train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        y = torch.randint(0, N_CLASSES, (x.shape[0],), device=device)
    else:
        model.eval()

    def _step(x_in):
        if train:
            optimizer.zero_grad()
            out  = model(x_in)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
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


def bench_variant(label, model, device, n_warmup=N_WARMUP, n_timed=N_TIMED):
    """Benchmark a pre-built model over all BENCH_SIZES and TRAIN_BS."""
    print(f"\n── {label} ──")
    torch.cuda.empty_cache()

    inf_results = {}
    for bs in BENCH_SIZES:
        x = torch.randn(bs, N_IN, device=device)
        r = cuda_event_latency(model, x, device, n_warmup=n_warmup, n_timed=n_timed,
                               train=False)
        inf_results[bs] = r
        print(f"  Inf bs={bs:<4}  med={r['median_ms']:.3f}ms  "
              f"p5={r['p5_ms']:.3f}  p95={r['p95_ms']:.3f}  "
              f"tput={r['throughput_sps']:.0f} sps")

    x_train = torch.randn(TRAIN_BS, N_IN, device=device)
    print(f"  Train bs={TRAIN_BS}  ", end="", flush=True)

    def _train_bench():
        return cuda_event_latency(model, x_train, device, n_warmup=n_warmup,
                                  n_timed=n_timed, train=True)

    train_r, gpu_util, gpu_pow = poll_gpu_during(_train_bench, interval=0.2)
    print(f"med={train_r['median_ms']:.3f}ms  "
          f"tput={train_r['throughput_sps']:.0f} sps  "
          f"GPU util={gpu_util}%  power={gpu_pow}W")

    return {
        "inference":       {str(k): v for k, v in inf_results.items()},
        "training":        train_r,
        "gpu_util_avg":    gpu_util,
        "gpu_power_avg_w": gpu_pow,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Step 803: make_graphed_callables on K_iter routing loop"
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

    torch_ver = torch.__version__
    print(f"\n{'='*70}")
    print("STEP 803 — make_graphed_callables on K_iter routing loop")
    print(f"{'='*70}")
    print(f"Device     : {device} ({torch.cuda.get_device_name(device)})")
    print(f"PyTorch    : {torch_ver}")
    print(f"Config     : N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"Hypothesis : graphed routing eliminates CPU kernel-launch overhead "
          f"across {K_ITER} serial iterations.")
    print(f"NOTE       : reflection accumulator computed outside graph (stateful "
          f"across iterations — cannot be graphed without full-loop capture).")

    results = {}

    # ── V1: reduce-overhead fp32 (step802 production baseline) ───────────────
    print("\n[Building V1...]")
    m_v1 = make_base(device)
    print("  Compiling V1 (reduce-overhead)... ", end="", flush=True)
    t0   = time.perf_counter()
    m_v1 = torch.compile(m_v1, mode="reduce-overhead")
    # Trigger compilation
    _x_warm = torch.randn(TRAIN_BS, N_IN, device=device)
    with torch.no_grad():
        m_v1(_x_warm)
    sync()
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    del _x_warm

    results["V1_reduce_overhead_fp32"] = bench_variant(
        "V1 — reduce-overhead fp32 (step802 baseline)",
        m_v1, device, n_warmup=args.warmup, n_timed=args.timed,
    )
    results["V1_reduce_overhead_fp32"]["compile_mode"]    = "reduce-overhead"
    results["V1_reduce_overhead_fp32"]["graphed_routing"] = False
    del m_v1
    torch.cuda.empty_cache()

    # ── V9: reduce-overhead + make_graphed_callables on routing step ──────────
    print("\n[Building V9...]")
    m_base_v9 = make_base(device)

    # Attempt CUDA Graph capture for each bench size we'll use.
    # For simplicity capture at TRAIN_BS (the most performance-critical shape).
    # Inference at other batch sizes falls back to eager routing inside the wrapper.
    graphed_model, graph_ok, graph_msg = make_graphed_model(m_base_v9, TRAIN_BS, device)

    if graph_ok and graphed_model is not None:
        print("  Compiling V9 (reduce-overhead + graphed routing)... ", end="", flush=True)
        t0 = time.perf_counter()
        graphed_model_compiled = torch.compile(graphed_model, mode="reduce-overhead")
        _x_warm = torch.randn(TRAIN_BS, N_IN, device=device)
        with torch.no_grad():
            graphed_model_compiled(_x_warm)
        sync()
        print(f"done ({time.perf_counter()-t0:.1f}s)")
        del _x_warm

        results["V9_graphed_routing"] = bench_variant(
            "V9 — reduce-overhead + graphed routing step",
            graphed_model_compiled, device, n_warmup=args.warmup, n_timed=args.timed,
        )
        results["V9_graphed_routing"]["compile_mode"]    = "reduce-overhead"
        results["V9_graphed_routing"]["graphed_routing"] = True
        results["V9_graphed_routing"]["graph_capture_bs"] = TRAIN_BS
        results["V9_graphed_routing"]["graph_status"]    = graph_msg
        del graphed_model_compiled
    else:
        print(f"\n  [V9] SKIPPED — {graph_msg}")
        results["V9_graphed_routing"] = {
            "skipped":       True,
            "reason":        graph_msg,
            "graphed_routing": False,
            "compile_mode":  "reduce-overhead",
        }
    torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — inference + training speedup vs V1")
    print(f"{'='*70}")

    v1_train_ms = results["V1_reduce_overhead_fp32"]["training"]["median_ms"]
    v1_inf_ms   = results["V1_reduce_overhead_fp32"]["inference"][str(TRAIN_BS)]["median_ms"]

    print(f"\n  {'Variant':<40}  {'Train (ms)':<12}  {'vs V1':<8}  "
          f"{'Inf bs={TRAIN_BS} (ms)':<18}  {'vs V1'}")
    for name, r in results.items():
        if r.get("skipped"):
            print(f"  {name:<40}  SKIPPED — {r['reason'][:40]}")
            continue
        t_ms    = r["training"]["median_ms"]
        i_ms    = r["inference"][str(TRAIN_BS)]["median_ms"]
        t_su    = round(v1_train_ms / t_ms, 3)
        i_su    = round(v1_inf_ms   / i_ms, 3)
        print(f"  {name:<40}  {t_ms:<12.3f}  {t_su:<8.3f}x  {i_ms:<18.3f}  {i_su:.3f}x")

    print(f"\n  Inference across batch sizes (median ms):")
    header = f"  {'Variant':<40}" + "".join(f"  bs={bs:<5}" for bs in BENCH_SIZES)
    print(header)
    for name, r in results.items():
        if r.get("skipped"):
            continue
        row = f"  {name:<40}"
        for bs in BENCH_SIZES:
            row += f"  {r['inference'][str(bs)]['median_ms']:<7.3f}"
        print(row)

    # ── Save results ──────────────────────────────────────────────────────────
    out_data = {
        "system": {
            "hostname":      platform.node(),
            "gpu":           torch.cuda.get_device_name(device),
            "torch_version": torch_ver,
            "cuda_version":  torch.version.cuda,
        },
        "config": {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "alpha_ahebb": ALPHA_AHEBB, "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing": ALPHA_TURING,
        },
        "bench_config": {
            "n_warmup": args.warmup,
            "n_timed":  args.timed,
            "bench_sizes": BENCH_SIZES,
            "train_bs": TRAIN_BS,
        },
        "hypothesis": (
            "make_graphed_callables on routing_step eliminates CPU kernel-launch "
            "overhead across K_iter=5 serial iterations, measurable at small BS."
        ),
        "design_note": (
            "reflection accumulator computed outside graph (stateful across iterations). "
            "graphed callable: relu → gather conn_hh → AH-suppress (supp_w) → sum → normalize. "
            "captured_bs=TRAIN_BS; inference at other batch sizes falls back to eager."
        ),
        "variants": results,
    }

    out_path = args.output or str(ROOT / "results" / "bench_step803_graphed_5060ti.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
