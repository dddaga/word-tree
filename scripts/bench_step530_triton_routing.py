"""Step 530 / 530b: Triton fused routing kernel vs PyTorch baseline.

TARGET: beat the current best of 0.280ms (bench_step811 V2 max-autotune fp32,
K=5, bs=32, RTX 5060 Ti).

VARIANTS
  V_ref            : SGNNET_DeltaProj PyTorch eager fp32 (step268 class)
  V_ref_ma         : V_ref + torch.compile max-autotune    ← 0.280ms king
  V_triton         : SGNNET_DeltaProjTriton eager Triton (per-step, K launches)
  V_triton_cg      : SGNNET_DeltaProjTriton + CUDA Graph   (step803 fix: int32 conn)
  V_triton_c       : SGNNET_DeltaProjTriton + torch.compile reduce-overhead
  V_fused_parallel : fused K_iter=5, single launch, fixed-Z neighbor reads (approx)
  V_fused_parallel_cg : V_fused_parallel + CUDA Graph
  V_fused_seq      : fused K_iter=5, single launch, ping-pong buffers (approx)
  V_fused_seq_cg   : V_fused_seq + CUDA Graph

step530b additions: V_fused_parallel*, V_fused_seq* — fused kernel iteration #2.
See src/sgnnet/triton/routing_kernel_fused.py for design rationale.

Hardware: RTX 5060 Ti, CUDA 12.8, PyTorch 2.11+cu128, Triton 3.6.0
Shapes tested: N=2048 D=16 K_hh=2 K_iter=5 bs=32 fp32

Output:
  results/bench_step530_triton_routing__<slot>.json
  Console: speedup table vs 0.280ms baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--iters",   type=int, default=200)
parser.add_argument("--warmup",  type=int, default=50)
parser.add_argument("--bs",      type=int, default=32)
parser.add_argument("--k_iter",  type=int, default=5)
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
assert DEVICE.type == "cuda", "step530 is CUDA-only (Triton requires CUDA)"

import os
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"bench_step530_triton_routing__{SLOT}.json"

print(f"GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")
print(f"slot={SLOT}  bs={args.bs}  k_iter={args.k_iter}  iters={args.iters}")

try:
    import triton
    print(f"triton: {triton.__version__}  OK")
    HAS_TRITON = True
except ImportError:
    print("triton: NOT INSTALLED")
    HAS_TRITON = False

# ── Model constants ────────────────────────────────────────────────────────
N_IN = 25088; N_OUT = 10; SEED = 42
N = 2048; D = 16; K_HH = 2; K_IN = 25
K_ITER = args.k_iter
ALPHA_REFLECT = 0.5
BASELINE_MS = 0.280  # step811 V2 max-autotune K=5

# ── Reference PyTorch model (SGNNET_DeltaProj from step268) ───────────────
from src.sgnnet.model_smallworld   import SGNNET_SmallWorld
from src.sgnnet.model_resonant     import SGNNET_Resonant


class SGNNET_DeltaProj(nn.Module):
    """Exact copy of SGNNET_DeltaProj from train_step268 (no Triton)."""
    def __init__(self, N_hidden, N_out, D_, N_in, K_in, K_iter, K_local, K_random,
                 n_groups, alpha_reflect, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D_, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)
        self.alpha_reflect = alpha_reflect

    @property
    def W_pos(self): return self.base.W_pos

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh   = self.base.conn_hh
        N_h       = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h       = self.base.W_pos[:N_h]
        delta_w   = W_h.unsqueeze(1) - W_h[conn_hh]
        dw_norm   = F.normalize(delta_w, dim=-1).unsqueeze(0)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]
            proj     = (Z_nb * dw_norm).sum(-1, keepdim=True)
            Z_nb     = Z_nb * proj.abs()
            Z_struct = Z_nb.sum(dim=2)
            Z_rem    = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_rem
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


def make_ref():
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    return SGNNET_DeltaProj(
        N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), alpha_reflect=ALPHA_REFLECT, seed=SEED,
    )


# ── Triton model ──────────────────────────────────────────────────────────
from src.sgnnet.triton.sgnnet_proj_triton import SGNNET_DeltaProjTriton


def make_triton():
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    m = SGNNET_DeltaProjTriton(
        N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), alpha_reflect=ALPHA_REFLECT, seed=SEED,
    )
    # NOTE: tick_epoch() must be called AFTER .to(DEVICE) so dw_norm cache
    # is on the same device as W_pos. _get_dw_norm() auto-refreshes on
    # device mismatch, but explicit call here is clearer.
    return m


# ── Bench utilities ────────────────────────────────────────────────────────

def sync():
    torch.cuda.synchronize()


def bench(fn, x, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        sync()
        ts = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            fn(x)
            sync()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return {
        "median_ms":       float(np.median(ts)),
        "p95_ms":          float(np.percentile(ts, 95)),
        "p99_ms":          float(np.percentile(ts, 99)),
        "mean_ms":         float(np.mean(ts)),
        "throughput_sps":  float(args.bs / (np.median(ts) / 1000.0)),
    }


def peak_mem():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _prepare_model(make_fn):
    """Build model, move to device, call tick_epoch() (needed for Triton dw_norm cache)."""
    m = make_fn().to(DEVICE).eval()
    if hasattr(m, "tick_epoch"):
        m.tick_epoch()   # ensures dw_norm cache is on DEVICE, not CPU
    return m


def run_eager(make_fn, desc):
    m = _prepare_model(make_fn)
    x = torch.randn(args.bs, N_IN, device=DEVICE)
    reset_mem()
    r = bench(m, x, args.warmup, args.iters)
    r["peak_mib"] = peak_mem(); r["desc"] = desc
    del m; torch.cuda.empty_cache()
    return r


def run_compiled(make_fn, mode, desc):
    print(f"  compiling: mode={mode}")
    m = _prepare_model(make_fn)
    m = torch.compile(m, mode=mode, dynamic=False)
    x = torch.randn(args.bs, N_IN, device=DEVICE)
    reset_mem()
    r = bench(m, x, args.warmup, args.iters)
    r["peak_mib"] = peak_mem(); r["desc"] = desc
    del m; torch.cuda.empty_cache()
    return r


def run_cuda_graph(make_fn, desc):
    """CUDA Graph with fixed-shape replay — works because conn_hh is now int32."""
    m = _prepare_model(make_fn)
    x_static = torch.randn(args.bs, N_IN, device=DEVICE)

    # Stream warm-up
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(3):
                _ = m(x_static)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            out_static = m(x_static)

    # Warm-up replays
    x_data = torch.randn(args.bs, N_IN, device=DEVICE)
    for _ in range(args.warmup):
        x_static.copy_(x_data)
        g.replay()
    sync()

    # Timed
    ts = []
    for _ in range(args.iters):
        sync()
        t0 = time.perf_counter()
        x_static.copy_(x_data)
        g.replay()
        sync()
        ts.append((time.perf_counter() - t0) * 1000.0)

    ts = np.asarray(ts)
    mem = peak_mem()
    del m; torch.cuda.empty_cache()
    return {
        "median_ms":      float(np.median(ts)),
        "p95_ms":         float(np.percentile(ts, 95)),
        "p99_ms":         float(np.percentile(ts, 99)),
        "mean_ms":        float(np.mean(ts)),
        "throughput_sps": float(args.bs / (np.median(ts) / 1000.0)),
        "peak_mib": mem, "desc": desc,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    results = {
        "device":    str(DEVICE),
        "gpu":       torch.cuda.get_device_name(0),
        "torch":     torch.__version__,
        "bs":        args.bs,
        "k_iter":    K_ITER,
        "N":         N, "D": D, "K_HH": K_HH,
        "baseline_ms": BASELINE_MS,
        "has_triton": HAS_TRITON,
        "variants":  {},
    }

    print("\n── V_ref: DeltaProj PyTorch eager fp32 ──")
    results["variants"]["V_ref"] = run_eager(make_ref, "DeltaProj eager fp32")

    print("\n── V_ref_ma: DeltaProj PyTorch max-autotune ──")
    try:
        results["variants"]["V_ref_ma"] = run_compiled(make_ref, "max-autotune",
                                                       "DeltaProj max-autotune fp32")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["variants"]["V_ref_ma"] = {"error": str(e)}

    def make_triton_fused(mode):
        """Build SGNNET_DeltaProjTriton with a specific fused_mode."""
        def _make():
            m = make_triton()
            m._fused_mode = mode
            return m
        return _make

    if HAS_TRITON:
        print("\n── V_triton: DeltaProjTriton eager Triton (per-step) ──")
        try:
            results["variants"]["V_triton"] = run_eager(make_triton,
                                                        "DeltaProjTriton eager Triton")
        except Exception as e:
            print(f"  FAILED: {e}")
            results["variants"]["V_triton"] = {"error": str(e)}

        print("\n── V_triton_cg: DeltaProjTriton + CUDA Graph ──")
        try:
            results["variants"]["V_triton_cg"] = run_cuda_graph(make_triton,
                                                                 "DeltaProjTriton CUDA Graph")
        except Exception as e:
            print(f"  FAILED: {e}")
            results["variants"]["V_triton_cg"] = {"error": str(e)}

        print("\n── V_triton_c: DeltaProjTriton + torch.compile reduce-overhead ──")
        try:
            results["variants"]["V_triton_c"] = run_compiled(make_triton,
                                                              "reduce-overhead",
                                                              "DeltaProjTriton compiled")
        except Exception as e:
            print(f"  FAILED: {e}")
            results["variants"]["V_triton_c"] = {"error": str(e)}

        # ── step530b: fused K_iter variants ─────────────────────────────
        print("\n── V_fused_parallel: fused K_iter, single launch, fixed-Z neighbor ──")
        try:
            results["variants"]["V_fused_parallel"] = run_eager(
                make_triton_fused("parallel"),
                "fused_parallel K_iter single launch (approx semantics)"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results["variants"]["V_fused_parallel"] = {"error": str(e)}

        print("\n── V_fused_parallel_cg: V_fused_parallel + CUDA Graph ──")
        try:
            results["variants"]["V_fused_parallel_cg"] = run_cuda_graph(
                make_triton_fused("parallel"),
                "fused_parallel CUDA Graph"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results["variants"]["V_fused_parallel_cg"] = {"error": str(e)}

        print("\n── V_fused_seq: fused K_iter, single launch, ping-pong buffers ──")
        try:
            results["variants"]["V_fused_seq"] = run_eager(
                make_triton_fused("sequential"),
                "fused_sequential K_iter single launch (approx semantics)"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results["variants"]["V_fused_seq"] = {"error": str(e)}

        print("\n── V_fused_seq_cg: V_fused_seq + CUDA Graph ──")
        try:
            results["variants"]["V_fused_seq_cg"] = run_cuda_graph(
                make_triton_fused("sequential"),
                "fused_sequential CUDA Graph"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results["variants"]["V_fused_seq_cg"] = {"error": str(e)}

    else:
        print("\n  Triton not available — skipping V_triton* variants")

    # ── Summary table ─────────────────────────────────────────────────
    print("\n\n========== STEP 530/530b SUMMARY ==========")
    print(f"{'Variant':<24} {'median_ms':>10} {'p95_ms':>8} {'sps':>9} "
          f"{'mem_MiB':>9} {'vs_ref_eager':>13} {'vs_0.280ms':>12}")

    valid = {k: v for k, v in results["variants"].items()
             if isinstance(v, dict) and "median_ms" in v}
    ref_ms = valid.get("V_ref", {}).get("median_ms", 1.0)

    for k, r in valid.items():
        vs_ref   = ref_ms / r["median_ms"]
        vs_base  = BASELINE_MS / r["median_ms"]
        print(f"{k:<24} {r['median_ms']:>10.3f} {r['p95_ms']:>8.3f} "
              f"{r['throughput_sps']:>9.0f} {r.get('peak_mib', -1):>9.1f} "
              f"{vs_ref:>12.2f}× {vs_base:>11.2f}×")

    best_k = min(valid, key=lambda k: valid[k]["median_ms"])
    best_ms = valid[best_k]["median_ms"]
    speedup = BASELINE_MS / best_ms
    print(f"\nBest variant : {best_k} @ {best_ms:.3f}ms")
    print(f"vs 0.280ms   : {speedup:.2f}× {'SPEEDUP' if speedup > 1.0 else 'REGRESSION'}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
