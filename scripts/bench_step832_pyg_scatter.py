"""Step 832: PyTorch Geometric `torch_scatter` wall-clock probe.

MOTIVATION
==========
Current best SGNNET inference: 0.280ms @ K=5 (bench_step811 V2 max-autotune) on
5060ti. Gather `Z_fwd[:, conn_hh, :]` + sum-over-K_hh is the routing hot path.

PyG ships `torch_scatter.scatter_add` — hand-tuned CUDA kernels for gather+reduce
operating on edge-list [E, D] format, avoiding the [B, N, K_hh, D] intermediate
tensor. step803 (CUDA graph) hit exactly this intermediate as its blocker.

VARIANTS
  V_ref    : SGNNET_AntiHebbian standard (fancy-index gather + K_hh-sum) — ref
  V_ref_c  : Ref + torch.compile max-autotune (should match step811 V2 0.280ms)
  V_scat   : Edge-list rewrite using scatter_add — eager fp32
  V_scat_c : V_scat + torch.compile max-autotune
  V_scat_cg: V_scat + CUDA Graph capture (fixed-shape inference)

Hypothesis (prior): 2-3× if memory-BW-bound. step802 showed GPU util ≈ 12.7%
in bf16 inference — we are BW-starved. scatter_add avoids the 4D intermediate
materialization → predicted win.

Install:
    ssh 5060ti "venv/bin/pip install torch-scatter -f \
      https://data.pyg.org/whl/torch-$(torch-version).html"

To run:
    python -u scripts/bench_step832_pyg_scatter.py --device cuda
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--iters",  type=int, default=300)
parser.add_argument("--warmup", type=int, default=80)
parser.add_argument("--bs",     type=int, default=32)
parser.add_argument("--k_iter", type=int, default=5, help="5 or 4 (test both for the K=4 win)")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
assert DEVICE.type == "cuda", "step832 is CUDA-only"

print(f"GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")

# Optional PyG import — variants V_scat* require it, others work without
try:
    from torch_scatter import scatter_add
    HAS_PYG = True
    import torch_scatter
    print(f"torch_scatter: {torch_scatter.__version__}  ✓")
except ImportError as e:
    HAS_PYG = False
    print(f"torch_scatter NOT INSTALLED — V_scat* variants will skip. Error: {e}")
    print("Install: pip install torch-scatter -f https://data.pyg.org/whl/torch-<ver>.html")

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

N_IN = 25088; N_OUT = 10; SEED = 42
N = 2048; D = 16; K_HH = 2; K_IN = 25
OUT_PATH = ROOT / "results" / f"bench_step832_pyg_scatter_k{args.k_iter}.json"


def make_ref():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                             K_in=K_IN, K_iter=args.k_iter, K_local=K_l, K_random=K_r,
                             n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
                          beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                          resonance_threshold=0.0)
    return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


class SGNNET_ScatterAH(nn.Module):
    """SGNNET_AntiHebbian rewritten using torch_scatter.scatter_add.

    Edge-list format: edges expanded from conn_hh [N, K_hh] to [E=N*K_hh].
    Routing loop uses scatter_add to accumulate per-receiver, avoiding the
    [B, N, K_hh, D] intermediate tensor.
    """
    def __init__(self, ah_model: SGNNET_AntiHebbian):
        super().__init__()
        self.m = ah_model.m
        self.alpha = ah_model.alpha_ahebb

        # Pre-compute edge list once
        conn_hh = self.m.base.conn_hh                # [N, K_hh] int64
        N_h = self.m.base.N_hidden
        K_h = conn_hh.shape[1]
        self.N_h = N_h
        self.K_h = K_h

        # Flatten: edge i → (recv=i // K_h, send=conn_hh.flat[i])
        edge_send = conn_hh.reshape(-1)              # [E]
        edge_recv = torch.arange(N_h, device=conn_hh.device).repeat_interleave(K_h)  # [E]
        self.register_buffer("edge_send", edge_send)
        self.register_buffer("edge_recv", edge_recv)

        # Pre-compute per-edge suppression weight (static wpos variant)
        W_n = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)   # [N, K_hh]
        supp_e = (1.0 - self.alpha * pos_sim.clamp(min=0)).reshape(-1)  # [E]
        self.register_buffer("supp_e", supp_e)

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def forward(self, x):
        Z = self.m.base._seed(x)                                  # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1) # [1, N, 1]
        Z_reflected = torch.zeros_like(Z)

        edge_send = self.edge_send
        edge_recv = self.edge_recv
        supp_e = self.supp_e.unsqueeze(0).unsqueeze(-1)           # [1, E, 1]
        N_h = self.N_h

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)                         # [B, N, D]

            # Edge-list gather: msg[b, e, d] = Z_fwd[b, edge_send[e], d]
            msg = Z_fwd.index_select(1, edge_send)                # [B, E, D]
            msg = msg * supp_e                                    # per-edge AH suppression

            # Reduce per receiver via scatter_add
            Z_struct = scatter_add(msg, edge_recv, dim=1, dim_size=N_h)  # [B, N, D]

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


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


def run_plain(make_fn, compile_mode=None, desc=""):
    m = make_fn().to(DEVICE).eval()
    if compile_mode is not None:
        print(f"  compiling mode={compile_mode}")
        m = torch.compile(m, mode=compile_mode, dynamic=False)
    x = torch.randn(args.bs, N_IN, device=DEVICE)
    reset_mem()
    r = bench(m, x, args.warmup, args.iters)
    r["peak_mib"] = peak_mem(); r["desc"] = desc
    del m; torch.cuda.empty_cache()
    return r


def run_cuda_graph(make_fn, desc=""):
    m = make_fn().to(DEVICE).eval()
    x = torch.randn(args.bs, N_IN, device=DEVICE)
    # Stream warm-up
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(3): _ = m(x)
    torch.cuda.current_stream().wait_stream(s)
    # Capture
    g = torch.cuda.CUDAGraph()
    static_x = torch.empty_like(x)
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_out = m(static_x)
    for _ in range(args.warmup):
        static_x.copy_(x); g.replay()
    sync()
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
        "peak_mib": mem, "desc": desc,
    }


def make_scatter(): return SGNNET_ScatterAH(make_ref())


def main():
    print(f"\nStep 832 — PyG scatter_add vs fancy-index routing")
    print(f"  N={N} D={D} K_hh={K_HH} K_iter={args.k_iter} bs={args.bs}")
    results = {"device": str(DEVICE), "bs": args.bs, "k_iter": args.k_iter,
               "has_pyg": HAS_PYG, "variants": {}}

    print("\n── V_ref: fancy-index eager fp32 ──")
    results["variants"]["V_ref"] = run_plain(make_ref, None, "fancy-index eager fp32")

    print("\n── V_ref_c: fancy-index max-autotune fp32 ──")
    try:
        results["variants"]["V_ref_c"] = run_plain(make_ref, "max-autotune", "fancy-index max-autotune fp32")
    except Exception as e:
        print(f"  V_ref_c failed: {e}")

    if HAS_PYG:
        print("\n── V_scat: scatter_add eager fp32 ──")
        try:
            results["variants"]["V_scat"] = run_plain(make_scatter, None, "scatter_add eager fp32")
        except Exception as e:
            print(f"  V_scat failed: {e}")

        print("\n── V_scat_c: scatter_add max-autotune fp32 ──")
        try:
            results["variants"]["V_scat_c"] = run_plain(make_scatter, "max-autotune", "scatter_add max-autotune fp32")
        except Exception as e:
            print(f"  V_scat_c failed: {e}")

        print("\n── V_scat_cg: scatter_add + CUDA Graph fp32 ──")
        try:
            results["variants"]["V_scat_cg"] = run_cuda_graph(make_scatter, "scatter_add CUDA Graph fp32")
        except Exception as e:
            print(f"  V_scat_cg failed: {e}")

    # Summary
    print("\n\n========== SUMMARY ==========")
    print(f"{'Variant':<18} {'median_ms':>10} {'p95_ms':>8} {'sps':>9} {'mem_MiB':>9}")
    base_ms = results["variants"]["V_ref"]["median_ms"]
    for k, r in results["variants"].items():
        if not r: continue
        print(f"{k:<18} {r['median_ms']:>10.3f} {r['p95_ms']:>8.3f} "
              f"{r['throughput_sps']:>9.0f} {r.get('peak_mib', -1):>9.1f}")

    print(f"\nSpeedups vs V_ref ({base_ms:.3f}ms):")
    for k, r in results["variants"].items():
        if not r: continue
        print(f"  {k:<18} {base_ms / r['median_ms']:>6.2f}×")

    best = min((k for k in results["variants"] if results["variants"][k]),
               key=lambda k: results["variants"][k]["median_ms"])
    print(f"\nBest variant: {best} @ {results['variants'][best]['median_ms']:.3f}ms")
    # vs step811 V2 max-autotune record: 0.280ms (K=5)
    print(f"vs step811 V2 max-autotune (0.280ms K=5): "
          f"{0.280 / results['variants'][best]['median_ms']:.2f}× speedup")
    # vs VGG_FC
    print(f"vs VGG_FC (1.570ms from step810): "
          f"{1.570 / results['variants'][best]['median_ms']:.2f}× faster")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
