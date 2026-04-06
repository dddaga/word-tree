"""
benchmarks/bench_sgnnet_routing.py

Benchmark the routing hot-path of SGNNET step86 Config A against mps_sparse.

7 strategies x 2 modes (inference, training) = 14 measurements:

  1. baseline/mps     current gather-sum on MPS (reference)
  2. baseline/cpu     same gather-sum on CPU
  3. auto/mps         StaticSparseMatrix(strategy="auto") on MPS — embedding_bag
  4. auto/cpu         same on CPU — falls back to scatter
  5. scipy/cpu        scipy CSR via Apple Accelerate on CPU (no MPS transfers)
  6. scipy_rcm/cpu    scipy CSR + Reverse Cuthill-McKee reordering (+43% cache)
  7. scipy/mps        scipy on CPU but Z starts/ends on MPS — measures transfer overhead

Does NOT modify any model or script files.
Read-only imports from src/ and mps_sparse/.

Usage:
    python benchmarks/bench_sgnnet_routing.py
    python benchmarks/bench_sgnnet_routing.py --warmup 5 --iters 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mps_sparse"))

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from mps_sparse import StaticSparseMatrix

# ---------------------------------------------------------------------------
# Step86 Config A parameters
# ---------------------------------------------------------------------------
N             = 4096
D             = 64
K_LOCAL       = 2
K_RAND        = 2
K_HH          = K_LOCAL + K_RAND     # 4
K_ITER        = 8
N_IN          = 25088
N_OUT         = 10
ALPHA_AHEBB   = 1.0
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
K_PHASE       = 8
BEAM_SIZE     = 16
GEO_GAMMA     = 0.5
SEED          = 42

B_INFER = [1, 8, 32, 128]
B_TRAIN = 128


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def measure_ms(fn: Callable, device: torch.device, warmup: int, iters: int) -> float:
    """Return median wall-clock time in ms over `iters` calls."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Routing kernels
# ---------------------------------------------------------------------------

def route_baseline(
    Z: torch.Tensor,
    conn_hh: torch.Tensor,     # [N, K_hh] on same device as Z
    theta_pos: torch.Tensor,   # [1, N, 1]
    supp_w: torch.Tensor,      # [1, N, K_hh, 1]
) -> torch.Tensor:
    """Reference: current indexed gather-sum routing (unmodified from model)."""
    Z_refl = torch.zeros_like(Z)
    for _ in range(K_ITER):
        Z_fwd    = F.relu(Z - theta_pos)
        Z_nb     = Z_fwd[:, conn_hh, :]               # [B, N, K_hh, D]
        Z_struct = (Z_nb * supp_w).sum(dim=2)          # [B, N, D]
        Z_refl   = ALPHA_REFLECT * Z_refl + (Z_fwd - Z)
        Z        = F.normalize((Z_struct + Z_refl).clamp(-10, 10), dim=-1)
    return Z


def route_spmm_native(
    Z: torch.Tensor,
    adj: StaticSparseMatrix,
    theta_pos: torch.Tensor,   # on same device as Z
) -> torch.Tensor:
    """SpMM routing where Z and adj live on the same device.

    Covers: auto/mps (Z on MPS, embedding_bag),
            auto/cpu (Z on CPU, scatter fallback),
            scipy/cpu (Z on CPU, scipy CSR).
    """
    B_, N_, D_ = Z.shape
    Z_refl = torch.zeros_like(Z)
    for _ in range(K_ITER):
        Z_fwd    = F.relu(Z - theta_pos)
        Z_flat   = Z_fwd.permute(1, 0, 2).reshape(N_, B_ * D_)   # [N, B*D]
        Z_out    = adj @ Z_flat                                    # [N, B*D]
        Z_struct = Z_out.reshape(N_, B_, D_).permute(1, 0, 2)     # [B, N, D]
        Z_refl   = ALPHA_REFLECT * Z_refl + (Z_fwd - Z)
        Z        = F.normalize((Z_struct + Z_refl).clamp(-10, 10), dim=-1)
    return Z


def route_spmm_mismatch(
    Z: torch.Tensor,           # lives on MPS
    adj_cpu: StaticSparseMatrix,   # scipy adj, always returns CPU tensor
    theta_pos_cpu: torch.Tensor,
    theta_pos_mps: torch.Tensor,
) -> torch.Tensor:
    """scipy/mps: intentional mismatch — scipy forces CPU, Z starts on MPS.

    Per routing step: MPS→CPU transfer + scipy SpMM (CPU) + CPU→MPS transfer.
    This quantifies the per-step transfer overhead.
    """
    mps_dev = Z.device
    B_, N_, D_ = Z.shape
    Z_refl = torch.zeros_like(Z)
    for _ in range(K_ITER):
        Z_fwd_mps = F.relu(Z - theta_pos_mps)
        # Transfer to CPU for scipy
        Z_cpu  = Z_fwd_mps.cpu()
        Z_flat = Z_cpu.permute(1, 0, 2).reshape(N_, B_ * D_)    # [N, B*D] on CPU
        Z_out  = adj_cpu @ Z_flat                                 # [N, B*D] on CPU (scipy)
        Z_struct = Z_out.reshape(N_, B_, D_).permute(1, 0, 2)    # [B, N, D] on CPU
        # Transfer back to MPS
        Z_struct = Z_struct.to(mps_dev)
        Z_refl   = ALPHA_REFLECT * Z_refl + (Z_fwd_mps - Z)
        Z        = F.normalize((Z_struct + Z_refl).clamp(-10, 10), dim=-1)
    return Z


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model() -> SGNNET_AntiHebbian:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=K_LOCAL, K_random=K_RAND,
        K_in=50, K_iter=K_ITER,
        n_groups=max(8, N // 8),
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=K_PHASE, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=BEAM_SIZE,
        geo_gamma=GEO_GAMMA, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

def build_adjacencies(
    conn_hh: torch.Tensor,       # [N, K_hh] CPU
    supp_w_flat: torch.Tensor,   # [N*K_hh] CPU — AntiHebb edge weights
    mps_ok: bool,
) -> dict:
    row_idx = torch.arange(N, dtype=torch.long).repeat_interleave(K_HH)
    col_idx = conn_hh.flatten().long()
    vals    = supp_w_flat.float()

    adjs: dict = {}

    # auto/cpu — strategy="auto", Z stays on CPU (scatter fallback)
    adjs["auto_cpu"] = StaticSparseMatrix(
        vals, row_idx, col_idx, (N, N), strategy="auto"
    )
    # scipy/cpu — Apple Accelerate sparse BLAS
    adjs["scipy_cpu"] = StaticSparseMatrix(
        vals, row_idx, col_idx, (N, N), strategy="scipy", reorder=False
    )
    # scipy_rcm/cpu — same + Reverse Cuthill-McKee cache reordering
    adjs["scipy_rcm"] = StaticSparseMatrix(
        vals, row_idx, col_idx, (N, N), strategy="scipy", reorder=True
    )

    if mps_ok:
        # auto/mps — embedding_bag on MPS
        adjs["auto_mps"] = StaticSparseMatrix(
            vals, row_idx, col_idx, (N, N), strategy="auto"
        )
        adjs["auto_mps"].to("mps")
        # scipy/mps — scipy (CPU path) but Z is on MPS → mismatch benchmark
        adjs["scipy_mps"] = StaticSparseMatrix(
            vals, row_idx, col_idx, (N, N), strategy="scipy", reorder=False
        )

    return adjs


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_infer_table(results: dict[str, dict[int, Optional[float]]], b_sizes: list[int]) -> None:
    # Determine reference: baseline/mps if available, else baseline/cpu
    ref128 = (results.get("baseline/mps") or {}).get(128)
    if ref128 is None:
        ref128 = (results.get("baseline/cpu") or {}).get(128)

    col_w = 10
    header = f"  {'Config':<22}" + "".join(f"  {'B='+str(b):>{col_w}}" for b in b_sizes)
    header += f"  {'ms/smp@128':>{col_w}}  {'spdup@128':>10}"
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)
    for name, bdict in results.items():
        row = f"  {name:<22}"
        for b in b_sizes:
            t = bdict.get(b)
            row += f"  {t:>{col_w}.2f}" if t is not None else f"  {'N/A':>{col_w}}"
        t128 = bdict.get(128)
        if t128 is not None:
            row += f"  {t128/128:>{col_w}.3f}"
            row += f"  {ref128/t128:>9.2f}x" if ref128 else f"  {'ref':>10}"
        else:
            row += f"  {'N/A':>{col_w}}  {'N/A':>10}"
        print(row)


def print_train_table(results: dict[str, Optional[float]]) -> None:
    ref_mps = results.get("baseline/mps")
    ref_cpu = results.get("baseline/cpu")
    ref     = ref_mps or ref_cpu

    header = (f"  {'Config':<22}  {'ms/batch':>10}  {'ms/sample':>10}"
              f"  {'spdup/mps':>10}  {'spdup/cpu':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, t in results.items():
        if t is None:
            print(f"  {name:<22}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")
            continue
        ms_smp = t / B_TRAIN
        spd_mps = f"{ref_mps/t:.2f}x" if ref_mps else "  ref"
        spd_cpu = f"{ref_cpu/t:.2f}x" if ref_cpu else "  ref"
        print(f"  {name:<22}  {t:>10.2f}  {ms_smp:>10.3f}  {spd_mps:>10}  {spd_cpu:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(warmup: int, iters: int) -> None:
    mps_ok = torch.backends.mps.is_available()
    cpu    = torch.device("cpu")
    mps    = torch.device("mps") if mps_ok else cpu

    print(f"PyTorch {torch.__version__}  |  MPS: {'available' if mps_ok else 'NOT available'}")
    print(f"step86 Config A  N={N}  K_hh={K_HH}  K_iter={K_ITER}  D={D}")
    print(f"Routing density: {K_HH}/{N} = {K_HH/N:.4%}  NNZ: {N*K_HH}")
    print(f"Warmup={warmup}  Iters={iters}")

    # ---- Build model, extract static tensors ----
    print("\nBuilding model...", end=" ", flush=True)
    model = build_model()
    model.eval()
    conn_hh_cpu  = model.m.base.conn_hh.cpu()           # [N, K_hh]
    theta_cpu    = model.m.theta.detach().cpu()          # [N]
    W_pos_cpu    = model.m.W_pos.detach().cpu()[:N]      # [N, D]
    print("done")

    # Pre-compute AntiHebb suppression weights from W_pos (static per model)
    W_n           = F.normalize(W_pos_cpu, dim=-1)
    pos_sim       = (W_n.unsqueeze(1) * W_n[conn_hh_cpu]).sum(-1)   # [N, K_hh]
    supp_w_2d     = (1.0 - ALPHA_AHEBB * pos_sim.clamp(min=0))      # [N, K_hh]
    supp_w_flat   = supp_w_2d.flatten()                               # [N*K_hh]
    supp_w_4d_cpu = supp_w_2d.unsqueeze(0).unsqueeze(-1)             # [1, N, K_hh, 1]
    theta_pos_cpu = theta_cpu.abs().unsqueeze(0).unsqueeze(-1)       # [1, N, 1]

    print(f"  supp_w: min={supp_w_flat.min():.3f}  max={supp_w_flat.max():.3f}"
          f"  mean={supp_w_flat.mean():.3f}  zeros={( supp_w_flat <= 0).sum().item()}/{len(supp_w_flat)}")

    # MPS copies
    if mps_ok:
        conn_hh_mps   = conn_hh_cpu.to(mps)
        supp_w_4d_mps = supp_w_4d_cpu.to(mps)
        theta_pos_mps = theta_pos_cpu.to(mps)
    else:
        conn_hh_mps = conn_hh_cpu
        supp_w_4d_mps = supp_w_4d_cpu
        theta_pos_mps = theta_pos_cpu

    # ---- Build adjacency matrices ----
    print("Building adjacencies...", end=" ", flush=True)
    adjs = build_adjacencies(conn_hh_cpu, supp_w_flat, mps_ok)
    nnz  = adjs["scipy_cpu"].nnz
    print(f"done  nnz={nnz}  density={adjs['scipy_cpu'].density:.4%}")
    if adjs["scipy_rcm"]._rcm_perm is not None:
        print(f"  RCM perm: {adjs['scipy_rcm']._rcm_perm.shape}  "
              f"(permutes indices — output differs numerically by design)")

    # ---- Correctness check ----
    print("\n=== CORRECTNESS (CPU, B=4) ===")
    torch.manual_seed(0)
    Z_test = torch.randn(4, N, D)
    with torch.no_grad():
        ref_out = route_baseline(Z_test, conn_hh_cpu, theta_pos_cpu, supp_w_4d_cpu)
        for label, adj in [("auto/cpu", adjs["auto_cpu"]), ("scipy/cpu", adjs["scipy_cpu"])]:
            out = route_spmm_native(Z_test, adj, theta_pos_cpu)
            err = (ref_out - out).abs().max().item()
            status = "PASS" if err < 1e-4 else f"FAIL (err={err:.2e})"
            print(f"  {label:<22}  max_err={err:.2e}  {status}")
        out_rcm = route_spmm_native(Z_test, adjs["scipy_rcm"], theta_pos_cpu)
        err_rcm = (ref_out - out_rcm).abs().max().item()
        print(f"  {'scipy_rcm/cpu':<22}  max_err={err_rcm:.2e}  EXPECTED (RCM permutes indices)")
    print()

    # ======================================================================
    # INFERENCE benchmark
    # ======================================================================
    print(f"=== INFERENCE LATENCY (no_grad, warmup={warmup}, iters={iters}) ===")
    infer_results: dict[str, dict[int, Optional[float]]] = {n: {} for n in [
        "baseline/mps", "baseline/cpu",
        "auto/mps", "auto/cpu",
        "scipy/cpu", "scipy_rcm/cpu",
        "scipy/mps",
    ]}

    for B in B_INFER:
        Z_cpu_b = torch.randn(B, N, D)
        Z_mps_b = Z_cpu_b.to(mps) if mps_ok else None

        runs: list[tuple[str, Callable, torch.device, bool]] = [
            ("baseline/mps",
             (lambda Z=Z_mps_b, ch=conn_hh_mps, tp=theta_pos_mps, sw=supp_w_4d_mps:
              route_baseline(Z, ch, tp, sw)),
             mps, mps_ok),

            ("baseline/cpu",
             (lambda Z=Z_cpu_b, ch=conn_hh_cpu, tp=theta_pos_cpu, sw=supp_w_4d_cpu:
              route_baseline(Z, ch, tp, sw)),
             cpu, True),

            ("auto/mps",
             (lambda Z=Z_mps_b, adj=adjs.get("auto_mps"), tp=theta_pos_mps:
              route_spmm_native(Z, adj, tp)),
             mps, mps_ok),

            ("auto/cpu",
             (lambda Z=Z_cpu_b, adj=adjs["auto_cpu"], tp=theta_pos_cpu:
              route_spmm_native(Z, adj, tp)),
             cpu, True),

            ("scipy/cpu",
             (lambda Z=Z_cpu_b, adj=adjs["scipy_cpu"], tp=theta_pos_cpu:
              route_spmm_native(Z, adj, tp)),
             cpu, True),

            ("scipy_rcm/cpu",
             (lambda Z=Z_cpu_b, adj=adjs["scipy_rcm"], tp=theta_pos_cpu:
              route_spmm_native(Z, adj, tp)),
             cpu, True),

            ("scipy/mps",
             (lambda Z=Z_mps_b, adj=adjs.get("scipy_mps"), tp_cpu=theta_pos_cpu, tp_mps=theta_pos_mps:
              route_spmm_mismatch(Z, adj, tp_cpu, tp_mps)),
             mps, mps_ok),
        ]

        for name, fn, device, available in runs:
            if not available or fn is None:
                infer_results[name][B] = None
                continue
            with torch.no_grad():
                t = measure_ms(fn, device, warmup, iters)
            infer_results[name][B] = t

    print_infer_table(infer_results, B_INFER)

    # ======================================================================
    # TRAINING benchmark  (forward + backward, B_TRAIN)
    # ======================================================================
    print(f"\n=== TRAINING THROUGHPUT (fwd+bwd, B={B_TRAIN}, warmup={warmup}, iters={iters}) ===")
    train_results: dict[str, Optional[float]] = {}

    B = B_TRAIN

    def make_train_baseline(z_dev, ch, tp, sw):
        def fn():
            Z = torch.randn(B, N, D, device=z_dev, requires_grad=True)
            route_baseline(Z, ch, tp, sw).sum().backward()
        return fn

    def make_train_spmm_native(z_dev, adj, tp):
        def fn():
            Z = torch.randn(B, N, D, device=z_dev, requires_grad=True)
            route_spmm_native(Z, adj, tp).sum().backward()
        return fn

    def make_train_spmm_mismatch():
        def fn():
            Z = torch.randn(B, N, D, device=mps, requires_grad=True)
            route_spmm_mismatch(
                Z, adjs["scipy_mps"], theta_pos_cpu, theta_pos_mps
            ).sum().backward()
        return fn

    train_runs: list[tuple[str, Optional[Callable], torch.device, bool]] = [
        ("baseline/mps",
         make_train_baseline(mps, conn_hh_mps, theta_pos_mps, supp_w_4d_mps) if mps_ok else None,
         mps, mps_ok),

        ("baseline/cpu",
         make_train_baseline(cpu, conn_hh_cpu, theta_pos_cpu, supp_w_4d_cpu),
         cpu, True),

        ("auto/mps",
         make_train_spmm_native(mps, adjs["auto_mps"], theta_pos_mps) if mps_ok else None,
         mps, mps_ok),

        ("auto/cpu",
         make_train_spmm_native(cpu, adjs["auto_cpu"], theta_pos_cpu),
         cpu, True),

        ("scipy/cpu",
         make_train_spmm_native(cpu, adjs["scipy_cpu"], theta_pos_cpu),
         cpu, True),

        ("scipy_rcm/cpu",
         make_train_spmm_native(cpu, adjs["scipy_rcm"], theta_pos_cpu),
         cpu, True),

        ("scipy/mps",
         make_train_spmm_mismatch() if mps_ok else None,
         mps, mps_ok),
    ]

    for name, fn, device, available in train_runs:
        if not available or fn is None:
            train_results[name] = None
        else:
            t = measure_ms(fn, device, warmup, iters)
            train_results[name] = t

    print_train_table(train_results)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGNNET step86 routing benchmark")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations before timing (default: 10)")
    parser.add_argument("--iters",  type=int, default=50,
                        help="Timing iterations for median (default: 50)")
    args = parser.parse_args()
    main(args.warmup, args.iters)
