"""Step 700: Multi-hop neighborhood precomputation — reduce K_iter latency.

CLAIM UNDER TEST
================
SGNNET's K_iter=5 sequential gather-scatter passes are the #1 latency bottleneck
at N=2048 on CUDA (each pass depends on the previous → cannot parallelize).

Hypothesis: precomputing k-hop neighborhoods into a wider conn_hh buffer lets us
reduce K_iter while preserving receptive-field coverage.

1 iter with a k-hop precomputed table ≈ k sequential iters with the original K_hh=2 table.

CONFIGS (N=2048, D=16, AH=1.0, reflect=0.5, turing=0.0, 50% data, 20ep Tier-0)
  Ref         — K_iter=5, K_hh=2 (standard efficiency config, step199)
  A_k3_kh6   — K_iter=3, K_hh_eff=6  (2-hop precomp)
  B_k2_kh14  — K_iter=2, K_hh_eff=14 (3-hop precomp)
  C_k1_kh32  — K_iter=1, K_hh_eff=32 (5-hop precomp)  — max speedup if it works
  D_k3_kh4   — K_iter=3, K_hh=4  (wider 1-hop, no precomp) — control: topology vs width
  E_k2_kh8   — K_iter=2, K_hh=8  (wider 1-hop, K_iter=2) — control: topology vs width

A/B/C use precomputed k-hop buffers; D/E use larger K_hh at 1-hop (standard construction).

FLOPs formula: N × K_iter × K_hh_eff × D × 2
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
import torch.utils.data

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset import make_loaders

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Step 700: Multi-hop gather — reduce K_iter via precomputed neighborhoods"
)
parser.add_argument("--device", default="auto",
                    help="Device: auto | cpu | mps | cuda")
parser.add_argument("--epochs", type=int, default=20,
                    help="Training epochs (default 20 for Tier-0)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (default: all)")
args = parser.parse_args()

if args.device == "auto":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(args.device)

EPOCHS = args.epochs
BATCH = 128
SEED = 42
DATA = "data/store.h5"

N = 2048
N_IN = 25088
N_OUT = 10
D = 16
K_HH_BASE = 2
K_IN = 25
ALPHA_AH = 1.0
ALPHA_REFLECT = 0.5
ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step700_multihop_gather.json"

# ---------------------------------------------------------------------------
# Config table
# Format: (K_iter, K_hh_eff, k_hops_precomp)
#   k_hops_precomp=1 → standard 1-hop conn_hh (no precomp)
#   k_hops_precomp>1 → build k-hop table from base K_hh=2 graph
#   K_hh_eff is the target width of the precomputed table
# ---------------------------------------------------------------------------
CONFIG_TABLE = {
    "Ref":       (5, 2,  1),   # baseline — standard efficiency config
    "A_k3_kh6":  (3, 6,  2),   # 2-hop precomp → ~3 iters coverage
    "B_k2_kh14": (2, 14, 3),   # 3-hop precomp → ~4 iters coverage
    "C_k1_kh32": (1, 32, 5),   # 5-hop precomp → single pass, max speedup
    "D_k3_kh4":  (3, 4,  1),   # control: wider 1-hop, no precomp
    "E_k2_kh8":  (2, 8,  1),   # control: wider 1-hop, K_iter=2
}

ALL_CONFIGS = list(CONFIG_TABLE.keys())


# ---------------------------------------------------------------------------
# Multi-hop connection precomputation
# ---------------------------------------------------------------------------

def build_khop_conn(conn_hh: torch.Tensor, k_hops: int, K_target: int) -> torch.Tensor:
    """Return [N, K_target] connections reachable within k_hops hops.

    Breadth-first: start with conn_hh (1-hop). For each additional hop,
    expand via conn_hh[conn_hh[...]], deduplicate per row, cap at K_target.
    If fewer than K_target unique k-hop neighbors, pad by repeating the last.

    NOTE: runs once at model init — O(N × K × ...) Python loop is acceptable.
    """
    N_nodes, K_1 = conn_hh.shape
    conn_cpu = conn_hh.cpu()
    current = conn_cpu.clone()  # [N, K_current] — starts as 1-hop

    for _hop in range(k_hops - 1):
        # Expand: gather 1-hop neighbors OF current neighbors
        expanded = conn_cpu[current]          # [N, K_current, K_1]
        expanded = expanded.reshape(N_nodes, -1)  # [N, K_current * K_1]
        # Concatenate current + expanded neighbors per row
        combined = torch.cat([current, expanded], dim=1)  # [N, K_current*(K_1+1)]

        new_rows: list[list[int]] = []
        for i in range(N_nodes):
            row = combined[i].tolist()
            seen: set[int] = set()
            unique: list[int] = []
            for x in row:
                xi = int(x)
                if xi != i and xi not in seen:
                    seen.add(xi)
                    unique.append(xi)
                if len(unique) >= K_target:
                    break
            # Pad if fewer than K_target unique neighbors found
            while len(unique) < K_target:
                unique.append(unique[-1] if unique else (i + 1) % N_nodes)
            new_rows.append(unique[:K_target])

        current = torch.tensor(new_rows, dtype=torch.long)

    # At this point current may have more or fewer columns than K_target;
    # trim or pad to exactly K_target
    cur_K = current.shape[1]
    if cur_K >= K_target:
        return current[:, :K_target]
    else:
        # Should not happen due to padding above, but be safe
        pad_col = current[:, -1:]  # repeat last column
        pad = pad_col.expand(N_nodes, K_target - cur_K)
        return torch.cat([current, pad], dim=1)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build(K_iter: int, K_hh_eff: int, k_hops: int) -> SGNNET_AntiHebbian:
    """Build model with given K_iter / K_hh_eff / precomputation depth.

    For k_hops > 1: build base with K_hh=K_HH_BASE=2, then replace conn_hh
    with the precomputed k-hop table (width K_hh_eff).

    For k_hops == 1: build base with K_hh=K_hh_eff directly (wider 1-hop).
    """
    torch.manual_seed(SEED)

    if k_hops == 1:
        # Standard construction — use K_hh_eff directly
        K_r = max(1, K_hh_eff // 4)
        K_l = K_hh_eff - K_r
        ng = max(8, N // 8)
        base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
            n_groups=ng, norm_mode="l2", encoding_mode="fourier",
        )
    else:
        # Build base with K_HH_BASE=2, then expand conn_hh to k_hops depth
        K_r = max(1, K_HH_BASE // 4)
        K_l = K_HH_BASE - K_r
        ng = max(8, N // 8)
        base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
            n_groups=ng, norm_mode="l2", encoding_mode="fourier",
        )
        # Precompute k-hop neighborhood table
        print(f"  [build_khop_conn] k_hops={k_hops}, K_target={K_hh_eff} — computing...",
              flush=True)
        t_khop = time.time()
        new_conn = build_khop_conn(base.conn_hh, k_hops=k_hops, K_target=K_hh_eff)
        elapsed_khop = time.time() - t_khop
        print(f"  [build_khop_conn] done in {elapsed_khop:.1f}s — shape={list(new_conn.shape)}",
              flush=True)

        # Replace conn_hh buffer in-place so the forward loop reads the new table
        # conn_hh is registered as a buffer (not a parameter), so we delete + re-register
        delattr(base, "conn_hh")
        base.register_buffer("conn_hh", new_conn)

    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AH, variant="wpos")


# ---------------------------------------------------------------------------
# FLOPs estimate
# ---------------------------------------------------------------------------

def estimate_flops(K_iter: int, K_hh_eff: int) -> int:
    """Approximate FLOPs: N × K_iter × K_hh_eff × D × 2."""
    return N * K_iter * K_hh_eff * D * 2


# ---------------------------------------------------------------------------
# Forward wall-time measurement
# ---------------------------------------------------------------------------

def measure_forward_ms(model: nn.Module, val_loader,
                       n_warmup: int = 20, n_timed: int = 50) -> float:
    """Return mean forward-pass time in ms over a single batch."""
    model.eval()
    batch = next(iter(val_loader))
    x_batch = batch[0].to(DEVICE)  # (features, soft_labels, labels)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elif DEVICE.type == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_timed):
            _ = model(x_batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elif DEVICE.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

    return (elapsed / n_timed) * 1000.0


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def run_training(model: nn.Module, tr, va, n_epochs: int, label: str) -> dict:
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}", flush=True)

    kw = trainer_kwargs(N, n_epochs=n_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)

    def _log(m):
        ep = m["epoch"] + 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

    t0 = time.time()
    history = trainer.train(n_epochs=n_epochs, log_fn=_log)
    elapsed = time.time() - t0

    top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1h)
    bep = int(np.argmax(top1h)) + 1

    return {
        "top1_best": best,
        "top1_last": top1h[-1],
        "best_epoch": bep,
        "top1_history": top1h,
        "elapsed_s": round(elapsed, 1),
        "n_params": n_p,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_keys = ALL_CONFIGS if not args.configs else [k.strip() for k in args.configs.split(",")]
    unknown = [k for k in run_keys if k not in CONFIG_TABLE]
    if unknown:
        parser.error(f"Unknown config keys: {unknown}. Valid: {ALL_CONFIGS}")

    print(f"\n{'='*70}")
    print(f"Step 700 — Multi-hop Gather (N={N}, D={D}, K_hh_base={K_HH_BASE})")
    print(f"Device: {DEVICE}  |  Epochs: {EPOCHS}  |  Configs: {run_keys}")
    print(f"Goal: reduce K_iter latency via precomputed k-hop neighborhoods")
    print(f"{'='*70}")
    print(f"\n  {'Config':<16}  {'K_iter':>6}  {'K_hh_eff':>8}  {'k_hops':>6}  {'FLOPs':>12}")
    for k in run_keys:
        ki, kh, kp = CONFIG_TABLE[k]
        fl = estimate_flops(ki, kh)
        print(f"  {k:<16}  {ki:>6}  {kh:>8}  {kp:>6}  {fl:>12,}")

    # Data loaders
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results: dict = {}
    timings: dict = {}

    for key in run_keys:
        K_iter, K_hh_eff, k_hops = CONFIG_TABLE[key]
        flops = estimate_flops(K_iter, K_hh_eff)

        print(f"\n{'─'*60}")
        print(f"Config {key}: K_iter={K_iter}, K_hh_eff={K_hh_eff}, "
              f"k_hops={k_hops}, FLOPs={flops:,}")
        print(f"{'─'*60}")

        model = build(K_iter, K_hh_eff, k_hops).to(DEVICE)

        # Measure forward latency before training
        fwd_ms = measure_forward_ms(model, va)
        timings[key] = round(fwd_ms, 3)
        print(f"  forward_ms (pre-train) = {fwd_ms:.3f} ms", flush=True)

        # Verify conn_hh shape on the built model
        actual_conn_shape = list(model.m.base.conn_hh.shape)
        print(f"  conn_hh shape = {actual_conn_shape}", flush=True)

        r = run_training(model, tr, va, EPOCHS, key)
        r.update({
            "K_iter": K_iter,
            "K_hh_eff": K_hh_eff,
            "k_hops_precomp": k_hops,
            "flops_estimate": flops,
            "forward_ms_pretrain": round(fwd_ms, 3),
            "conn_hh_shape": actual_conn_shape,
        })
        results[key] = r
        print(f"  → best={r['top1_best']:.4f} @ ep{r['best_epoch']}  ({r['elapsed_s']:.0f}s)")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output = {
        "step": 700,
        "config": {
            "N": N, "D": D, "K_hh_base": K_HH_BASE, "K_in": K_IN,
            "alpha_ah": ALPHA_AH, "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing": ALPHA_TURING, "epochs": EPOCHS,
            "data_fraction": 0.5, "batch_size": BATCH, "seed": SEED,
            "device": str(DEVICE),
        },
        "results": results,
        "timings_ms": timings,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    ref = results.get("Ref", {})
    ref_best = ref.get("top1_best", 0.0)
    ref_ms = timings.get("Ref", None)
    ref_flops = estimate_flops(*CONFIG_TABLE["Ref"][:2])

    print(f"\n{'='*70}")
    print(f"STEP 700 SUMMARY — Multi-hop Gather")
    print(f"{'='*70}")
    print(f"  {'Config':<16}  {'acc':>8}  {'Δ vs Ref':>10}  {'fwd_ms':>8}  "
          f"{'speedup':>8}  {'FLOPs':>12}  {'FLOP_ratio':>10}")
    for key in [k for k in ALL_CONFIGS if k in results]:
        r = results[key]
        delta = f"{r['top1_best'] - ref_best:+.4f}" if key != "Ref" else "    —"
        ms = timings.get(key, None)
        ms_str = f"{ms:.3f}" if ms else "  —"
        spd = f"{ref_ms / ms:.2f}x" if (ref_ms and ms and ms > 0) else "  —"
        fl = r["flops_estimate"]
        fl_ratio = f"{fl / ref_flops:.2f}x" if ref_flops else "  —"
        print(f"  {key:<16}  {r['top1_best']:>8.4f}  {delta:>10}  "
              f"{ms_str:>8}  {spd:>8}  {fl:>12,}  {fl_ratio:>10}")

    print(f"\n  Ref FLOPs = {ref_flops:,}")
    print(f"  Ref forward_ms = {ref_ms:.3f}" if ref_ms else "  Ref not run")

    print(f"\n  Interpretation guide:")
    print(f"  - Multi-hop wins if acc close to Ref AND forward_ms < Ref")
    print(f"  - Control configs (D/E) isolate width effect from topology effect")
    print(f"  - C_k1_kh32 is the strong-form test: 1 pass vs 5 passes")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
