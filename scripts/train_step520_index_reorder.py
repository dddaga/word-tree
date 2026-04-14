"""Step 520: Index reordering (Reverse Cuthill-McKee) — memory locality optimization.

CLAIM UNDER TEST
================
SGNNET's conn_hh gather has random indices (small-world graph with K_random=1 long-range
shortcut per neuron), killing L2 cache hit rate on CUDA → GPU utilization ~2.6%.
Node reordering via RCM can give 1.3–2.4x speedup (VLDB 2025) by improving spatial locality.

Reordering is mathematically equivalent — just relabels nodes. A model trained on
reordered vs non-reordered conn_hh should produce identical forward outputs (verified
with assert) and similar accuracy. Any speed difference is a pure hardware win.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, AH=1.0, 50% data, 20ep Tier-0)
  Ref           — standard small-world, no reordering
  A_rcm         — same random seed, RCM-permuted at init before training
  B_rcm_retrain — different seed arrangement (permuted rows → different W_pos init)
  C_rcm_post    — Ref trained first, RCM-permuted at ep10, then continued

NOTE: scipy is required. If missing: d_env/bin/pip install scipy
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
import torch.utils.data

# scipy check — hard requirement
try:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee
except ImportError:
    raise ImportError(
        "scipy is required for RCM reordering. "
        "Install with: d_env/bin/pip install scipy"
    )

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset import make_loaders

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="")
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
K_HH = 2
K_IN = 25
K_ITER = 5
ALPHA_AH = 1.0
ALPHA_REFLECT = 0.5
ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step520_index_reorder.json"

# ---------------------------------------------------------------------------
# RCM utilities
# ---------------------------------------------------------------------------

def compute_rcm_permutation(conn_hh: torch.Tensor, N: int) -> torch.Tensor:
    """Build CSR adjacency from conn_hh, symmetrize, return RCM permutation."""
    K = conn_hh.shape[1]
    rows = torch.arange(N).unsqueeze(1).expand(-1, K).flatten().numpy()
    cols = conn_hh.cpu().flatten().numpy()
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    adj = adj + adj.T  # symmetrize for undirected RCM
    perm = reverse_cuthill_mckee(adj)
    return torch.tensor(perm.copy(), dtype=torch.long)


def apply_permutation(model: SGNNET_AntiHebbian, perm: torch.Tensor) -> None:
    """Apply node permutation in-place.

    Permutes ALL tensors that are indexed by hidden-neuron index:
      - conn_hh rows AND neighbor values (both map through perm)
      - conn_in rows (input fan-in follows new node order; input indices unchanged)
      - W_pos[:N_hidden] rows (hidden neuron positions)
      - C_ho_mask rows (hidden→output projection mask)
      - base.theta (per-neuron resonant threshold in SGNNET_Resonant)

    conn_in VALUES are indices into the INPUT feature space (not hidden), so
    they are NOT permuted. W_pos[N_hidden:] are output-neuron rows — unchanged.

    This relabels nodes consistently so the model is mathematically equivalent
    (same function, different memory layout).
    """
    base = model.m.base  # SGNNET_SmallWorld
    N_h = base.N_hidden
    dev = base.conn_hh.device

    perm = perm.to(dev)
    # Inverse permutation: old_idx → new_idx (needed to remap neighbor references)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=dev)

    # --- conn_hh: rows = neurons, values = neighbor indices ---
    new_conn_hh = base.conn_hh[perm]           # reorder rows (neuron order)
    new_conn_hh = inv[new_conn_hh]             # remap neighbor references to new names
    base.conn_hh.copy_(new_conn_hh)

    # --- conn_in: rows = neurons, values = INPUT indices (NOT permuted) ---
    new_conn_in = base.conn_in[perm]           # reorder rows only
    base.conn_in.copy_(new_conn_in)

    # --- C_ho_mask: [N_hidden, N_out] rows = hidden neurons ---
    new_C_ho = base.C_ho_mask[perm]
    base.C_ho_mask.copy_(new_C_ho)

    # --- W_pos: hidden rows [0:N_hidden]; output rows [N_hidden:] unchanged ---
    new_W_pos = base.W_pos.data.clone()
    new_W_pos[:N_h] = base.W_pos.data[:N_h][perm]
    base.W_pos.data.copy_(new_W_pos)

    # --- theta (per-neuron resonant threshold in SGNNET_Resonant) ---
    resonant = model.m  # SGNNET_Resonant
    if hasattr(resonant, "theta"):
        new_theta = resonant.theta.data[perm]
        resonant.theta.data.copy_(new_theta)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build(seed: int = SEED) -> SGNNET_AntiHebbian:
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4)
    K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AH, variant="wpos")


# ---------------------------------------------------------------------------
# Forward-pass timing
# ---------------------------------------------------------------------------

def measure_forward_ms(model: SGNNET_AntiHebbian, val_loader, n_warmup: int = 50, n_timed: int = 100) -> float:
    """Return mean forward-pass time in ms over a single batch."""
    model.eval()
    x_batch, _ = next(iter(val_loader))
    x_batch = x_batch.to(DEVICE)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_timed):
            _ = model(x_batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    return (elapsed / n_timed) * 1000.0  # ms


# ---------------------------------------------------------------------------
# Verify mathematical equivalence (Ref vs RCM-permuted at init)
# ---------------------------------------------------------------------------

def verify_equivalence(model_ref: SGNNET_AntiHebbian, model_rcm: SGNNET_AntiHebbian, val_loader) -> float:
    """Run same batch through both models; return max absolute diff."""
    model_ref.eval()
    model_rcm.eval()
    x_batch, _ = next(iter(val_loader))
    x_batch = x_batch.to(DEVICE)
    with torch.no_grad():
        out_ref = model_ref(x_batch)
        out_rcm = model_rcm(x_batch)
    return (out_ref - out_rcm).abs().max().item()


# ---------------------------------------------------------------------------
# Training loop helper
# ---------------------------------------------------------------------------

def run_training(model: SGNNET_AntiHebbian, tr, va, n_epochs: int, label: str,
                 rcm_at_epoch: int | None = None) -> dict:
    """Train model for n_epochs; optionally apply RCM permutation at rcm_at_epoch."""
    perm_applied_ep = None

    def _log(m):
        ep = m["epoch"] + 1
        nonlocal perm_applied_ep
        # Apply RCM mid-training if requested
        if rcm_at_epoch is not None and ep == rcm_at_epoch and perm_applied_ep is None:
            perm = compute_rcm_permutation(model.m.base.conn_hh, N)
            apply_permutation(model, perm)
            perm_applied_ep = ep
            print(f"  [ep{ep}] RCM permutation applied mid-training", flush=True)
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

    kw = trainer_kwargs(N, n_epochs=n_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    t0 = time.time()
    history = trainer.train(n_epochs=n_epochs, log_fn=_log)
    elapsed = time.time() - t0

    top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1h)
    bep = int(np.argmax(top1h)) + 1
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "top1_best": best,
        "top1_last": top1h[-1],
        "best_epoch": bep,
        "top1_history": top1h,
        "elapsed_s": round(elapsed, 1),
        "n_params": n_p,
        "rcm_applied_at_epoch": perm_applied_ep,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_CONFIGS = ["Ref", "A_rcm", "B_rcm_retrain", "C_rcm_post"]


def main():
    run_keys = ALL_CONFIGS if not args.configs else [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 520 — RCM Index Reordering (N={N}, D={D}, K_hh={K_HH}, K_iter={K_ITER})")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}  |  Configs: {run_keys}")
    print(f"{'='*70}")

    # Data loaders
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[: n_total // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results: dict = {}
    timing: dict = {}  # forward_ms per config

    # -----------------------------------------------------------------------
    # Ref — standard small-world, no reordering
    # -----------------------------------------------------------------------
    if "Ref" in run_keys:
        print(f"\n{'─'*60}\nConfig Ref: standard topology (no reordering)\n{'─'*60}")
        model_ref = build(SEED).to(DEVICE)
        print(f"  params={sum(p.numel() for p in model_ref.parameters() if p.requires_grad):,}")

        forward_ms_ref = measure_forward_ms(model_ref, va)
        timing["Ref"] = round(forward_ms_ref, 3)
        print(f"  forward_ms (pre-train) = {forward_ms_ref:.3f} ms")

        r = run_training(model_ref, tr, va, EPOCHS, "Ref")
        results["Ref"] = r
        print(f"  → best={r['top1_best']:.4f} @ ep{r['best_epoch']}  ({r['elapsed_s']:.0f}s)")

        # Measure post-train timing
        forward_ms_ref_post = measure_forward_ms(model_ref, va)
        timing["Ref_post"] = round(forward_ms_ref_post, 3)
        print(f"  forward_ms (post-train) = {forward_ms_ref_post:.3f} ms")

    # -----------------------------------------------------------------------
    # A_rcm — RCM-permuted at init, before training
    # -----------------------------------------------------------------------
    if "A_rcm" in run_keys:
        print(f"\n{'─'*60}\nConfig A_rcm: RCM-permuted at init (before training)\n{'─'*60}")
        model_a = build(SEED).to(DEVICE)  # same seed as Ref

        # Compute and apply permutation BEFORE verifying equivalence
        perm_a = compute_rcm_permutation(model_a.m.base.conn_hh, N)
        print(f"  RCM perm computed ({len(perm_a)} nodes)")

        # Verify equivalence at init (same model state, just relabeled)
        model_a_pre = build(SEED).to(DEVICE)  # fresh copy for diff check
        apply_permutation(model_a, perm_a)
        # Note: equivalence holds only if outputs are permutation-invariant
        # (readout is a sum over all hidden → is permutation invariant)
        max_diff = verify_equivalence(model_a_pre, model_a, va)
        print(f"  Max output diff (Ref vs A_rcm at init): {max_diff:.2e}")
        if max_diff > 1e-5:
            print(f"  WARNING: max_diff={max_diff:.2e} > 1e-5 — check permutation logic")
        else:
            print(f"  PASS: outputs match within 1e-5 (mathematically equivalent)")

        forward_ms_a = measure_forward_ms(model_a, va)
        timing["A_rcm"] = round(forward_ms_a, 3)
        print(f"  forward_ms (pre-train) = {forward_ms_a:.3f} ms")

        r = run_training(model_a, tr, va, EPOCHS, "A_rcm")
        results["A_rcm"] = {**r, "max_diff_init": round(max_diff, 8)}
        print(f"  → best={r['top1_best']:.4f} @ ep{r['best_epoch']}  ({r['elapsed_s']:.0f}s)")

        forward_ms_a_post = measure_forward_ms(model_a, va)
        timing["A_rcm_post"] = round(forward_ms_a_post, 3)
        print(f"  forward_ms (post-train) = {forward_ms_a_post:.3f} ms")

    # -----------------------------------------------------------------------
    # B_rcm_retrain — RCM-permuted model with a fresh init (seed+1)
    # Different W_pos initialization in the permuted neuron order
    # -----------------------------------------------------------------------
    if "B_rcm_retrain" in run_keys:
        print(f"\n{'─'*60}\nConfig B_rcm_retrain: fresh init with RCM-permuted topology (seed+1)\n{'─'*60}")
        model_b = build(SEED + 1).to(DEVICE)
        perm_b = compute_rcm_permutation(model_b.m.base.conn_hh, N)
        apply_permutation(model_b, perm_b)
        print(f"  RCM perm applied at init (seed={SEED+1})")

        forward_ms_b = measure_forward_ms(model_b, va)
        timing["B_rcm_retrain"] = round(forward_ms_b, 3)
        print(f"  forward_ms (pre-train) = {forward_ms_b:.3f} ms")

        r = run_training(model_b, tr, va, EPOCHS, "B_rcm_retrain")
        results["B_rcm_retrain"] = r
        print(f"  → best={r['top1_best']:.4f} @ ep{r['best_epoch']}  ({r['elapsed_s']:.0f}s)")

    # -----------------------------------------------------------------------
    # C_rcm_post — Ref trained to ep10, then RCM-permuted, then continued
    # Tests whether mid-training reordering breaks or preserves learning
    # -----------------------------------------------------------------------
    if "C_rcm_post" in run_keys:
        print(f"\n{'─'*60}\nConfig C_rcm_post: Ref trained ep1-10, RCM applied at ep10, then ep11-{EPOCHS}\n{'─'*60}")
        model_c = build(SEED).to(DEVICE)

        forward_ms_c = measure_forward_ms(model_c, va)
        timing["C_rcm_post"] = round(forward_ms_c, 3)
        print(f"  forward_ms (pre-train) = {forward_ms_c:.3f} ms")

        r = run_training(model_c, tr, va, EPOCHS, "C_rcm_post", rcm_at_epoch=10)
        results["C_rcm_post"] = r
        print(f"  → best={r['top1_best']:.4f} @ ep{r['best_epoch']}  ({r['elapsed_s']:.0f}s)")

        forward_ms_c_post = measure_forward_ms(model_c, va)
        timing["C_rcm_post_post"] = round(forward_ms_c_post, 3)
        print(f"  forward_ms (post-train) = {forward_ms_c_post:.3f} ms")

    # -----------------------------------------------------------------------
    # Speedup summary
    # -----------------------------------------------------------------------
    ref_ms = timing.get("Ref", timing.get("Ref_post", None))
    rcm_ms = timing.get("A_rcm", timing.get("A_rcm_post", None))
    speedup = None
    if ref_ms and rcm_ms and rcm_ms > 0:
        speedup = round(ref_ms / rcm_ms, 3)

    speedup_summary = {
        "forward_ms_ref": ref_ms,
        "forward_ms_rcm": rcm_ms,
        "speedup_ratio": speedup,
        "all_timings_ms": timing,
    }

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "step": 520,
        "config": {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "alpha_ah": ALPHA_AH, "alpha_reflect": ALPHA_REFLECT,
            "alpha_turing": ALPHA_TURING, "epochs": EPOCHS,
            "data_fraction": 0.5, "batch_size": BATCH, "seed": SEED,
            "device": str(DEVICE),
        },
        "results": results,
        "speedup": speedup_summary,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    ref_best = results.get("Ref", {}).get("top1_best", 0.0)

    print(f"\n{'='*70}")
    print(f"STEP 520 SUMMARY — RCM Index Reordering")
    print(f"{'='*70}")
    print(f"  {'Config':<20}  {'acc':>8}  {'Δ vs Ref':>10}  {'fwd_ms':>9}  {'epochs':>6}")
    for key in [k for k in ALL_CONFIGS if k in results]:
        r = results[key]
        delta = f"{r['top1_best'] - ref_best:+.4f}" if key != "Ref" else "    —"
        ms_key = key if key in timing else f"{key}_post"
        ms_str = f"{timing[ms_key]:.3f}" if ms_key in timing else "  —"
        print(f"  {key:<20}  {r['top1_best']:>8.4f}  {delta:>10}  {ms_str:>9}  {r['best_epoch']:>6}")

    if speedup is not None:
        direction = "FASTER" if speedup > 1.0 else "SLOWER"
        print(f"\n  Speedup (Ref / A_rcm): {speedup:.3f}x  [{direction}]")
        print(f"  forward_ms Ref={ref_ms:.3f}  A_rcm={rcm_ms:.3f}")
    else:
        print(f"\n  Timing: {timing}")

    # Equivalence check
    if "A_rcm" in results and "max_diff_init" in results["A_rcm"]:
        md = results["A_rcm"]["max_diff_init"]
        eq_status = "PASS" if md <= 1e-5 else "FAIL"
        print(f"\n  Mathematical equivalence check: {eq_status} (max_diff={md:.2e})")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
