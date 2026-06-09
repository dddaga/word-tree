"""Step 969: Gradient-threshold firing — two reset policies (T0 scout).

MECHANISM: "Fire when ready" gradient accumulation.
Instead of top-K per batch (step968), accumulate gradients across batches.
When a param's accumulated |grad| exceeds threshold τ → fire update, then reset.

Two reset policies:
  SELECTIVE — zero only the fired params' grad. Rest keep accumulating.
  FULL      — zero ALL params' grad after any fire. Fresh start each time.

Physics analogy:
  SELECTIVE = leaky-integrate-and-fire (LIF): each neuron has its own
              charge bucket; fires independently when it reaches threshold.
  FULL      = synchronous reset: one fire event resets the entire network.

Hypothesis:
  SELECTIVE enables asynchronous, fine-grained updates (different params
  update at different rates based on gradient pressure).
  FULL enforces synchrony — prevents stale gradient artifacts from old
  mini-batches mixing with new gradients in different params.

THRESHOLD CALIBRATION:
  Auto-calibrated from first N_CALIB=10 batches.
  Collect all |grad| values → compute percentile thresholds:
    τ_lo = percentile(|grads|, 50)   → ~50% params fire per batch initially
    τ_hi = percentile(|grads|, 90)   → ~10% params fire per batch initially

CONFIGS (N=2048, D=16, K_in=25, Imagenette, 50% aug, T0=20ep)
  Ref              — AdamW fp32, standard
  GTF_sel_lo       — selective, τ=τ_lo (high fire rate)
  GTF_sel_hi       — selective, τ=τ_hi (low fire rate)
  GTF_full_lo      — full reset, τ=τ_lo
  GTF_full_hi      — full reset, τ=τ_hi
  GTF_sel_adaptive — selective, τ updated each epoch from current grad stats

SECONDARY METRICS:
  avg_fire_pct — average % of params updated per batch (fire rate)
  This tells us how sparse the effective updates are.

ADVANCE CRITERION (T0):
  Any config ≥ Ref − 2pp → viable, advance to T1 with longer accumulation
  Goal is not accuracy at T0 (only 20ep) but convergence trajectory shape.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.sgnnet.model_smallworld import SGNNET_SmallWorld

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_aug.h5")
parser.add_argument("--configs", default="Ref,GTF_sel_lo,GTF_sel_hi,GTF_full_lo,GTF_full_hi,GTF_sel_adaptive")
parser.add_argument("--lr",      type=float, default=3e-3)
parser.add_argument("--frac",    type=float, default=0.5, help="Training data fraction")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED    = args.seed
EPOCHS  = 20
N       = 2048
D       = 16
N_IN    = 25088
N_OUT   = 10
K_IN    = 25
K_ITER  = 5
BATCH   = 64
REF_ACC = 0.9552  # step199 T2 baseline
N_CALIB = 10      # batches for threshold calibration

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step969_threshold_grad_seed{SEED}__{SLOT}.json"


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    torch.manual_seed(SEED)
    return SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=1, K_random=1,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    ).to(DEVICE)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)
    with h5py.File(data_path, "r") as f:
        tr_x = torch.tensor(f["train/features"][:], dtype=torch.float32)
        tr_y = torch.tensor(f["train/labels"][:],   dtype=torch.long)
        va_x = torch.tensor(f["val/features"][:],   dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:],     dtype=torch.long)
    # Subsample training data
    n_tr = int(len(tr_x) * args.frac)
    idx  = torch.randperm(len(tr_x), generator=torch.Generator().manual_seed(SEED))[:n_tr]
    tr_x, tr_y = tr_x[idx], tr_y[idx]
    tr = DataLoader(TensorDataset(tr_x, tr_y), batch_size=BATCH, shuffle=True,
                    num_workers=10, pin_memory=True)
    va = DataLoader(TensorDataset(va_x, va_y), batch_size=256, shuffle=False,
                    num_workers=10)
    print(f"  Data: {len(tr_x)} train ({args.frac:.0%}) / {len(va_x)} val / {len(tr)} batches/ep")
    return tr, va


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, va) -> float:
    model.eval()
    correct = total = 0
    for x, y in va:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── Threshold calibration ─────────────────────────────────────────────────────
def calibrate_threshold(model, tr, n_batches: int) -> tuple[float, float]:
    """
    Run n_batches forward+backward, collect per-parameter max(|grad|).
    Using per-param max avoids the sparsity trap: SGNNET's K_in gather leaves
    >99% of W_seed elements zero-grad, making element-wise p50 = 0.
    Returns (tau_lo=p50, tau_hi=p90) of per-parameter grad-max values.
    """
    model.train()
    model.zero_grad()
    it = iter(tr)
    for _ in range(n_batches):
        try:
            bx, by = next(it)
        except StopIteration:
            break
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        loss = F.cross_entropy(model(bx), by)
        loss.backward()  # accumulate across N_CALIB batches

    # Collect per-parameter max |grad| (not per-element)
    param_maxgrads = []
    for p in model.parameters():
        if p.grad is not None:
            param_maxgrads.append(p.grad.abs().max().item())
    model.zero_grad()

    if not param_maxgrads:
        return 1e-4, 1e-3

    arr = np.array(param_maxgrads)
    tau_lo = float(np.percentile(arr, 50))
    tau_hi = float(np.percentile(arr, 90))
    print(f"  Threshold calibration over {n_batches} batches ({len(arr)} params):")
    print(f"    τ_lo (p50 per-param max|g|) = {tau_lo:.2e}")
    print(f"    τ_hi (p90 per-param max|g|) = {tau_hi:.2e}", flush=True)
    return tau_lo, tau_hi


def current_tau(model) -> float:
    """p50 of per-parameter max|grad| among params with non-zero grad."""
    vals = [p.grad.abs().max().item() for p in model.parameters()
            if p.grad is not None and p.grad.abs().max().item() > 0]
    if not vals:
        return 1e-4
    return float(np.percentile(vals, 50))


# ── Threshold firing update ────────────────────────────────────────────────────
def threshold_fire(model: nn.Module, tau: float, lr: float, reset_mode: str) -> int:
    """
    Apply update to all params with |grad|.max() > tau.
    reset_mode: "selective" → zero only fired params' grad
                "full"      → zero all params' grad after any fire
    Returns: number of params that fired.
    """
    fired: list[nn.Parameter] = []
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().max().item() > tau:
            fired.append(p)

    if fired:
        with torch.no_grad():
            for p in fired:
                p.data.sub_(lr * p.grad)

        if reset_mode == "selective":
            for p in fired:
                p.grad.zero_()
        else:  # full
            model.zero_grad()

    return len(fired)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_config(key: str, cfg: dict, tr, va, tau_lo: float, tau_hi: float, results: dict):
    mode       = cfg["mode"]        # "ref", "selective", "full"
    tau_key    = cfg["tau"]         # "lo", "hi", "adaptive", or None
    lr         = cfg["lr"]
    adaptive   = (tau_key == "adaptive")

    if tau_key == "lo":     tau = tau_lo
    elif tau_key == "hi":   tau = tau_hi
    else:                   tau = tau_lo   # adaptive starts at lo

    model = build_model()
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.zero_grad()

    # Standard AdamW for Ref only
    if mode == "ref":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    hist_acc       = []
    hist_ep        = []
    hist_fire_pct  = []
    best_acc       = 0.0
    t_start        = time.time()
    n_param_tensors = sum(1 for p in model.parameters() if p.requires_grad)

    print(f"\n{'─'*60}")
    print(f"{key}: mode={mode}  τ_key={tau_key}  τ={tau:.2e}  lr={lr}  params={n_p:,}", flush=True)

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_fired_total = 0
        ep_batches     = 0

        for bx, by in tr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)

            if mode == "ref":
                opt.zero_grad()
                loss = F.cross_entropy(model(bx), by)
                loss.backward()
                opt.step()
            else:
                # Gradient accumulates across batches — do NOT zero_grad here
                loss = F.cross_entropy(model(bx), by)
                loss.backward()

                if adaptive:
                    tau = current_tau(model)

                n_fired = threshold_fire(model, tau, lr, mode)
                ep_fired_total += n_fired
                ep_batches     += 1

        # Compute avg fire pct for this epoch (fraction of param tensors, not elements)
        if mode != "ref" and ep_batches > 0:
            avg_fire_pct = 100.0 * ep_fired_total / (ep_batches * n_param_tensors)
        else:
            avg_fire_pct = 100.0  # ref updates all params every batch

        acc     = evaluate(model, va)
        elapsed = time.time() - t_start
        if acc > best_acc:
            best_acc = acc

        hist_acc.append(round(acc, 4))
        hist_ep.append(ep)
        hist_fire_pct.append(round(avg_fire_pct, 2))

        print(f"  ep{ep:3d}  acc={acc:.4f}  best={best_acc:.4f}  "
              f"Δref={100*(acc-REF_ACC):+.2f}pp  "
              f"fire={avg_fire_pct:.1f}%  "
              f"τ={tau:.2e}  [{elapsed:.1f}s]", flush=True)

        results[key] = {
            "mode": mode, "tau_key": tau_key, "tau_final": round(tau, 6),
            "best_acc": round(best_acc, 4),
            "delta_vs_ref": round(best_acc - REF_ACC, 4),
            "hist_ep": hist_ep, "hist_acc": hist_acc,
            "hist_fire_pct": hist_fire_pct,
            "elapsed_s": round(time.time() - t_start, 1),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    elapsed = time.time() - t_start
    print(f"  DONE: best={best_acc:.4f}  Δref={100*(best_acc-REF_ACC):+.2f}pp  "
          f"avg_fire={sum(hist_fire_pct)/len(hist_fire_pct):.1f}%  {elapsed:.1f}s")
    return best_acc


CONFIGS = {
    "Ref":              {"mode": "ref",       "tau": None,       "lr": 3e-3},
    "GTF_sel_lo":       {"mode": "selective", "tau": "lo",       "lr": args.lr},
    "GTF_sel_hi":       {"mode": "selective", "tau": "hi",       "lr": args.lr},
    "GTF_full_lo":      {"mode": "full",      "tau": "lo",       "lr": args.lr},
    "GTF_full_hi":      {"mode": "full",      "tau": "hi",       "lr": args.lr},
    "GTF_sel_adaptive": {"mode": "selective", "tau": "adaptive", "lr": args.lr},
}


def main():
    tr, va = load_data()

    print(f"\n{'='*70}")
    print(f"step969 — Gradient Threshold Firing (selective vs full reset)")
    print(f"  device={DEVICE}  seed={SEED}  epochs={EPOCHS}  frac={args.frac}")
    print(f"  SELECTIVE: fire independently, zero only fired params' grad")
    print(f"  FULL:      fire, then reset ALL params' grad (fresh start)")
    print(f"{'='*70}")

    # Build a fresh model just for calibration
    cal_model = build_model()
    tau_lo, tau_hi = calibrate_threshold(cal_model, tr, N_CALIB)
    del cal_model

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}

    for key in keys:
        train_config(key, CONFIGS[key], tr, va, tau_lo, tau_hi, results)

    print(f"\n{'='*70}")
    print(f"STEP 969 SUMMARY")
    print(f"  τ_lo={tau_lo:.2e}  τ_hi={tau_hi:.2e}")
    print(f"{'='*70}")
    for k, r in results.items():
        avg_fire = sum(r["hist_fire_pct"]) / len(r["hist_fire_pct"])
        print(f"  {k:<22}  best={r['best_acc']:.4f}  "
              f"Δref={r['delta_vs_ref']*100:+.2f}pp  "
              f"avg_fire={avg_fire:.1f}%")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
