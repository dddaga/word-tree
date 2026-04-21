"""Step 973: Adiabatic + threshold gradient scout with KL soft-label loss (100ep).

PURPOSE: Calibrate LR and threshold hyperparams for adiabatic/threshold training
before committing to the long 50K-epoch run. All configs use KL divergence against
VGG FC soft labels — NOT hard CE. Previous step968 used hard CE → 57% ceiling,
wrong training objective.

SOFT LABEL LOSS:
  L = KL(log_softmax(logits) || soft_labels)
  soft_labels from store_aug.h5/train/soft_labels — shape (N, 10), rows sum to 1.0
  These are VGG16.classifier outputs on Imagenette pool5 features.
  This is the SAME loss the existing Trainer uses, giving 95.52% at step199.

GPU UTILIZATION FIX:
  Batch=512 (was 64). Model has 32,928 params — tiny for a 5060ti.
  B=512 saturates tensor cores; B=64 leaves ~85% GPU idle.

ADIABATIC MECHANISM (top-K by |grad|):
  After backward, zero all grads except top-K by magnitude. SGD step.
  K controls update sparsity: K=5 → 1 full pass per ~6600 batches = ~22ep.
  K=50 → 1 full pass per ~660 batches = ~2.2ep. More practical for 100ep scout.

THRESHOLD MECHANISM (fire when ready):
  Accumulate gradients across batches. When param's max|grad| > τ → fire update.
  τ calibrated from first 10 batches: τ_lo=p50, τ_hi=p75 of per-param max|grad|.
  reset=selective: zero only fired params. reset=full: zero all after any fire.

CONFIGS (100ep each, full store_aug.h5 data, B=512):
  Ref              — AdamW lr=3e-3, soft KL (calibration baseline)
  Adiab_K5_*       — SGD, top-5, LR sweep {1e-4, 3e-4, 1e-3, 3e-3}
  Adiab_K20_*      — SGD, top-20, LR sweep
  Adiab_K50_*      — SGD, top-50, LR sweep
  GTF_sel_p50_*    — threshold p50, selective reset, LR sweep {1e-4, 1e-3, 3e-3}
  GTF_sel_p75_*    — threshold p75, selective reset, LR sweep
  GTF_full_p50_*   — threshold p50, full reset, LR sweep
  GTF_full_p75_*   — threshold p75, full reset, LR sweep

Total: 1 + 3×4 + 4×3 = 25 configs × 100ep × ~0.15s/ep ≈ 6 min on 5060ti @ B=512.

ADVANCE CRITERION:
  Any config ≥ 80% of Ref's 100ep accuracy (within 4pp) → viable for long run.
  Record best LR per (mechanism, K/tau) for step974 long-horizon run.
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

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_aug.h5")
parser.add_argument("--epochs",  type=int, default=100)
parser.add_argument("--batch",   type=int, default=512)
parser.add_argument("--configs", default="all")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps")  if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED   = args.seed
N, D   = 2048, 16
N_IN, N_OUT = 25088, 10
K_IN, K_ITER = 25, 5
REF_ACC = 0.9552

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step973_adiabatic_kl_scout_seed{SEED}__{SLOT}.json"


# ── KL soft-label loss ────────────────────────────────────────────────────────
def kl_loss(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")


# ── Adiabatic masking ─────────────────────────────────────────────────────────
def adiabatic_mask(model: nn.Module, k: int):
    grads = [p.grad.abs().flatten() for p in model.parameters() if p.grad is not None]
    if not grads: return
    flat = torch.cat(grads)
    if k >= flat.numel(): return
    thr = flat.topk(k).values.min()
    for p in model.parameters():
        if p.grad is not None:
            p.grad[p.grad.abs() < thr] = 0.0


# ── Threshold gradient firing ─────────────────────────────────────────────────
def threshold_fire(model: nn.Module, tau: float, lr: float, reset: str) -> int:
    fired = [p for p in model.parameters()
             if p.grad is not None and p.grad.abs().max().item() > tau]
    if fired:
        with torch.no_grad():
            for p in fired:
                p.data.sub_(lr * p.grad)
        if reset == "sel":
            for p in fired: p.grad.zero_()
        else:
            model.zero_grad()
    return len(fired)


def calibrate_tau(model: nn.Module, tr, n_batches: int = 10) -> tuple[float, float]:
    model.train(); model.zero_grad()
    it = iter(tr)
    for _ in range(n_batches):
        try: bx, by, sf = next(it)
        except StopIteration: break
        bx, sf = bx.to(DEVICE), sf.to(DEVICE)
        kl_loss(model(bx), sf).backward()
    vals = [p.grad.abs().max().item() for p in model.parameters() if p.grad is not None]
    model.zero_grad()
    if not vals: return 1e-3, 1e-2
    arr = np.array(vals)
    tau_lo = float(np.percentile(arr, 50))
    tau_hi = float(np.percentile(arr, 75))
    print(f"  τ_lo(p50)={tau_lo:.3e}  τ_hi(p75)={tau_hi:.3e}", flush=True)
    return tau_lo, tau_hi


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    torch.manual_seed(SEED)
    return SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=1, K_random=1,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    ).to(DEVICE)


# ── Data (returns soft labels too) ────────────────────────────────────────────
def load_data():
    path = ROOT / args.data
    if not path.exists(): print(f"ERROR: {path}"); sys.exit(1)
    with h5py.File(path, "r") as f:
        tr_x  = torch.tensor(f["train/features"][:],    dtype=torch.float32)
        tr_y  = torch.tensor(f["train/labels"][:],      dtype=torch.long)
        tr_sf = torch.tensor(f["train/soft_labels"][:], dtype=torch.float32)
        va_x  = torch.tensor(f["val/features"][:],      dtype=torch.float32)
        va_y  = torch.tensor(f["val/labels"][:],        dtype=torch.long)
    tr = DataLoader(TensorDataset(tr_x, tr_y, tr_sf), batch_size=args.batch,
                    shuffle=True, num_workers=4,
                    pin_memory=(DEVICE.type == "cuda"))
    va = DataLoader(TensorDataset(va_x, va_y), batch_size=512, shuffle=False, num_workers=2)
    print(f"  Data: {len(tr_x)} train / {len(va_x)} val  B={args.batch}  {len(tr)} batches/ep")
    return tr, va


@torch.no_grad()
def evaluate(model, va) -> float:
    model.eval()
    correct = total = 0
    for x, y in va:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── Single config train ───────────────────────────────────────────────────────
def train_config(tag: str, cfg: dict, tr, va, tau_lo: float, tau_hi: float,
                 results: dict) -> float:
    mech  = cfg["mech"]   # "adamw", "adiabatic", "gtf"
    lr    = cfg["lr"]
    k     = cfg.get("k")
    tau   = tau_lo if cfg.get("tau_pct") == 50 else tau_hi
    reset = cfg.get("reset", "sel")

    model = build_model()
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.zero_grad()

    if mech == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    hist_acc, best_acc = [], 0.0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        for bx, by, sf in tr:
            bx, sf = bx.to(DEVICE), sf.to(DEVICE)
            loss = kl_loss(model(bx), sf)

            if mech == "adamw":
                opt.zero_grad()
                loss.backward()
                opt.step()

            elif mech == "adiabatic":
                # Accumulate grads, mask top-K, SGD step
                opt_zero_needed = (ep == 1 and bx is bx)  # always zero before backward
                model.zero_grad()
                loss.backward()
                adiabatic_mask(model, k)
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None and p.grad.abs().max() > 0:
                            p.data.sub_(lr * p.grad)

            elif mech == "gtf":
                loss.backward()  # accumulate
                threshold_fire(model, tau, lr, reset)

        acc = evaluate(model, va)
        if acc > best_acc: best_acc = acc
        hist_acc.append(round(acc, 4))

        if ep % 10 == 0 or ep == args.epochs:
            elapsed = time.time() - t0
            print(f"  ep{ep:3d}  acc={acc:.4f}  best={best_acc:.4f}"
                  f"  Δref={100*(acc-REF_ACC):+.2f}pp  [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0
    results[tag] = {
        "mech": mech, "lr": lr, "k": k,
        "tau_pct": cfg.get("tau_pct"), "reset": reset,
        "best_acc": round(best_acc, 4),
        "final_acc": hist_acc[-1],
        "delta_vs_ref": round(best_acc - REF_ACC, 4),
        "hist_acc": hist_acc,
        "elapsed_s": round(elapsed, 1),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"  DONE {tag}: best={best_acc:.4f}  Δref={100*(best_acc-REF_ACC):+.2f}pp  {elapsed:.0f}s")
    return best_acc


# ── Config table ──────────────────────────────────────────────────────────────
CONFIGS: dict[str, dict] = {
    "Ref": {"mech": "adamw", "lr": 3e-3},
}

# Adiabatic: K × LR
for _k in [5, 20, 50]:
    for _lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        tag = f"Adiab_K{_k}_lr{_lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
        CONFIGS[tag] = {"mech": "adiabatic", "lr": _lr, "k": _k}

# GTF: tau_pct × reset × LR
for _pct in [50, 75]:
    for _reset in ["sel", "full"]:
        for _lr in [1e-4, 1e-3, 3e-3]:
            tag = f"GTF_{_reset}_p{_pct}_lr{_lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
            CONFIGS[tag] = {"mech": "gtf", "lr": _lr, "tau_pct": _pct, "reset": _reset}


def main():
    tr, va = load_data()

    # Calibrate thresholds using soft-label KL gradients
    print(f"\nCalibrating τ from soft-label gradients...")
    cal_model = build_model()
    tau_lo, tau_hi = calibrate_tau(cal_model, tr)
    del cal_model

    n_p = sum(p.numel() for p in build_model().parameters() if p.requires_grad)
    keys = list(CONFIGS.keys()) if args.configs == "all" else \
           [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]

    est_s = len(keys) * args.epochs * (len(tr) * 0.002)  # rough estimate
    print(f"\n{'='*70}")
    print(f"step973 — Adiabatic + GTF scout (KL soft-label loss)")
    print(f"  device={DEVICE}  epochs={args.epochs}  B={args.batch}")
    print(f"  params={n_p:,}  τ_lo={tau_lo:.3e}  τ_hi={tau_hi:.3e}")
    print(f"  {len(keys)} configs × {args.epochs}ep  ≈ {est_s/60:.0f} min estimated")
    print(f"{'='*70}")

    results: dict = {}
    for tag in keys:
        print(f"\n── {tag} ──")
        train_config(tag, CONFIGS[tag], tr, va, tau_lo, tau_hi, results)

    # Summary sorted by best_acc
    ranked = sorted(results.items(), key=lambda x: x[1]["best_acc"], reverse=True)
    ref_acc = results.get("Ref", {}).get("best_acc", 0)

    print(f"\n{'='*70}")
    print(f"STEP 973 SCOUT SUMMARY  (Ref@100ep = {ref_acc:.4f})")
    print(f"{'='*70}")
    print(f"{'config':<30}  best    Δref      mech          lr       k/τ")
    for tag, r in ranked:
        k_str = str(r["k"]) if r["k"] else f"p{r['tau_pct']}/{r['reset']}"
        print(f"  {tag:<28}  {r['best_acc']:.4f}  {r['delta_vs_ref']*100:+.2f}pp"
              f"  {r['mech']:<12}  {r['lr']:.0e}  {k_str}")

    # Best per mechanism
    print(f"\nBest per mechanism:")
    for mech in ["adiabatic", "gtf"]:
        best = max((r for r in results.values() if r["mech"] == mech),
                   key=lambda r: r["best_acc"], default=None)
        if best:
            tag = next(t for t, r in results.items() if r is best)
            print(f"  {mech:<12}: {tag}  → {best['best_acc']:.4f}  "
                  f"(LR={best['lr']:.0e}  k/τ={best['k'] or best['tau_pct']})")
    print(f"\n→ {OUT_PATH}")
    print(f"→ Use best LRs/thresholds in step974 for long-horizon run.")


if __name__ == "__main__":
    main()
