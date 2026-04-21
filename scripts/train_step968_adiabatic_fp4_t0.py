"""Step 968: Adiabatic training + NV FP4 — long-horizon SGNNET on VGG FC.

TWO IDEAS:
1. "Hold my neuron" (adiabatic update) — update only top-K params by |grad| per batch.
   Physics analogy: adiabatic process = one quantum of change at a time.
   Hypothesis: sparse coordinate-descent finds lower-energy (better-generalized) minima.
   Cost: needs far more epochs to converge. At k=1 on 5060ti: ~2.5s/ep → 70K ep/2 days.

2. NV FP4 quantization-aware training (simulated E2M1).
   Per-group-16 scale, STE (straight-through estimator) for gradients.
   Values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} — 15 representable levels.
   Hypothesis: coarser parameter space → sparser effective representation → better efficiency.

ADIABATIC MECHANISM
  Each batch:
    1. Forward + backward (compute full gradients)
    2. Zero all gradients EXCEPT top-K by |grad| magnitude
    3. Optimizer step (only K params updated)
  At K=1: pure coordinate descent, one param per batch.
  At K=5: 5 params per batch, ~74K batches to cover all 32,928 params once.

FP4 SIMULATION (E2M1 grid, per-group-16)
  grid = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  scale[g] = max(|w_g|) / 6.0  (fp32 stored, not quantized)
  w_q = round(w / scale) via nearest-grid-point → w_dq = w_q * scale
  Grad flows through via STE: backward uses fp32 weight gradient.
  Re-quantize after each optimizer step.

CONFIGS (N=2048, D=16, K_in=25, Imagenette, 100% aug data)
  Ref              — AdamW fp32, 500ep            (~4 min on 5060ti)
  FP4              — FP4 QAT + AdamW, 3000ep      (~25 min)
  Adiab_k5         — top-5 per batch, 50000ep     (~17h)
  FP4_Adiab_k5     — FP4 + top-5 per batch, 50000ep (~25h)
  Adiab_k1         — top-1 per batch, 30000ep     (~21h)

Total ≈ 133500 epochs ≈ 62h ≈ 2.5 days on 5060ti.
Logs every LOG_INTERVAL epochs. Checkpoint every CKPT_INTERVAL epochs.

ADVANCE CRITERION
  Any config reaches accuracy within 0.5pp of Ref (95.52%) → mechanism is viable.
  Any config reaches Ref accuracy with fewer epochs → faster convergence path.
  Adiabatic: watch the learning curve shape — does it keep improving past Ref's plateau?
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
parser.add_argument("--configs", default="Ref,FP4,Adiab_k5,FP4_Adiab_k5,Adiab_k1")
parser.add_argument("--lr",      type=float, default=3e-4,
                    help="LR for adiabatic configs (lower than standard AdamW)")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED     = args.seed
N        = 2048
D        = 16
N_IN     = 25088
N_OUT    = 10
K_IN     = 25
K_ITER   = 5
BATCH    = 64
REF_ACC  = 0.9552    # step199 T2 baseline

LOG_INTERVAL  = 100   # log every N epochs
CKPT_INTERVAL = 2000  # save checkpoint every N epochs

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step968_adiabatic_fp4_seed{SEED}__{SLOT}.json"
CKPT_DIR = ROOT / "checkpoints" / "step968"

CONFIGS = {
    "Ref":          {"fp4": False, "k": None, "epochs": 500,   "lr": 3e-3},
    "FP4":          {"fp4": True,  "k": None, "epochs": 3000,  "lr": 3e-3},
    "Adiab_k5":     {"fp4": False, "k": 5,    "epochs": 50000, "lr": args.lr},
    "FP4_Adiab_k5": {"fp4": True,  "k": 5,    "epochs": 50000, "lr": args.lr},
    "Adiab_k1":     {"fp4": False, "k": 1,    "epochs": 30000, "lr": args.lr},
}


# ── FP4 simulation (E2M1 grid, per-group-16, STE) ────────────────────────────
FP4_GRID  = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX   = 6.0
FP4_GROUP = 16

class FP4Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor) -> torch.Tensor:
        grid = FP4_GRID.to(w.device)
        orig = w.shape
        w_flat = w.reshape(-1)
        n_g = (w_flat.numel() + FP4_GROUP - 1) // FP4_GROUP
        w_pad = F.pad(w_flat, (0, n_g * FP4_GROUP - w_flat.numel()))
        w_g = w_pad.reshape(n_g, FP4_GROUP)
        scale = w_g.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / FP4_MAX
        w_norm = w_g / scale
        sign = w_norm.sign()
        abs_w = w_norm.abs()
        dists = (abs_w.unsqueeze(-1) - grid).abs()
        idx = dists.argmin(dim=-1)
        w_q = sign * grid[idx]
        w_dq = (w_q * scale).reshape(-1)[: w_flat.numel()].reshape(orig)
        return w_dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output  # straight-through estimator


def fp4_quantize_dequantize(w: torch.Tensor) -> torch.Tensor:
    return FP4Quantize.apply(w)


# ── Adiabatic gradient masking ────────────────────────────────────────────────
def adiabatic_mask(model: nn.Module, k: int):
    """Zero all gradients except top-k by absolute magnitude."""
    all_grads = []
    for p in model.parameters():
        if p.grad is not None:
            all_grads.append(p.grad.abs().flatten())
    if not all_grads:
        return
    flat = torch.cat(all_grads)
    if k >= flat.numel():
        return
    threshold = flat.topk(k).values.min()
    for p in model.parameters():
        if p.grad is not None:
            p.grad[p.grad.abs() < threshold] = 0.0


# ── FP4 re-quantization after optimizer step ──────────────────────────────────
def fp4_requantize(model: nn.Module):
    """Re-quantize all fp32 master weights to FP4 grid in-place (simulated)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.data.copy_(fp4_quantize_dequantize(p.data))


# ── Model builder ─────────────────────────────────────────────────────────────
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
    tr = DataLoader(TensorDataset(tr_x, tr_y), batch_size=BATCH, shuffle=True,
                    num_workers=4, pin_memory=True)
    va = DataLoader(TensorDataset(va_x, va_y), batch_size=256, shuffle=False,
                    num_workers=2)
    print(f"  Data: {len(tr_x)} train / {len(va_x)} val / {len(tr)} batches/ep")
    return tr, va


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, va, fp4: bool) -> float:
    model.eval()
    correct = total = 0
    for x, y in va:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if fp4:
            # Forward with quantized weights (STE not needed at eval)
            with torch.no_grad():
                orig_params = {}
                for name, p in model.named_parameters():
                    orig_params[name] = p.data.clone()
                    p.data.copy_(fp4_quantize_dequantize(p.data))
                logits = model(x)
                for name, p in model.named_parameters():
                    p.data.copy_(orig_params[name])
        else:
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── Training loop ─────────────────────────────────────────────────────────────
def train_config(key: str, cfg: dict, tr, va, results: dict):
    fp4    = cfg["fp4"]
    k      = cfg["k"]         # None = standard, int = adiabatic top-K
    epochs = cfg["epochs"]
    lr     = cfg["lr"]

    model = build_model()
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Use SGD for adiabatic (cleaner coordinate-descent semantics); AdamW for standard
    if k is not None:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    if fp4:
        fp4_requantize(model)  # start from FP4 grid

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{key}_seed{SEED}.pt"

    hist_acc   = []
    hist_ep    = []
    best_acc   = 0.0
    t_start    = time.time()
    t_ep_start = time.time()

    print(f"\n{'─'*60}")
    print(f"{key}: fp4={fp4}  k={k}  epochs={epochs:,}  lr={lr}  params={n_p:,}")
    print(f"  Estimated time @ {2.46 if k==1 else 1.2 if k==5 else 0.52:.2f}s/ep: "
          f"{epochs * (2.46 if k==1 else 1.2 if k==5 else 0.52) / 3600:.1f}h", flush=True)

    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in tr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)

            if fp4:
                # Forward with quantized weights (STE)
                with torch.no_grad():
                    orig = {n: p.data.clone() for n, p in model.named_parameters()}
                    for p in model.parameters(): p.data.copy_(fp4_quantize_dequantize(p.data))
                loss = F.cross_entropy(model(bx), by)
                # Restore fp32 master weights before backward
                for n, p in model.named_parameters(): p.data.copy_(orig[n])
            else:
                loss = F.cross_entropy(model(bx), by)

            opt.zero_grad()
            loss.backward()

            if k is not None:
                adiabatic_mask(model, k)

            opt.step()

            if fp4:
                fp4_requantize(model)

        if ep % LOG_INTERVAL == 0 or ep == epochs or ep <= 5:
            acc = evaluate(model, va, fp4=False)  # eval on fp32 weights
            elapsed = time.time() - t_start
            ep_rate = elapsed / ep
            eta_h   = (epochs - ep) * ep_rate / 3600

            if acc > best_acc:
                best_acc = acc

            hist_acc.append(round(acc, 4))
            hist_ep.append(ep)

            print(f"  ep{ep:7,}  acc={acc:.4f}  best={best_acc:.4f}  "
                  f"Δ_ref={100*(acc-REF_ACC):+.2f}pp  "
                  f"eta={eta_h:.1f}h  [{elapsed/3600:.1f}h elapsed]", flush=True)

            # Save partial results
            results[key] = {
                "fp4": fp4, "k": k, "epochs_run": ep, "epochs_target": epochs,
                "best_acc": round(best_acc, 4),
                "delta_vs_ref": round(best_acc - REF_ACC, 4),
                "hist_ep": hist_ep, "hist_acc": hist_acc,
                "elapsed_h": round(elapsed / 3600, 2),
            }
            OUT_PATH.parent.mkdir(exist_ok=True)
            OUT_PATH.write_text(json.dumps(results, indent=2))

        if ep % CKPT_INTERVAL == 0:
            torch.save({"model_state": model.state_dict(), "ep": ep, "acc": best_acc}, ckpt_path)
            print(f"  [ckpt saved @ ep{ep:,}]", flush=True)

    elapsed = time.time() - t_start
    print(f"  DONE: best={best_acc:.4f}  Δ_ref={100*(best_acc-REF_ACC):+.2f}pp  "
          f"total={elapsed/3600:.1f}h")
    return best_acc


def main():
    tr, va = load_data()

    print(f"\n{'='*70}")
    print(f"step968 — Adiabatic + FP4 long-horizon training")
    print(f"  device={DEVICE}  seed={SEED}")
    print(f"  Ref baseline: {REF_ACC:.4f}")
    print(f"  Adiabatic: update top-K params per batch (coordinate descent)")
    print(f"  FP4: per-group-16 E2M1 quantization + STE")
    print(f"{'='*70}")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}

    for key in keys:
        train_config(key, CONFIGS[key], tr, va, results)

    print(f"\n{'='*70}")
    print(f"STEP 968 SUMMARY")
    print(f"{'='*70}")
    for k, r in results.items():
        print(f"  {k:<20}  best={r['best_acc']:.4f}  "
              f"Δ_ref={r['delta_vs_ref']*100:+.2f}pp  "
              f"ep_run={r['epochs_run']:,}/{r['epochs_target']:,}  "
              f"{r['elapsed_h']:.1f}h")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
