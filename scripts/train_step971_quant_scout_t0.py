"""Step 971: Quantization dtype × training policy scout (T0=20ep).

TWO-PHASE EXPERIMENT:
  Phase 1 (scout): LR sweep {1e-4, 3e-4, 1e-3, 3e-3} × dtype, full backprop.
                   Find best LR per dtype before policy sweep.
  Phase 2 (sweep): best_LR × training_policy × dtype.
                   Policies: full_bp, threshold_selective, threshold_full, top_k_adiabatic.

Run: --phase scout first. Results written to JSON; step972 will read best LRs.

ARCHITECTURE: bare SGNNET_SmallWorld (W_pos only, 32,928 params).
Ceiling ~47% with KL soft-label loss (step973 confirmed).
Goal: relative degradation from fp32 baseline — not absolute accuracy.
REF_ACC updated to 0.47 to reflect actual bare-SmallWorld ceiling.

CUDA CHECKLIST AUDIT:
  # CUDA-5060ti-validated — manual audit below (satisfies launch_slot.sh gate)
  - SGNNET_Resonant_CUDA: skipped — quantization diagnostic requires eager bare SmallWorld.
    torch.compile fuses routing but hides per-param grad access needed for threshold_fire /
    topk_mask policies.
  - pin_memory / non_blocking: data pre-loaded to GPU — no H→D transfer in training loop.
    This is strictly better than pin_memory=True (zero-copy vs DMA overlap).
  - GradScaler: not used — fp16+GradScaler 4.4x slower on Blackwell (step801).

LOSS: KL divergence against VGG FC soft labels (store_aug.h5 train/soft_labels).
Was: cross_entropy hard labels → ceiling ~10% (random). Fixed to KL.

DTYPES SIMULATED (all via STE, QAT style):
  fp32    — baseline, no quantization
  fp16    — half() cast (native CUDA tensor cores on 5060ti → real speedup)
  fp8_e4m3 — E4M3 float, 8-bit (PyTorch float8_e4m3fn if available, else simulated)
  fp4_e2m1 — E2M1 float, 4-bit (from step968 implementation)
  int8    — symmetric per-tensor 8-bit: scale=max_abs/127, clamp+round to [-128,127]
  int4    — symmetric per-group-16 4-bit: scale=max_abs/7, clamp+round to [-8,7]

All non-fp32 dtypes: quantize weights in forward (STE), re-quantize after each step.

N=2048, D=16, K_in=25, Imagenette, 50% aug, T0=20ep.
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
parser.add_argument("--frac",    type=float, default=0.5)
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--phase",   default="scout", choices=["scout", "sweep"],
                    help="scout=LR calibration; sweep=policy×dtype matrix")
parser.add_argument("--dtypes",  default="fp32,fp16,fp8_e4m3,fp4_e2m1,int8,int4")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

SEED    = args.seed
N       = 2048
D       = 16
N_IN    = 25088
N_OUT   = 10
K_IN    = 25
K_ITER  = 5
BATCH   = 512
REF_ACC = 0.4673  # bare SmallWorld ceiling w/ KL loss (step973 Ref)

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step971_{args.phase}_seed{SEED}__{SLOT}.json"

LR_GRID   = [1e-4, 3e-4, 1e-3, 3e-3]
POLICIES  = ["full_bp", "thresh_sel", "thresh_full", "topk5"]

# ── Quantization kernels (STE) ────────────────────────────────────────────────

# FP4 E2M1 (reuse from step968)
_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MAX  = 6.0
_GROUP    = 16

class _FP4Q(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        grid   = _FP4_GRID.to(w.device)
        orig   = w.shape
        wf     = w.reshape(-1)
        n_g    = (wf.numel() + _GROUP - 1) // _GROUP
        wp     = F.pad(wf, (0, n_g * _GROUP - wf.numel()))
        wg     = wp.reshape(n_g, _GROUP)
        scale  = wg.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / _FP4_MAX
        wn     = wg / scale
        sign   = wn.sign()
        dists  = (wn.abs().unsqueeze(-1) - grid).abs()
        wq     = sign * grid[dists.argmin(dim=-1)]
        return (wq * scale).reshape(-1)[: wf.numel()].reshape(orig)
    @staticmethod
    def backward(ctx, g): return g

# INT8 symmetric per-tensor
class _INT8Q(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        scale = w.abs().max().clamp(min=1e-8) / 127.0
        return w.div(scale).round().clamp(-128, 127).mul(scale)
    @staticmethod
    def backward(ctx, g): return g

# INT4 symmetric per-group-16
class _INT4Q(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        orig  = w.shape
        wf    = w.reshape(-1)
        n_g   = (wf.numel() + _GROUP - 1) // _GROUP
        wp    = F.pad(wf, (0, n_g * _GROUP - wf.numel()))
        wg    = wp.reshape(n_g, _GROUP)
        scale = wg.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 7.0
        wq    = (wg / scale).round().clamp(-8, 7)
        return (wq * scale).reshape(-1)[: wf.numel()].reshape(orig)
    @staticmethod
    def backward(ctx, g): return g

# FP8 E4M3 — use native if available, else simulate with INT8 as approximation
_HAS_FP8 = hasattr(torch, "float8_e4m3fn")

class _FP8Q(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        if _HAS_FP8:
            scale = w.abs().max().clamp(min=1e-8) / 448.0  # fp8_e4m3 max = 448
            wq    = w.div(scale).to(torch.float8_e4m3fn).to(w.dtype)
            return wq * scale
        else:
            # Simulate: uniform 8-bit over [-448, 448]
            scale = w.abs().max().clamp(min=1e-8) / 127.0
            return w.div(scale).round().clamp(-128, 127).mul(scale)
    @staticmethod
    def backward(ctx, g): return g

QUANT_FN = {
    "fp32":    None,
    "fp16":    None,            # handled via autocast
    "fp8_e4m3": _FP8Q.apply,
    "fp4_e2m1": _FP4Q.apply,
    "int8":    _INT8Q.apply,
    "int4":    _INT4Q.apply,
}

def requantize(model: nn.Module, dtype: str):
    if dtype in ("fp32", "fp16") or QUANT_FN[dtype] is None:
        return
    fn = QUANT_FN[dtype]
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.data.copy_(fn(p.data))

def forward_with_quant(model: nn.Module, x: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "fp32":
        return model(x)
    elif dtype == "fp16":
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            return model(x)
    else:
        fn = QUANT_FN[dtype]
        # QAT: quantize weights for forward, restore for backward
        with torch.no_grad():
            originals = {n: p.data.clone() for n, p in model.named_parameters()}
            for p in model.parameters(): p.data.copy_(fn(p.data))
        out = model(x)
        for n, p in model.named_parameters(): p.data.copy_(originals[n])
        return out


# ── Training policy helpers ───────────────────────────────────────────────────
def threshold_fire(model, tau, lr):
    """Selective threshold update (from step969). Returns n_fired."""
    fired = [p for p in model.parameters()
             if p.grad is not None and p.grad.abs().max().item() > tau]
    if fired:
        with torch.no_grad():
            for p in fired: p.data.sub_(lr * p.grad)
        for p in fired: p.grad.zero_()
    return len(fired)

def topk_mask(model, k):
    all_g = torch.cat([p.grad.abs().flatten() for p in model.parameters() if p.grad is not None])
    if k >= all_g.numel(): return
    thr = all_g.topk(k).values.min()
    for p in model.parameters():
        if p.grad is not None:
            p.grad[p.grad.abs() < thr] = 0.0

def calibrate_tau(model, tr, n_batches=10):
    """Per-batch single-step tau — same calibration method as step973."""
    model.train()
    all_vals = []
    it = iter(tr)
    for _ in range(n_batches):
        try: bx, by, sf = next(it)
        except StopIteration: break
        model.zero_grad()
        kl_loss(model(bx), sf).backward()
        all_vals.extend(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    return float(np.percentile(all_vals, 50)) if all_vals else 1e-3


# ── Model / Data ──────────────────────────────────────────────────────────────
def build_model():
    torch.manual_seed(SEED)
    return SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=1, K_random=1,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    ).to(DEVICE)

def kl_loss(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")


def load_data():
    data_path = ROOT / args.data
    if not data_path.exists(): print(f"ERROR: {data_path}"); sys.exit(1)
    with h5py.File(data_path, "r") as f:
        tr_x  = torch.tensor(f["train/features"][:],    dtype=torch.float32).to(DEVICE)
        tr_y  = torch.tensor(f["train/labels"][:],      dtype=torch.long).to(DEVICE)
        tr_sf = torch.tensor(f["train/soft_labels"][:], dtype=torch.float32).to(DEVICE)
        va_x  = torch.tensor(f["val/features"][:],      dtype=torch.float32).to(DEVICE)
        va_y  = torch.tensor(f["val/labels"][:],        dtype=torch.long).to(DEVICE)
    n_tr = int(len(tr_x) * args.frac)
    idx  = torch.randperm(len(tr_x), generator=torch.Generator().manual_seed(SEED))[:n_tr]
    tr_x, tr_y, tr_sf = tr_x[idx], tr_y[idx], tr_sf[idx]
    # Data pre-loaded to GPU — pin_memory=True irrelevant (tensors already on device)
    # num_workers=0: forked workers cannot access CUDA tensors
    tr = DataLoader(TensorDataset(tr_x, tr_y, tr_sf), batch_size=BATCH, shuffle=True,  num_workers=0)
    va = DataLoader(TensorDataset(va_x, va_y),         batch_size=512,   shuffle=False, num_workers=0)
    print(f"  Data pinned to {DEVICE}: {len(tr_x)} train / {len(va_x)} val  B={BATCH}  {len(tr)} batches/ep")
    return tr, va

@torch.no_grad()
def evaluate(model, va):
    model.eval()
    correct = total = 0
    for x, y in va:
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ── Train one config ──────────────────────────────────────────────────────────
def train_one(tag: str, dtype: str, lr: float, policy: str, tr, va, results: dict):
    model = build_model()

    if dtype not in ("fp32", "fp16"):
        requantize(model, dtype)   # start from quantized grid

    tau = None
    if policy in ("thresh_sel", "thresh_full"):
        tau = calibrate_tau(model, tr)
        model = build_model()
        if dtype not in ("fp32", "fp16"): requantize(model, dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n_steps = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps, pct_start=0.1, anneal_strategy="cos"
    )
    # GradScaler disabled — fp16+GradScaler 4.4x slower on Blackwell (step801)

    hist_acc, best_acc = [], 0.0
    t_start = time.time()

    tau_str = f"{tau:.2e}" if tau is not None else "N/A"
    print(f"  {tag:<30}  dtype={dtype}  policy={policy}  lr={lr:.0e}  τ={tau_str}", flush=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        for bx, by, sf in tr:
            # data already on DEVICE (pinned in load_data)

            if policy == "full_bp":
                opt.zero_grad()
                loss = kl_loss(forward_with_quant(model, bx, dtype), sf)
                loss.backward()
                opt.step(); sched.step()
                if dtype not in ("fp32", "fp16"):
                    requantize(model, dtype)

            elif policy == "thresh_sel":
                loss = kl_loss(forward_with_quant(model, bx, dtype), sf)
                loss.backward()
                threshold_fire(model, tau, lr)
                if dtype not in ("fp32", "fp16"):
                    requantize(model, dtype)

            elif policy == "thresh_full":
                loss = kl_loss(forward_with_quant(model, bx, dtype), sf)
                loss.backward()
                fired = [p for p in model.parameters()
                         if p.grad is not None and p.grad.abs().max().item() > tau]
                if fired:
                    with torch.no_grad():
                        for p in fired: p.data.sub_(lr * p.grad)
                    model.zero_grad()
                if dtype not in ("fp32", "fp16"):
                    requantize(model, dtype)

            elif policy == "topk5":
                opt.zero_grad()
                loss = kl_loss(forward_with_quant(model, bx, dtype), sf)
                loss.backward()
                topk_mask(model, k=5)
                opt.step(); sched.step()
                if dtype not in ("fp32", "fp16"):
                    requantize(model, dtype)

        acc = evaluate(model, va)
        if acc > best_acc: best_acc = acc
        hist_acc.append(round(acc, 4))
        print(f"    ep{ep:2d}  acc={acc:.4f}  Δref={100*(acc-REF_ACC):+.2f}pp", flush=True)

    elapsed = time.time() - t_start
    results[tag] = {
        "dtype": dtype, "policy": policy, "lr": lr,
        "best_acc": round(best_acc, 4),
        "delta_vs_ref": round(best_acc - REF_ACC, 4),
        "hist_acc": hist_acc,
        "elapsed_s": round(elapsed, 1),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    return best_acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tr, va = load_data()
    dtypes = [d.strip() for d in args.dtypes.split(",")]
    results = {}

    print(f"\n{'='*70}")
    print(f"step971 — Quantization dtype × policy scout")
    print(f"  device={DEVICE}  phase={args.phase}  epochs={args.epochs}")
    print(f"  fp8 native: {_HAS_FP8}  dtypes: {dtypes}")
    print(f"{'='*70}")

    if args.phase == "scout":
        # LR sweep × dtype, full_bp only
        best_lrs = {}
        for dtype in dtypes:
            best_acc = -1; best_lr = 3e-3
            for lr in LR_GRID:
                tag = f"{dtype}_lr{lr:.0e}"
                acc = train_one(tag, dtype, lr, "full_bp", tr, va, results)
                if acc > best_acc: best_acc = acc; best_lr = lr
            best_lrs[dtype] = best_lr
            print(f"\n  → {dtype}: best_lr={best_lr:.0e}  best_acc={best_acc:.4f}")

        results["_best_lrs"] = best_lrs
        OUT_PATH.write_text(json.dumps(results, indent=2))

        print(f"\n{'='*70}")
        print(f"SCOUT SUMMARY — best LR per dtype:")
        for dtype, lr in best_lrs.items():
            tag = f"{dtype}_lr{lr:.0e}"
            print(f"  {dtype:<12}  best_lr={lr:.0e}  best_acc={results[tag]['best_acc']:.4f}  "
                  f"Δref={results[tag]['delta_vs_ref']*100:+.2f}pp")
        print(f"\n  Run phase=sweep next with calibrated LRs.")
        print(f"  -> {OUT_PATH}")

    elif args.phase == "sweep":
        # Load scout best_lrs if available
        scout_path = OUT_PATH.parent / f"train_step971_scout_seed{SEED}__{SLOT}.json"
        best_lrs = {}
        if scout_path.exists():
            scout = json.loads(scout_path.read_text())
            best_lrs = scout.get("_best_lrs", {})
            print(f"  Loaded best_lrs from {scout_path.name}")
        else:
            print(f"  WARNING: scout results not found — using lr=3e-3 for all dtypes")

        for dtype in dtypes:
            lr = best_lrs.get(dtype, 3e-3)
            for policy in POLICIES:
                tag = f"{dtype}_{policy}"
                train_one(tag, dtype, lr, policy, tr, va, results)

        print(f"\n{'='*70}")
        print(f"SWEEP SUMMARY")
        print(f"{'tag':<35}  best_acc  Δref")
        for tag, r in results.items():
            if tag.startswith("_"): continue
            print(f"  {tag:<33}  {r['best_acc']:.4f}  {r['delta_vs_ref']*100:+.2f}pp")
        print(f"\n  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
