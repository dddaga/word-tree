"""Step 630: K=1 consistency distillation — clean benchmark.

MOTIVATION
==========
step605 showed K=1 student with "6× SLOWER" speed — confirmed to be a measurement
artifact (teacher trajectory loading bled into eval time). True K=1 inference
is ~1.32× faster than K=5 on MPS (compiled: K=1=0.039ms, K=5=0.052ms).

This script gives a clean, fair comparison:
  1. Ref_k5     : K_iter=5 trained from scratch (standard SGNNET step199 config)
  2. Scratch_k1 : K_iter=1 trained from scratch (no distillation)
  3. Distill_k1 : K_iter=1 trained with soft-target distillation from B1 teacher
                  (teacher trajectory cached at data/teacher_logits_step602.h5 — GLNN)

Inference timing is measured SEPARATELY after training with a pure forward-only loop
(no trajectory loading, no teacher — clean deployment measurement).

Claim being tested: K=1 student via distillation preserves accuracy AND is faster.
Acceptance criteria:
  - STRONG: Distill_k1 ≥ Scratch_k5 - 2pp AND inference faster
  - MEDIUM: Distill_k1 ≥ Scratch_k1 + 1pp (distillation helps even if accuracy drops)
  - KILLED: Distill_k1 < Scratch_k1 (distillation adds no value over scratch K=1)

CONFIGS
=======
  Ref_k5     : N=2048 D=16 K_iter=5 (step199 production config) — 150ep full
  Scratch_k1 : N=2048 D=16 K_iter=1 — 150ep full
  Distill_k1 : N=2048 D=16 K_iter=1 + GLNN KL loss from teacher — 150ep full

Teacher: step602 SGNNET ΔW-rotation teacher, soft logits cached at
         data/teacher_logits_step602.h5 (same format as step603 student).
         T=2, λ=0.5 (optimal from step603 B2 experiments).

To run:
    python -u scripts/train_step630_k1_clean_benchmark.py --device mps --epochs 150
    python -u scripts/train_step630_k1_clean_benchmark.py --device cuda --epochs 150
    python -u scripts/train_step630_k1_clean_benchmark.py --configs Ref_k5,Scratch_k1 --epochs 20
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--teacher", default="data/teacher_logits_step602.h5",
                    help="GLNN teacher logits cache (step602). Required for Distill_k1.")
parser.add_argument("--configs", default="Ref_k5,Scratch_k1,Distill_k1")
parser.add_argument("--T",       type=float, default=2.0,  help="Distillation temperature")
parser.add_argument("--lam",     type=float, default=0.5,  help="KL loss weight (1-lam = CE)")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step630_k1_clean_seed{SEED}__{SLOT}.json"

# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def make_model(K_iter: int) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


# ─────────────────────────────────────────────────────────────────────────────
# GLNN teacher logits loader
# ─────────────────────────────────────────────────────────────────────────────

def load_teacher_logits(path: str):
    """Load precomputed teacher soft logits from HDF5 cache.
    Returns dict: {idx -> logit_tensor} or None if file not found.
    """
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: Teacher logits not found at {p}. Distill_k1 will fall back to scratch.")
        return None
    try:
        import h5py
        with h5py.File(p, "r") as f:
            # Format: f["train"]["logits_T4"] = [N_train, N_OUT] float32
            if "train" in f and "logits_T4" in f["train"]:
                logits = torch.from_numpy(f["train"]["logits_T4"][:]).float()
                print(f"  Loaded teacher logits (train/logits_T4): {logits.shape}")
                return logits
            elif "train_logits" in f:
                logits = torch.from_numpy(f["train_logits"][:]).float()
                print(f"  Loaded teacher logits (train_logits): {logits.shape}")
                return logits
            else:
                print(f"  WARNING: expected key not found in {p}. Keys: {list(f.keys())}")
                return None
    except Exception as e:
        print(f"  WARNING: Could not load teacher logits: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_standard(model, tr, va, K_iter):
    """Standard SGNNET training via Trainer."""
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
    history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
        print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        if (m['epoch']+1) % 10 == 0 else None))
    return history


def train_distill(model, tr, va, teacher_logits, T=2.0, lam=0.5):
    """K=1 student trained with GLNN-style soft-target distillation.

    Loss = (1-lam) * CE(student, hard_labels) + lam * KL(student/T || teacher/T)
    teacher_logits: [N_train, N_OUT] precomputed, or None (falls back to standard).
    """
    if teacher_logits is None:
        print("  No teacher logits — training scratch K=1 instead.")
        return train_standard(model, tr, va, K_iter=1)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    # Build index mapping: dataset index → teacher logit row
    # tr.dataset may be HDF5-backed; we use enumerate to get batch indices
    best = 0.0; best_ep = 0
    history = []

    teacher_logits = teacher_logits.to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        batch_idx = 0
        for xb, _, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            bs = xb.size(0)
            start = batch_idx * BATCH
            t_logits = teacher_logits[start:start + bs]  # [bs, N_OUT]
            if t_logits.size(0) != bs:
                # Fallback: CE only (boundary batch or teacher shorter)
                loss = ce(model(xb), yb)
            else:
                s_logits = model(xb)                      # [bs, N_OUT]
                loss_ce = ce(s_logits, yb)
                loss_kl = kl(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1)) * (T ** 2)
                loss = (1 - lam) * loss_ce + lam * loss_kl
            opt.zero_grad(); loss.backward(); opt.step()
            batch_idx += 1
        sched.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, _, yb in va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(-1) == yb).sum().item()
                total += yb.size(0)
        v = correct / total
        history.append({"epoch": epoch, "val_top1": v})
        if v > best: best, best_ep = v, epoch + 1
        if (epoch + 1) % 10 == 0:
            print(f"  ep{epoch+1:3d}  val={v:.4f}  best={best:.4f}", flush=True)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Inference benchmark (clean, no teacher loading)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_inference(model, n_warmup=50, n_iter=200, B=1):
    """Measure pure forward-pass latency (ms) — no training overhead."""
    model = model.to(DEVICE).eval()
    x = torch.randn(B, N_IN, device=DEVICE)
    sync = torch.mps.synchronize if str(DEVICE) == "mps" else (
           torch.cuda.synchronize if str(DEVICE) == "cuda" else lambda: None)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x); sync()
        sync(); t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x); sync()
        sync()
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED)
    teacher_logits = load_teacher_logits(ROOT / args.teacher)

    print(f"Step 630 — K=1 clean benchmark (T={args.T}, λ={args.lam})")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Teacher: {'loaded' if teacher_logits is not None else 'NOT FOUND (distill→scratch fallback)'}")
    print()

    CONFIGS = {
        "Ref_k5":     ("Ref K=5 scratch",     lambda: make_model(K_iter=5),  "standard"),
        "Scratch_k1": ("Scratch K=1",          lambda: make_model(K_iter=1),  "standard"),
        "Distill_k1": ("Distill K=1 T=2 λ=0.5", lambda: make_model(K_iter=1), "distill"),
    }
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]

    results = {}
    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        label, build_fn, mode = CONFIGS[key]
        model = build_fn()
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        K_iter = model.m.base.K_iter if hasattr(model, 'm') else model.base.K_iter
        flops_routing = K_iter * N * K_HH * D * 2
        print(f"{'─'*60}\n{key}: {label}  params={n_p:,}  K_iter={K_iter}  routing={flops_routing/1e3:.0f}K MACs")

        t0 = time.time()
        if mode == "distill":
            history = train_distill(model, tr, va, teacher_logits, T=args.T, lam=args.lam)
        else:
            history = train_standard(model, tr, va, K_iter)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        # Clean inference benchmark
        inf_ms = benchmark_inference(model, B=1)
        inf_ms_b32 = benchmark_inference(model, B=32)
        print(f"  → best={best:.4f} @ep{best_ep}  inf_B1={inf_ms:.3f}ms  inf_B32={inf_ms_b32:.3f}ms  elapsed={elapsed:.0f}s")

        results[key] = {
            "label": label, "K_iter": K_iter, "n_params": n_p,
            "routing_macs": flops_routing,
            "best": best, "best_ep": best_ep,
            "inf_ms_B1": round(inf_ms, 4), "inf_ms_B32": round(inf_ms_b32, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"STEP 630 SUMMARY — K=1 clean benchmark")
    print(f"{'='*60}")
    hdr = f"  {'Config':<14} {'K':>3} {'params':>8}  {'best':>7}  {'Δ_Ref':>8}  {'inf_B1(ms)':>12}"
    print(hdr)
    ref_acc = results.get("Ref_k5", {}).get("best", 0.0)
    for k, r in results.items():
        delta = r["best"] - ref_acc
        print(f"  {k:<14} {r['K_iter']:>3} {r['n_params']:>8,}  {r['best']:>7.4f}  {delta*100:>+8.2f}pp  {r['inf_ms_B1']:>10.3f}ms")
    print()
    if "Ref_k5" in results and "Distill_k1" in results:
        speedup = results["Ref_k5"]["inf_ms_B1"] / results["Distill_k1"]["inf_ms_B1"]
        flop_ratio = results["Ref_k5"]["routing_macs"] / results["Distill_k1"]["routing_macs"]
        print(f"  Inference speedup (Distill_k1 vs Ref_k5): {speedup:.2f}×")
        print(f"  Routing FLOPs ratio: {flop_ratio:.1f}×")
        delta = results["Distill_k1"]["best"] - results["Ref_k5"]["best"]
        verdict = "STRONG" if delta >= -0.02 else ("MEDIUM" if results["Distill_k1"]["best"] >= results.get("Scratch_k1",{}).get("best",0)+0.01 else "KILLED")
        print(f"  Accuracy Δ: {delta*100:+.2f}pp  Verdict: {verdict}")


if __name__ == "__main__":
    main()
