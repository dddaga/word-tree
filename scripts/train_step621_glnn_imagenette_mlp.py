"""Step 621: GLNN distillation on Imagenette with MLP student.

MOTIVATION
==========
step603 showed GLNN distillation (SGNNET teacher → SGNNET student) gives +0.10pp
at T2. Question: does the teacher's knowledge benefit a fundamentally weaker
student architecture (MLP)? If GLNN improves an MLP student, it suggests the
teacher soft-targets carry calibration beyond what hard labels provide.

Configs:
  Ref_mlp   : MLP (25088→h→10) scratch, ~67K params, matched to SGNNET
  Distill_mlp : same MLP + GLNN KL loss from SGNNET teacher (T=2, λ=0.5)

Teacher: step602 SGNNET ΔW-rotation logits cached at data/teacher_logits_step602.h5
  Key: f['train']['logits_T4'] — shape [9469, 10], float32

Accept if Distill_mlp > Ref_mlp + 0.5pp (distillation provides measurable lift).
Kill if Distill_mlp ≤ Ref_mlp (zero benefit).

Scale: N_in=25088, h=2 (~50K) or h=3 (~75K) for matched budget
Tier: T2 (150ep, 100% data)

To run:
    python -u scripts/train_step621_glnn_imagenette_mlp.py --device cpu --epochs 150
    python -u scripts/train_step621_glnn_imagenette_mlp.py --device mps --epochs 150
    python -u scripts/train_step621_glnn_imagenette_mlp.py --device cuda --epochs 150
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

from src.training.dataset import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--teacher", default="data/teacher_logits_step602.h5")
parser.add_argument("--h",       type=int, default=2, help="MLP hidden dim (h=2→50K, h=3→75K params)")
parser.add_argument("--T",       type=float, default=2.0)
parser.add_argument("--lam",     type=float, default=0.5)
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10
T = args.T; LAM = args.lam

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step621_glnn_imagenette_mlp_h{args.h}_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_in: int, h: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(),
            nn.Linear(h, n_out),
        )

    def forward(self, x):
        return self.net(x)


def make_mlp():
    torch.manual_seed(SEED)
    return MLP(N_IN, args.h, N_OUT)


# ─────────────────────────────────────────────────────────────────────────────
# Teacher logits
# ─────────────────────────────────────────────────────────────────────────────

def load_teacher(path):
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: teacher not found at {p}"); return None
    try:
        import h5py
        with h5py.File(p, "r") as f:
            if "train" in f and "logits_T4" in f["train"]:
                logits = torch.from_numpy(f["train"]["logits_T4"][:]).float()
                print(f"  Teacher loaded (train/logits_T4): {logits.shape}")
                return logits
            print(f"  WARNING: expected key not found. Keys: {list(f.keys())}"); return None
    except Exception as e:
        print(f"  WARNING: {e}"); return None


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(model, tr, va, teacher_logits=None):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    best = 0.0; best_ep = 0; history = []
    teacher_on_device = teacher_logits.to(DEVICE) if teacher_logits is not None else None

    for epoch in range(EPOCHS):
        model.train()
        for bi, (xb, _, yb) in enumerate(tr):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            s_logits = model(xb)
            if teacher_on_device is not None:
                start = bi * BATCH
                t_batch = teacher_on_device[start:start + xb.size(0)]
                if t_batch.size(0) == xb.size(0):
                    loss_ce = ce(s_logits, yb)
                    loss_kl = kl(
                        F.log_softmax(s_logits / T, dim=-1),
                        F.softmax(t_batch / T, dim=-1)) * (T ** 2)
                    loss = (1 - LAM) * loss_ce + LAM * loss_kl
                else:
                    loss = ce(s_logits, yb)
            else:
                loss = ce(s_logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, _, yb in va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(-1) == yb).sum().item()
                total += yb.size(0)
        v = correct / total
        history.append(v)
        if v > best: best, best_ep = v, epoch + 1
        if (epoch + 1) % 15 == 0:
            print(f"  ep{epoch+1:3d}  val={v:.4f}  best={best:.4f}", flush=True)

    return history, best, best_ep


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED)
    teacher_logits = load_teacher(ROOT / args.teacher)

    n_p = N_IN * args.h + args.h + args.h * N_OUT + N_OUT
    print(f"Step 621 — GLNN distillation on Imagenette (MLP student)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  h={args.h}  params≈{n_p:,}")
    print(f"  T={T}  λ={LAM}  teacher={'loaded' if teacher_logits is not None else 'NOT FOUND'}\n")

    results = {}
    ref_best = None

    for label, use_teacher in [("Ref_mlp", False), ("Distill_mlp", True)]:
        if use_teacher and teacher_logits is None:
            print(f"  {label}: skipped (no teacher)"); continue
        tl = teacher_logits if use_teacher else None
        model = make_mlp()
        desc = f"MLP h={args.h} scratch" if not use_teacher else f"MLP h={args.h} + GLNN T={T} λ={LAM}"
        print(f"{'─'*55}\n{label}: {desc}")
        t0 = time.time()
        history, best, best_ep = train(model, tr, va, tl)
        elapsed = time.time() - t0
        if label == "Ref_mlp": ref_best = best
        delta = best - (ref_best or 0)
        print(f"  → best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")
        results[label] = {"label": desc, "h": args.h, "n_params": n_p,
                          "best": best, "best_ep": best_ep,
                          "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*55}")
    print(f"STEP 621 SUMMARY — GLNN distillation MLP student (Imagenette)")
    for k, r in results.items():
        print(f"  {k:<14} best={r['best']:.4f}  Δ={r['delta_vs_ref']*100:+.2f}pp")
    if "Ref_mlp" in results and "Distill_mlp" in results:
        d = results["Distill_mlp"]["delta_vs_ref"]
        verdict = "STRONG (≥+0.5pp)" if d >= 0.005 else ("MEDIUM (+δ)" if d > 0 else "KILLED (no benefit)")
        print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
