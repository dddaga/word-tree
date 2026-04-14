"""Step 612: K_iter knowledge distillation on efficiency config.

MOTIVATION
==========
Step127 distillation (K=12 teacher, K=6 student with KL loss) hurt vs scratch
on an older config. This retests on the confirmed efficiency config (N=2048,
D=16, K_hh=2) with T=2 softmax temperature and varying alpha.

The teacher (Ref: K_iter=12) is trained first in this same script run, then
loaded for distillation. The student copies teacher's soft logits via KL
divergence with temperature T=2.

CONFIGS (N=2048, D=16, K_hh=2, AH=1.0, 75ep, 50% data — Tier-1)
==================================================================
  Ref            : K_iter=12, train from scratch → becomes distillation teacher
  A_scratch_k3   : K_iter=3 from scratch (no distillation baseline)
  B_distill_k3_a50: K_iter=3, loss=0.5*CE + 0.5*KL(student||teacher)
  C_distill_k3_a30: K_iter=3, alpha=0.3 (more CE weight)
  D_distill_k5_a50: K_iter=5, alpha=0.5 (less aggressive compression)

Distillation loss:
  L = (1-alpha)*CE(logits, labels) + alpha*KL(log_softmax(s/T) || softmax(t/T)) * T^2

Note: Ref is always run first to produce the teacher checkpoint.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_resonant         import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory  import SGNNET_AntiHebbian
from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs
from src.training.dataset              import make_loaders

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. 'Ref' runs automatically first if any distill config is selected.")
args = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25
ALPHA_AHEBB = 1.0; ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

DISTILL_T = 2.0   # softmax temperature for KL distillation

TEACHER_CKPT = ROOT / "results" / "train_step612_teacher.pt"
OUT_PATH     = ROOT / "results" / "train_step612_kiter_distill_efficiency.json"

# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------
CONFIGS = {
    "Ref":              {"k_iter": 12, "distill": False, "alpha": 0.0,
                         "label": "Ref   K_iter=12 teacher (scratch)"},
    "A_scratch_k3":     {"k_iter": 3,  "distill": False, "alpha": 0.0,
                         "label": "A     K_iter=3 scratch (distill baseline)"},
    "B_distill_k3_a50": {"k_iter": 3,  "distill": True,  "alpha": 0.5,
                         "label": "B     K_iter=3, distill alpha=0.5"},
    "C_distill_k3_a30": {"k_iter": 3,  "distill": True,  "alpha": 0.3,
                         "label": "C     K_iter=3, distill alpha=0.3"},
    "D_distill_k5_a50": {"k_iter": 5,  "distill": True,  "alpha": 0.5,
                         "label": "D     K_iter=5, distill alpha=0.5"},
}

# ---------------------------------------------------------------------------
# Data (cached, 50%)
# ---------------------------------------------------------------------------
_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = Subset(tr_full.dataset, idx.tolist())
        tr = DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model(k_iter: int, seed_offset: int = 0):
    torch.manual_seed(SEED + seed_offset)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=k_iter,
                              K_local=K_l, K_random=K_r, n_groups=ng,
                              norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def compute_flops(k_iter: int) -> int:
    seed     = N * K_IN * D
    per_step = N * K_HH * D * 2 + N * D + N * D * 2
    routing  = k_iter * per_step
    readout  = N * N_OUT * D
    return seed + routing + readout


# ---------------------------------------------------------------------------
# Standard training (no distillation)
# ---------------------------------------------------------------------------
def train_standard(model, n_epochs: int) -> list:
    tr, va = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)

    top1_hist = []

    def _log(m):
        ep = m["epoch"] + 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

    history = trainer.train(n_epochs=n_epochs, log_fn=_log)
    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    return top1_hist


# ---------------------------------------------------------------------------
# Distillation training loop
# ---------------------------------------------------------------------------
class DistillationTrainer:
    """Wraps a student model and a frozen teacher for KL distillation."""

    def __init__(self, student: nn.Module, teacher: nn.Module,
                 alpha: float, temperature: float, device: torch.device):
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.T = temperature
        self.device = device

    def _distill_loss(self, s_logits: torch.Tensor,
                      t_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(s_logits, labels)
        s_log = F.log_softmax(s_logits / self.T, dim=-1)
        t_soft = F.softmax(t_logits / self.T, dim=-1)
        kl_loss = F.kl_div(s_log, t_soft, reduction="batchmean") * (self.T ** 2)
        return (1 - self.alpha) * ce_loss + self.alpha * kl_loss

    def train_epoch(self, tr_loader, optimizer) -> float:
        self.student.train()
        self.teacher.eval()
        correct = 0; total = 0
        for x, y in tr_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                t_logits = self.teacher(x)
            s_logits = self.student(x)
            loss = self._distill_loss(s_logits, t_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (s_logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / total

    @torch.no_grad()
    def eval_epoch(self, va_loader) -> float:
        self.student.eval()
        correct = 0; total = 0
        for x, y in va_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.student(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / total


def train_with_distill(student: nn.Module, teacher: nn.Module,
                       alpha: float, n_epochs: int) -> list:
    """Full distillation training loop with cosine LR."""
    tr, va = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)
    lr = kw.get("lr", 1e-3)
    wd = kw.get("weight_decay", 1e-4)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    dt = DistillationTrainer(student, teacher, alpha, DISTILL_T, DEVICE)
    top1_hist = []

    for ep in range(1, n_epochs + 1):
        dt.train_epoch(tr, optimizer)
        val_acc = dt.eval_epoch(va)
        top1_hist.append(round(val_acc, 4))
        scheduler.step()
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val={val_acc:.4f}", flush=True)

    return top1_hist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []

    # Determine active configs; always run Ref first if any distill config is requested
    all_keys = list(CONFIGS.keys())
    if cfg_filter:
        active_keys = cfg_filter
        needs_teacher = any(CONFIGS[k]["distill"] for k in active_keys if k in CONFIGS)
        if needs_teacher and "Ref" not in active_keys:
            active_keys = ["Ref"] + active_keys
    else:
        active_keys = all_keys

    print(f"\n{'='*70}")
    print(f"Step 612 — K_iter distillation on efficiency config")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%  T={DISTILL_T}  Tier-1")
    print(f"Running: {active_keys}")
    print(f"{'='*70}\n")

    get_loaders()
    results = {}
    teacher_model: nn.Module | None = None

    for i, key in enumerate(active_keys):
        if key not in CONFIGS:
            print(f"WARNING: unknown config '{key}', skipping.")
            continue
        cfg = CONFIGS[key]
        model = make_model(cfg["k_iter"], seed_offset=i).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        flops = compute_flops(cfg["k_iter"])

        print(f"\n{'─'*60}")
        print(f"Config {key}: {cfg['label']}")
        print(f"  k_iter={cfg['k_iter']}  distill={cfg['distill']}  "
              f"alpha={cfg['alpha']}  params={n_params:,}  flops={flops/1e6:.2f}M")
        print(f"{'─'*60}")

        t0 = time.time()

        if cfg["distill"]:
            if teacher_model is None:
                # Load from saved checkpoint
                if TEACHER_CKPT.exists():
                    print(f"  Loading teacher from {TEACHER_CKPT}")
                    teacher_model = make_model(12).to(DEVICE)
                    teacher_model.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE))
                else:
                    raise RuntimeError(
                        "Teacher checkpoint not found. Run Ref first or include 'Ref' in --configs.")
            teacher_model.eval()
            top1_hist = train_with_distill(model, teacher_model, cfg["alpha"], EPOCHS)
        else:
            top1_hist = train_standard(model, EPOCHS)
            if key == "Ref":
                # Save teacher checkpoint
                TEACHER_CKPT.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), TEACHER_CKPT)
                teacher_model = model
                print(f"  Teacher checkpoint saved → {TEACHER_CKPT}")

        elapsed = time.time() - t0
        best_idx = int(np.argmax(top1_hist))
        best_top1 = max(top1_hist)

        results[key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_in": K_IN,
            "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5,
            "label": cfg["label"],
            "k_iter": cfg["k_iter"],
            "distill": cfg["distill"],
            "distill_alpha": cfg["alpha"],
            "distill_T": DISTILL_T if cfg["distill"] else None,
            "top1_history": top1_hist,
            "best_top1": best_top1,
            "best_epoch": best_idx + 1,
            "flops": flops,
            "flops_M": round(flops / 1e6, 2),
            "n_params": n_params,
            "elapsed_s": round(elapsed, 1),
        }

        ref_best = results.get("Ref", {}).get("best_top1", 0)
        scratch_best = results.get("A_scratch_k3", {}).get("best_top1", 0)
        vs_ref = best_top1 - ref_best if ref_best > 0 else 0
        vs_scratch = (best_top1 - scratch_best
                      if scratch_best > 0 and key not in ("Ref", "A_scratch_k3") else 0)
        print(f"\n  best={best_top1:.4f} @ ep{best_idx+1}  "
              f"vs_Ref={vs_ref:+.4f}  "
              f"{'vs_scratch=' + f'{vs_scratch:+.4f}' if vs_scratch != 0 else ''}  "
              f"{elapsed/60:.1f}min")

        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    ref_best    = results.get("Ref",          {}).get("best_top1", 0)
    scratch_best = results.get("A_scratch_k3", {}).get("best_top1", 0)
    print(f"\n{'='*70}")
    print(f"STEP 612 SUMMARY — K_iter distillation")
    print(f"{'='*70}")
    print(f"{'Key':22s}  {'K':>3}  {'α':>4}  {'best':>7}  "
          f"{'vs_Ref':>8}  {'vs_scratch':>11}  {'FLOPs_M':>8}")
    print(f"{'─'*75}")
    for key, r in results.items():
        vs_r = r["best_top1"] - ref_best if ref_best > 0 else 0
        vs_s = (r["best_top1"] - scratch_best
                if scratch_best > 0 and key not in ("Ref", "A_scratch_k3") else float("nan"))
        vs_s_str = f"{vs_s:>+.4f}" if not (isinstance(vs_s, float) and vs_s != vs_s) else "    —   "
        print(f"{key:22s}  {r['k_iter']:>3}  {r['distill_alpha']:>4.1f}  "
              f"{r['best_top1']:.4f}  {vs_r:>+.4f}  {vs_s_str:>11}  {r['flops_M']:>8.2f}")
    print(f"\n→ {OUT_PATH}")
    if TEACHER_CKPT.exists():
        print(f"Teacher ckpt → {TEACHER_CKPT}")


if __name__ == "__main__":
    main()
