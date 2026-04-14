"""Step 127: Progressive K_iter distillation — 50% FLOPs reduction.

MOTIVATION
==========
K_iter=12 dominates SGNNET FLOPs. If we can train a K_iter=6 model that matches
K_iter=12 accuracy via knowledge distillation, we halve routing FLOPs.

Protocol:
  1. Train teacher (K_iter=12) to convergence
  2. Distill into student (K_iter=6 or 8) using:
     loss = (1-α)*task_loss + α*MSE(student_logits, teacher_logits.detach())

CONFIGS (N=1024, D=32, K_hh=4, K_iter=varies, K_in=50, AH=1.0, 50%/75ep)
==========================================================================
  Ref : K_iter=12 teacher (trained first, used for distillation)
  A   : K_iter=6 from scratch (no distillation — baseline)
  B   : K_iter=6 distilled from Ref, α=0.5
  C   : K_iter=6 distilled from Ref, α=0.7 (more task loss)
  D   : K_iter=8 distilled from Ref, α=0.5 (less aggressive reduction)

To reproduce:
    python -u scripts/train_step127_kiter_distillation.py --device mps
    python -u scripts/train_step127_kiter_distillation.py --device mps --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. B,D). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 32; K_IN = 50; K_HH = 4
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0


@dataclass
class Config:
    key: str
    label: str
    k_iter: int
    distill_alpha: float
    needs_teacher: bool = False

CONFIGS = [
    Config("Ref", "Ref  K_iter=12 teacher",              12, 0.0, False),
    Config("A",   "A    K_iter=6 from scratch",            6, 0.0, False),
    Config("B",   "B    K_iter=6 distilled α=0.5",         6, 0.5, True),
    Config("C",   "C    K_iter=6 distilled α=0.7",         6, 0.7, True),
    Config("D",   "D    K_iter=8 distilled α=0.5",         8, 0.5, True),
]


_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(k_iter: int, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    tk.pop("K_in", None); tk.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=K_IN, K_iter=k_iter, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def compute_flops(k_iter: int) -> int:
    seed     = N * K_IN * D
    per_step = N * K_HH * D * 2 + N * D + N * D * 2
    routing  = k_iter * per_step
    readout  = N * N_OUT * D
    return seed + routing + readout


def train_standard(model, device, n_epochs):
    tr, va = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=device, **kw)
    return trainer.train(n_epochs=n_epochs)


def train_distilled(student, teacher, device, n_epochs, alpha):
    """Distillation: combined task + MSE(student, teacher) loss."""
    tr, va = get_loaders()
    kw = trainer_kwargs(N, n_epochs=n_epochs)

    optimizer = torch.optim.AdamW(student.parameters(), lr=kw.get("lr", 1e-3),
                                   weight_decay=kw.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    teacher.eval()
    history = []

    for ep in range(1, n_epochs + 1):
        student.train()
        total_loss = 0
        for batch in tr:
            x = batch[0].to(device)
            soft_labels = batch[1].to(device)

            optimizer.zero_grad()
            student_logits = student(x)

            with torch.no_grad():
                teacher_logits = teacher(x)

            task_loss = criterion(student_logits, soft_labels)
            distill_loss = F.mse_loss(student_logits, teacher_logits)
            loss = (1 - alpha) * task_loss + alpha * distill_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        student.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in va:
                x = batch[0].to(device)
                labels = batch[2].to(device)
                out = student(x)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_top1 = correct / total
        scheduler.step(total_loss)
        history.append({"val_top1": val_top1, "train_loss": total_loss / len(tr)})

        if ep % 10 == 0 or ep == n_epochs:
            print(f"    ep={ep:3d}  val_top1={val_top1:.4f}  loss={total_loss/len(tr):.4f}")

    return history


def main():
    print(f"\n{'='*70}")
    print(f"Step 127 — K_iter Distillation")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_in={K_IN}  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    for c in CONFIGS:
        flops = compute_flops(c.k_iter)
        print(f"  {c.key:4s}  K={c.k_iter:2d}  α={c.distill_alpha:.1f}  "
              f"FLOPs={flops/1e6:.2f}M  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step127_kiter_distillation.json"
    teacher_model = None

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    needs_teacher = any(cfg.needs_teacher for _, cfg in active_configs)
    has_ref = any(cfg.key == "Ref" for _, cfg in active_configs)
    if needs_teacher and not has_ref:
        active_configs.insert(0, (0, CONFIGS[0]))

    for i, cfg in active_configs:
        model = make_model(cfg.k_iter, seed_offset=i).to(DEVICE)
        n_params = count_params(model)
        flops = compute_flops(cfg.k_iter)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  K_iter={cfg.k_iter}  params={n_params:,}  FLOPs={flops/1e6:.2f}M")
        print(f"{'─'*60}")

        t0 = time.time()
        if cfg.needs_teacher and teacher_model is not None:
            print(f"  Distilling from teacher (α={cfg.distill_alpha})")
            history = train_distilled(model, teacher_model, DEVICE, EPOCHS, cfg.distill_alpha)
        else:
            history = train_standard(model, DEVICE, EPOCHS)
        elapsed = time.time() - t0

        if cfg.key == "Ref":
            teacher_model = model
            teacher_model.eval()

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_hh": K_HH, "K_iter": cfg.k_iter,
            "distill_alpha": cfg.distill_alpha,
            "flops": flops, "flops_M": round(flops / 1e6, 2),
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0
        print(f"\n  top1={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"FLOPs={flops/1e6:.2f}M  elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 127 SUMMARY — K_iter Distillation")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    ref_flops = results.get("Ref", {}).get("flops_M", 0)
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        flops_pct = r["flops_M"] / ref_flops * 100 if ref_flops > 0 else 100
        print(f"  {key:4s}  K={r['K_iter']:2d}  α={r['distill_alpha']:.1f}  "
              f"FLOPs={r['flops_M']:.2f}M ({flops_pct:.0f}%)  "
              f"top1={r['top1_best']:.4f}  vs_Ref={vs:+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
