"""Step 612: K_iter KL distillation at efficiency config.

MOTIVATION
==========
step127 showed KL distillation HURT at N=1024 D=64 K_hh=8.
step611 confirmed warm transfer HURT at efficiency config (N=2048 D=16 K_hh=2).
This is the third P-KITER approach: use a trained K=5 teacher's soft logits
as a KL target for a K=3 student, training from scratch.

If distillation also hurts, the P-KITER section concludes:
  - Curriculum KILLED (step610)
  - Warm transfer KILLED (step611)
  - Distillation KILLED (step612, expected)

Confirming all three closes the K_iter reduction research direction.

Configs:
  Ref         : scratch K_iter=5, cross-entropy only (step197 = 93.96%)
  A_k3_ce     : scratch K_iter=3, cross-entropy only (step202 = 89.25% -- known floor)
  B_k3_distill: K_iter=3 student + KL(teacher K=5 logits, T=2), λ_kl=0.5
  C_k3_distill_strong: K_iter=3 + KL T=4 λ_kl=0.8 (stronger distillation)

Teacher: train K=5 for 40ep (warm-up phase), then use its soft logits.
Architecture: teacher and student share same model structure; only K_iter differs.

Scale: N=2048 D=16 K_hh=2, 75ep 50% data (Tier-1)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.experiment_config import trainer_kwargs
from src.training.trainer import Trainer
from src.training.dataset import H5Dataset

N_IN, N_CLASSES = 25088, 10
N, D            = 2048, 16
K_HH            = 2
K_IN            = 25
ALPHA_AHEBB     = 1.0
ALPHA_REFLECT   = 0.5
ALPHA_TURING    = 0.0
EPOCHS          = 75
TEACHER_EP      = 40   # epochs to train teacher before distillation starts
FRAC_DATA       = 0.5
BATCH           = 128
SEED            = 42

CONFIGS = {
    "Ref":              {"k_student": 5, "distill": False, "T": 1.0, "lam": 0.0},
    "A_k3_ce":          {"k_student": 3, "distill": False, "T": 1.0, "lam": 0.0},
    "B_k3_distill":     {"k_student": 3, "distill": True,  "T": 2.0, "lam": 0.5},
    "C_k3_distill_hard":{"k_student": 3, "distill": True,  "T": 4.0, "lam": 0.8},
}


def make_model(device, k_iter):
    torch.manual_seed(SEED)
    n_groups = max(8, N // 8)
    K_local  = max(1, K_HH - max(1, K_HH // 4))
    K_random = K_HH - K_local
    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_CLASSES, D=D, N_in=N_IN,
        K_in=K_IN, K_local=K_local, K_random=K_random,
        n_groups=n_groups, K_iter=k_iter,
        norm_mode="l2", encoding_mode="fourier",
    ).to(device)
    res = SGNNET_Resonant(base=sw, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=ALPHA_TURING, mode="dynamic_z_geo").to(device)
    return SGNNET_AntiHebbian(base=res, alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(device)


def run_config(label, cfg, device, tr, va):
    k_student  = cfg["k_student"]
    do_distill = cfg["distill"]
    T          = cfg["T"]
    lam        = cfg["lam"]

    print(f"\n{'--'*30}")
    print(f"Config {label}: k_student={k_student} distill={do_distill} T={T} lam={lam}")
    student = make_model(device, k_student)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  params={n_params:,}")

    if do_distill:
        # Build teacher K=5, train for TEACHER_EP, then provide soft targets
        teacher = make_model(device, k_iter=5)
        teacher_kw = trainer_kwargs(N, n_epochs=TEACHER_EP)
        teacher_trainer = Trainer(model=teacher, train_loader=tr, val_loader=va,
                                   device=str(device), **teacher_kw)
        print(f"  Training teacher K=5 for {TEACHER_EP}ep...")
        teacher_trainer.train(n_epochs=TEACHER_EP, log_fn=lambda m: (
            print(f"  [teacher] ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 10 == 0 else None
        ))
        teacher.eval()
        print(f"  Teacher trained. Starting distillation student K={k_student}...")
    else:
        teacher = None

    top1h = []
    opt   = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = torch.nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        student.train()
        if teacher: teacher.eval()
        for feats, soft_labels, labels in tr:
            feats  = feats.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            logits = student(feats)
            loss_ce = crit(logits, labels)
            if do_distill and teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(feats)
                loss_kl = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T ** 2)
                loss = (1 - lam) * loss_ce + lam * loss_kl
            else:
                loss = loss_ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
        sched.step()

        # Eval
        student.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, _, labels in va:
                feats  = feats.to(device); labels = labels.to(device)
                correct += (student(feats).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        val_top1 = round(correct / total, 4)
        top1h.append(val_top1)
        if (ep + 1) % 10 == 0:
            print(f"  ep{ep+1:3d}  val={val_top1:.4f}", flush=True)

    best = max(top1h) if top1h else 0.0
    bep  = int(np.argmax(top1h)) + 1 if top1h else 0
    print(f"  best={best:.4f} @ ep{bep}")
    return {
        "label": label, "k_student": k_student, "distill": do_distill,
        "T": T, "lam": lam, "n_params": n_params,
        "top1_best": best, "best_epoch": bep,
        "top1_last": top1h[-1] if top1h else None, "top1_history": top1h,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 612: K_iter distillation")
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--configs", default="", help="Comma-separated keys")
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    selected = [k.strip() for k in args.configs.split(",") if k.strip()] or list(CONFIGS.keys())

    print(f"\n{'='*70}")
    print(f"Step 612 -- K_iter KL Distillation at Efficiency Config (Tier-1 {args.epochs}ep)")
    print(f"N={N} D={D} K_hh={K_HH}  teacher_ep={TEACHER_EP}  configs={selected}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    train_ds = H5Dataset(str(ROOT / "data/store.h5"), split="train")
    val_ds   = H5Dataset(str(ROOT / "data/store.h5"), split="val")
    n_train  = int(len(train_ds) * FRAC_DATA)
    g  = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(train_ds), generator=g)[:n_train].tolist()
    train_sub = torch.utils.data.Subset(train_ds, idx)
    g2 = torch.Generator().manual_seed(SEED)
    tr = torch.utils.data.DataLoader(train_sub, batch_size=BATCH, shuffle=True, generator=g2)
    va = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    print(f"Data: {n_train}/{len(train_ds)} train, {len(val_ds)} val")

    results = {}
    for label in selected:
        results[label] = run_config(label, CONFIGS[label], device, tr, va)

    ref_acc = results.get("Ref", {}).get("top1_best")
    print(f"\n{'='*70}")
    print(f"{'Label':<22}  {'k_stud':>6}  {'distill':>7}  {'T':>4}  {'lam':>5}  {'best_val':>9}  {'delta':>8}  {'bep':>4}")
    print(f"{'-'*70}")
    for label, r in sorted(results.items(), key=lambda x: -x[1]["top1_best"]):
        delta = f"{r['top1_best']-ref_acc:+.4f}" if ref_acc else "--"
        print(f"  {label:<22}  {r['k_student']:>6}  {str(r['distill']):>7}  "
              f"{r['T']:>4}  {r['lam']:>5}  {r['top1_best']:>9.4f}  {delta:>8}  {r['best_epoch']:>4}")

    out_data = {
        "config": {"N": N, "D": D, "K_hh": K_HH, "alpha_ahebb": ALPHA_AHEBB,
                   "epochs": args.epochs, "teacher_ep": TEACHER_EP, "device": str(device)},
        "results": results,
    }
    out_path = args.output or str(ROOT / "results" / "train_step612_kiter_distill_efficiency.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
