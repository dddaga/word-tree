"""Step 611: K_iter warm transfer at efficiency config.

MOTIVATION
==========
step163 showed K=12->K=8 warm transfer gave +7.82pp at N=1024 D=16 K_hh=8.
step173 HURT at N=2048 D=32 K_hh=4 (-0.23pp).
step610 annealing running on mini_cpu.

Hypothesis: warm transfer (large-K teacher -> small-K student) may work at
efficiency config (N=2048 D=16 K_hh=2). K_iter is just a loop counter --
teacher and student share identical weights. Mid-training swap
`model.m.base.K_iter` from teacher_k to student_k at SWITCH_EP.

Configs:
  Ref       : scratch K_iter=5 all 75ep (control)
  A_12to5   : teacher K=12 -> student K=5 at ep40
  B_12to3   : teacher K=12 -> student K=3 at ep40
  C_8to5    : teacher K=8  -> student K=5 at ep40
  D_16to5   : teacher K=16 -> student K=5 at ep40

Scale: N=2048 D=16 K_hh=2, 75ep 50% data (Tier-1)
Ref: step197=93.96% (scratch K=5 Tier-1)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

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
SWITCH_EP       = 40
FRAC_DATA       = 0.5
BATCH           = 128
SEED            = 42

CONFIGS = {
    "Ref":     {"teacher_k": 5,  "student_k": 5,  "switch": False},
    "A_12to5": {"teacher_k": 12, "student_k": 5,  "switch": True},
    "B_12to3": {"teacher_k": 12, "student_k": 3,  "switch": True},
    "C_8to5":  {"teacher_k": 8,  "student_k": 5,  "switch": True},
    "D_16to5": {"teacher_k": 16, "student_k": 5,  "switch": True},
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
    res = SGNNET_Resonant(
        base=sw, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
    ).to(device)
    return SGNNET_AntiHebbian(base=res, alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(device)


def run_config(label, cfg, device, tr, va):
    teacher_k = cfg["teacher_k"]
    student_k = cfg["student_k"]
    do_switch = cfg["switch"]

    print(f"\n{'--'*30}")
    print(f"Config {label}: teacher_k={teacher_k} student_k={student_k}")
    model = make_model(device, teacher_k)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}")

    top1h = []

    def _log(m):
        ep = m['epoch'] + 1
        top1h.append(round(m.get('val_top1', 0.0), 4))
        if ep % 10 == 0:
            print(f"  ep{ep:3d}  val={top1h[-1]:.4f}", flush=True)
        if do_switch and ep == SWITCH_EP:
            model.m.base.K_iter = student_k
            print(f"  *** ep{ep}: K_iter {teacher_k} -> {student_k} ***", flush=True)

    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=str(device), **kw)
    trainer.train(n_epochs=EPOCHS, log_fn=_log)

    best = max(top1h) if top1h else 0.0
    bep  = int(np.argmax(top1h)) + 1 if top1h else 0
    print(f"  best={best:.4f} @ ep{bep}")
    return {
        "label": label, "teacher_k": teacher_k, "student_k": student_k,
        "switch": do_switch, "switch_ep": SWITCH_EP if do_switch else None,
        "n_params": n_params, "top1_best": best, "best_epoch": bep,
        "top1_last": top1h[-1] if top1h else None, "top1_history": top1h,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 611: K_iter warm transfer")
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--configs", default="", help="Comma-separated keys")
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    selected = [k.strip() for k in args.configs.split(",") if k.strip()] or list(CONFIGS.keys())

    print(f"\n{'='*70}")
    print(f"Step 611 -- K_iter Warm Transfer (Tier-1 {args.epochs}ep)")
    print(f"N={N} D={D} K_hh={K_HH}  switch_ep={SWITCH_EP}  configs={selected}")
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
    va = torch.utils.data.DataLoader(val_ds,    batch_size=BATCH, shuffle=False)
    print(f"Data: {n_train}/{len(train_ds)} train, {len(val_ds)} val")

    results = {}
    for label in selected:
        results[label] = run_config(label, CONFIGS[label], device, tr, va)

    ref_acc = results.get("Ref", {}).get("top1_best")
    print(f"\n{'='*70}")
    print(f"{'Label':<12}  {'teacher_k':>9}  {'student_k':>9}  {'best_val':>9}  {'delta':>8}  {'best_ep':>7}")
    print(f"{'-'*70}")
    for label, r in sorted(results.items(), key=lambda x: -x[1]["top1_best"]):
        delta = f"{r['top1_best']-ref_acc:+.4f}" if ref_acc else "--"
        print(f"  {label:<12}  {r['teacher_k']:>9}  {r['student_k']:>9}  "
              f"{r['top1_best']:>9.4f}  {delta:>8}  ep{r['best_epoch']:>3}")

    out_data = {
        "config": {"N": N, "D": D, "K_hh": K_HH, "alpha_ahebb": ALPHA_AHEBB,
                   "epochs": args.epochs, "switch_ep": SWITCH_EP, "device": str(device)},
        "results": results,
    }
    out_path = args.output or str(ROOT / "results" / "train_step611_kiter_warm_transfer.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
