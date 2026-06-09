"""cnnc_step002 — T0 scout: iso-MAC ablation (follow-up to cnnc_step001).

step001 showed B_multibranch (110.9M MACs) and C_crelu (94.1M) lose only
~2pp to Ref (224.0M MACs) at iso-params. Here B/C are scaled UP to matched
MACs (~224M +/-10%): B_isomac 236.2M / C_isomac 233.3M / Ref_deep 220.4M
control. Params may exceed Ref's 483K — MACs are the controlled variable.

Ref NOT rerun: step001 Ref best=0.7557 reused (identical seed 42, 20ep,
50% data, same pipeline/augmentation/optimizer). dRef reported vs 0.7557.

Data: raw imagenette2-320 images (LOCAL ONLY — 5060ti has .h5 files only,
no raw images). Run on mini_mps.

# CUDA-5060ti-validated  (pin_memory=True + non_blocking=True present; plain
# torchvision CNN path, no SGNNET seed kernels apply to this line)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts.cnn_compress.models_step002 import (
    CONFIG_DESCS, build_model, count_macs)

REF_ACC, REF_PARAMS, REF_MACS = 0.7557, 483_050, 224.0   # cnnc_step001 Ref

parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="auto")
parser.add_argument("--epochs",     type=int, default=20)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--batch",      type=int, default=32)
parser.add_argument("--data_img",   default="data/imagenette2-320")
parser.add_argument("--configs",    default="B_isomac,C_isomac,Ref_deep")
parser.add_argument("--tag",        default="t0")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)
SEED = args.seed
EPOCHS = 1 if args.smoke_test else args.epochs
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / "cnn_compress" / \
    f"cnnc_step002_{args.tag}_seed{SEED}__{SLOT}.json"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for image, label in loader:
        image = image.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        correct += (model(image).argmax(1) == label).sum().item()
        total += label.size(0)
    return correct / total


def main():
    img_root = ROOT / args.data_img
    if not img_root.exists():
        print(f"ERROR: {img_root} not found (raw imagenette is mini-local only)")
        sys.exit(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_full = ImageFolder(str(img_root / "train"), transform=TRAIN_TF)
    val_ds = ImageFolder(str(img_root / "val"), transform=VAL_TF)
    n = len(train_full)
    g = torch.Generator().manual_seed(SEED)
    sub = torch.randperm(n, generator=g)[: n // 2].tolist()    # T0: 50% data
    if args.smoke_test:
        sub = sub[:64]
        val_ds = Subset(val_ds, list(range(64)))
    train_ds = Subset(train_full, sub)

    pin = DEVICE.type == "cuda"
    nw = 4 if DEVICE.type == "cuda" else 2
    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=nw, pin_memory=True if pin else False,
                    persistent_workers=nw > 0)
    va = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    num_workers=nw, pin_memory=True if pin else False,
                    persistent_workers=nw > 0)

    keys = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIG_DESCS]
    assert keys, f"no valid configs in {args.configs!r}; valid: {list(CONFIG_DESCS)}"

    print(f"\n{'=' * 72}\ncnnc_step002 T0 iso-MAC  device={DEVICE}  epochs={EPOCHS}  "
          f"seed={SEED}  train={len(train_ds)}  val={len(val_ds)}\n{'=' * 72}")
    print(f"  Ref (step001, not rerun): params={REF_PARAMS:,}  MACs={REF_MACS:.1f}M"
          f"  acc={REF_ACC:.4f}")
    print(f"  {'config':<10} {'params':>9}  {'MACs(M)':>8}  desc")
    for k in keys:
        torch.manual_seed(SEED)
        m = build_model(k)
        n_p = sum(p.numel() for p in m.parameters())
        macs = count_macs(m) / 1e6
        assert abs(macs - REF_MACS) / REF_MACS <= 0.10, \
            f"{k}: {macs:.1f}M outside 224M +/-10%"
        print(f"  {k:<10} {n_p:>9,}  {macs:>8.1f}  {CONFIG_DESCS[k]}")
        del m

    results = {"Ref": {"desc": "step001 reference (reused, not rerun)",
                       "n_params": REF_PARAMS, "macs_M": REF_MACS,
                       "best": REF_ACC, "delta_vs_ref": 0.0}}
    for key in keys:
        torch.manual_seed(SEED)
        model = build_model(key)
        n_p = sum(p.numel() for p in model.parameters())
        macs = count_macs(model) / 1e6
        model = model.to(DEVICE)
        print(f"\n{'-' * 60}\n{key}: {CONFIG_DESCS[key]}\n"
              f"  params={n_p:,}  MACs={macs:.1f}M")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS, eta_min=1e-5)
        best, best_ep, ep_times = 0.0, 0, []
        for ep in range(EPOCHS):
            t_ep = time.time()
            model.train()
            for image, label in tr:
                image = image.to(DEVICE, non_blocking=True)
                label = label.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                loss = F.cross_entropy(model(image), label)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            ep_times.append(time.time() - t_ep)
            val_acc = evaluate(model, va)
            if val_acc > best:
                best, best_ep = val_acc, ep + 1
            print(f"  ep{ep + 1:3d}  val={val_acc:.4f}  best={best:.4f}  "
                  f"{ep_times[-1]:.0f}s", flush=True)

        delta = best - REF_ACC
        print(f"  -> best={best:.4f} @ep{best_ep}  dRef={delta * 100:+.2f}pp  "
              f"epoch={np.mean(ep_times):.0f}s")
        results[key] = {
            "desc": CONFIG_DESCS[key], "n_params": n_p, "macs_M": round(macs, 1),
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4),
            "epoch_time_s": round(float(np.mean(ep_times)), 1),
            "acc_per_Mmac": round(best / macs, 4),
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 72}\nCNNC_STEP002 T0 SUMMARY (iso-MAC ~224M, 20ep, 50% data, "
          f"dRef vs step001 Ref {REF_ACC:.4f})\n{'=' * 72}")
    print(f"  {'config':<10} {'params':>9}  {'MACs(M)':>8}  {'acc':>7}  "
          f"{'dRef':>9}  {'s/ep':>5}")
    for k, r in results.items():
        print(f"  {k:<10} {r['n_params']:>9,}  {r['macs_M']:>8.1f}  "
              f"{r['best']:>7.4f}  {r['delta_vs_ref'] * 100:>+8.2f}pp  "
              f"{r.get('epoch_time_s', 0):>5.0f}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
