"""Extract augmented VGG16 features from CIFAR-10 (hflip pass).

PURPOSE
=======
Creates data/store_cifar10_aug.h5 containing 2× the training samples:
  Pass 0 — original (Resize 224)               [50,000 samples]
  Pass 1 — horizontal flip + Resize 224        [50,000 samples]
  ──────────────────────────────────────────────────────────────
  Total train: 100,000 samples  (~2× original)

Val split is always extracted without augmentation (clean eval).

Mirrors the Imagenette augmentation in extract_features_augmented.py.
Hflip is safe for CIFAR-10: image classes are position-invariant (cars,
planes, birds, etc.) and hflip never removes the subject from frame.

Output: data/store_cifar10_aug.h5  (same schema as store_cifar10.h5)

Usage:
    python scripts/extract_cifar10_vgg16_features_aug.py --device mps
    python scripts/extract_cifar10_vgg16_features_aug.py --device cuda

Runtime: ~10-20 min on MPS (2× pass, 50K images × 2 each).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import h5py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="auto")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--output",     default="data/store_cifar10_aug.h5")
parser.add_argument("--data_dir",   default="data")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

OUT_PATH = ROOT / args.output


def make_extractor() -> nn.Module:
    vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1")
    model = nn.Sequential(vgg.features, vgg.avgpool, nn.Flatten())
    model.eval().to(DEVICE)
    return model


_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRANSFORMS = {
    "original": T.Compose([T.Resize((224, 224)), T.ToTensor(), _NORM]),
    "hflip":    T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM]),
}


def extract_pass(feat_extractor, ds, split_name: str, pass_name: str):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    n_total = len(ds)
    feats_out  = np.zeros((n_total, 25088), dtype=np.float32)
    labels_out = np.zeros((n_total,),       dtype=np.int64)
    soft_out   = np.zeros((n_total, 10),    dtype=np.float32)

    t0 = time.time()
    cursor = 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x    = x.to(DEVICE)
            feat = feat_extractor(x)   # [B, 25088]
            bs   = feat.shape[0]
            feats_out[cursor:cursor+bs]  = feat.cpu().numpy().astype(np.float32)
            labels_out[cursor:cursor+bs] = y.numpy().astype(np.int64)
            for i in range(bs):
                soft_out[cursor+i, y[i]] = 1.0
            cursor += bs
            if bi % 50 == 0:
                dt   = time.time() - t0
                rate = cursor / max(dt, 1e-6)
                eta  = (n_total - cursor) / max(rate, 1e-6)
                print(f"  [{split_name}/{pass_name}] {cursor}/{n_total}  "
                      f"{rate:.0f} img/s  ETA {eta:.0f}s", flush=True)

    print(f"  [{split_name}/{pass_name}] DONE {cursor} samples  "
          f"{time.time()-t0:.0f}s")
    return feats_out, soft_out, labels_out


def main():
    print(f"Device: {DEVICE}")
    feat_extractor = make_extractor()

    # ---------- Train: 2 passes (original + hflip) ----------
    tr_feats_all, tr_soft_all, tr_labels_all = [], [], []

    for pass_name, tfm in TRANSFORMS.items():
        print(f"\n--- train pass: {pass_name} ---")
        ds = torchvision.datasets.CIFAR10(
            root=str(ROOT / args.data_dir), train=True, download=False, transform=tfm)
        f, s, l = extract_pass(feat_extractor, ds, "train", pass_name)
        tr_feats_all.append(f)
        tr_soft_all.append(s)
        tr_labels_all.append(l)

    tr_feats  = np.concatenate(tr_feats_all,  axis=0)   # [100000, 25088]
    tr_soft   = np.concatenate(tr_soft_all,   axis=0)
    tr_labels = np.concatenate(tr_labels_all, axis=0)
    print(f"\nTrain combined: {tr_feats.shape[0]} samples")

    # ---------- Val: original only (clean eval) ----------
    print("\n--- val pass: original ---")
    val_tfm = TRANSFORMS["original"]
    val_ds = torchvision.datasets.CIFAR10(
        root=str(ROOT / args.data_dir), train=False, download=False, transform=val_tfm)
    va_feats, va_soft, va_labels = extract_pass(feat_extractor, val_ds, "val", "original")

    # ---------- Write HDF5 ----------
    OUT_PATH.parent.mkdir(exist_ok=True)
    print(f"\nWriting HDF5: {OUT_PATH}")
    with h5py.File(OUT_PATH, "w") as f:
        g_tr = f.create_group("train")
        g_tr.create_dataset("features",    data=tr_feats,  compression="gzip", compression_opts=1)
        g_tr.create_dataset("soft_labels", data=tr_soft,   compression="gzip", compression_opts=1)
        g_tr.create_dataset("labels",      data=tr_labels)
        g_va = f.create_group("val")
        g_va.create_dataset("features",    data=va_feats,  compression="gzip", compression_opts=1)
        g_va.create_dataset("soft_labels", data=va_soft,   compression="gzip", compression_opts=1)
        g_va.create_dataset("labels",      data=va_labels)

    size_gb = OUT_PATH.stat().st_size / 1024**3
    print(f"File size: {size_gb:.2f} GB  ({tr_feats.shape[0]} train + {va_feats.shape[0]} val)")
    print("DONE.")


if __name__ == "__main__":
    main()
