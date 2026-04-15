"""Extract VGG16 conv features from CIFAR-100 for cross-dataset SGNNET training.

PURPOSE
=======
Paper-blocking requirement (baselines_needed.md): cross-dataset validation of SGNNET
as a general-purpose classification head. Target: train efficiency-config SGNNET on
VGG16 features of CIFAR-100 (not ImageNet-trained Imagenette data), report accuracy.

OUTPUT FORMAT (matches src/training/dataset.py H5Dataset):
  data/store_cifar100.h5:
    /train/features     [50000, 25088] float32
    /train/soft_labels  [50000, 100]    float32 (one-hot)
    /train/labels       [50000]        int64
    /val/features       [10000, 25088] float32
    /val/soft_labels    [10000, 100]    float32
    /val/labels         [10000]        int64

Requires torchvision. Runs on MPS preferably; CPU acceptable (30-60 min).

Usage:
    d_env/bin/python3 scripts/extract_cifar100_vgg16_features.py --device mps
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
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--output", default="data/store_cifar100.h5")
parser.add_argument("--data_dir", default="data")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

OUT_PATH = ROOT / args.output


def main():
    print(f"Device: {DEVICE}")

    # Transforms: resize to 224, ImageNet normalization
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading CIFAR-100 datasets...")
    train_ds = torchvision.datasets.CIFAR100(
        root=str(ROOT / args.data_dir), train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.CIFAR100(
        root=str(ROOT / args.data_dir), train=False, download=True, transform=transform)
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    print("Loading VGG16 pretrained...")
    vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1")
    # Feature extractor: features + avgpool + flatten (25088-dim matches Imagenette)
    feat_extractor = nn.Sequential(vgg.features, vgg.avgpool, nn.Flatten())
    feat_extractor.eval().to(DEVICE)

    def extract(ds, split_name):
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        n_total = len(ds)
        feats_out = np.zeros((n_total, 25088), dtype=np.float32)
        labels_out = np.zeros((n_total,), dtype=np.int64)
        n_classes  = len(ds.classes) if hasattr(ds, 'classes') else 100
        soft_out   = np.zeros((n_total, n_classes), dtype=np.float32)

        t0 = time.time()
        cursor = 0
        with torch.no_grad():
            for bi, (x, y) in enumerate(loader):
                x = x.to(DEVICE)
                feats = feat_extractor(x)   # [B, 25088]
                bs = feats.shape[0]
                feats_out[cursor:cursor+bs] = feats.cpu().numpy().astype(np.float32)
                labels_out[cursor:cursor+bs] = y.numpy().astype(np.int64)
                for i in range(bs):
                    soft_out[cursor+i, y[i]] = 1.0
                cursor += bs
                if bi % 50 == 0:
                    dt = time.time() - t0
                    rate = cursor / max(dt, 1e-6)
                    eta = (n_total - cursor) / max(rate, 1e-6)
                    print(f"  [{split_name}] {cursor}/{n_total}  {rate:.0f} img/s  ETA {eta:.0f}s", flush=True)
        print(f"  [{split_name}] DONE {cursor} samples in {time.time()-t0:.0f}s")
        return feats_out, soft_out, labels_out

    print("\nExtracting train features...")
    tr_feats, tr_soft, tr_labels = extract(train_ds, "train")
    print("\nExtracting val features...")
    va_feats, va_soft, va_labels = extract(val_ds, "val")

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
    print(f"File size: {OUT_PATH.stat().st_size / 1024**3:.2f} GB")
    print("DONE.")


if __name__ == "__main__":
    main()
