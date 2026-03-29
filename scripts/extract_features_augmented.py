"""Extract VGG16 features with data augmentation to expand the training set.

Runs 2 passes over the imagenette train split with class-preserving transforms:
  Pass 0 — original (CenterCrop 224)              [9 469 samples]
  Pass 1 — horizontal flip + CenterCrop           [9 469 samples]
  ──────────────────────────────────────────────────────────────
  Total train: 18 938 samples  (~2× original)

Random crops are intentionally excluded: aggressive cropping can remove the
class-relevant subject from the frame, producing a corrupted soft label from
VGG16 (e.g., a "golf ball" crop showing only grass gets a wrong soft label).
Horizontal flip is safe — mirroring never loses the subject.

Val split is always extracted with the standard transform (no augmentation).
Augmenting validation would inflate scores — val stays clean.

Soft labels come from VGG16 output on each transformed image, so they
reflect the model's confidence on that specific crop/flip. Augmented
variants with poor crops get appropriately softer labels.

Output: data/store_aug.h5  (same schema as store.h5)
        data/manifest_aug.csv

Usage:
    python scripts/extract_features_augmented.py [--device auto|mps|cpu]
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import ImagenetteDataset, IMAGENETTE_CLASSES
from src.data.extractor import VGGExtractor
from src.data.store import TensorStore

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--out", default="data/store_aug.h5")
args = parser.parse_args()

if args.device == "auto":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = args.device
print(f"Device: {device}")


# ── Augmentation transforms ───────────────────────────────────────────────────
# All transforms end with the same ImageNet normalisation so VGG16 sees
# correctly pre-processed inputs.

_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

AUG_TRANSFORMS = [
    # Pass 0 — original (identical to DEFAULT_TRANSFORM)
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _NORM,
    ]),
    # Pass 1 — horizontal flip (class-preserving: mirroring never loses the subject)
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        _NORM,
    ]),
]

AUG_NAMES = ["original", "hflip"]


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_with_transform(extractor, transform, split, seed=0):
    """Extract features for one transform pass. Returns (features, soft_labels, labels)."""
    # Fix random seed so random transforms are deterministic and reproducible
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ImagenetteDataset(split=split, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    return extractor.extract_all(loader, desc=f"  {split}/{transform.transforms[0].__class__.__name__}")


def main():
    extractor = VGGExtractor(device=device)

    # ── Train: 3 augmented passes ─────────────────────────────────────────────
    all_features, all_soft, all_labels = [], [], []
    for i, (tf, name) in enumerate(zip(AUG_TRANSFORMS, AUG_NAMES)):
        print(f"\nPass {i} ({name}) — train")
        feat, soft, labs = extract_with_transform(extractor, tf, split="train", seed=i)
        all_features.append(feat)
        all_soft.append(soft)
        all_labels.append(labs)
        print(f"  → {feat.shape}")

        if device == "mps":
            torch.mps.empty_cache()

    train_features   = torch.cat(all_features, dim=0)
    train_soft_labels = torch.cat(all_soft,    dim=0)
    train_labels     = torch.cat(all_labels,   dim=0)
    print(f"\nTotal train: {train_features.shape}")

    # ── Val: standard transform only ──────────────────────────────────────────
    print("\nPass 0 (original) — val")
    val_features, val_soft_labels, val_labels = extract_with_transform(
        extractor, AUG_TRANSFORMS[0], split="val", seed=99,
    )
    print(f"  → {val_features.shape}")

    # ── Write ─────────────────────────────────────────────────────────────────
    out_path = args.out
    TensorStore.write(
        out_path,
        (train_features, train_soft_labels, train_labels),
        (val_features, val_soft_labels, val_labels),
    )
    print(f"\nstore_aug.h5 written → {out_path}")

    manifest_path = out_path.replace(".h5", ".csv").replace("store_aug", "manifest_aug")
    TensorStore.write_manifest(
        manifest_path,
        train_labels.numpy(),
        val_labels.numpy(),
        IMAGENETTE_CLASSES,
    )
    print(f"manifest_aug.csv written → {manifest_path}")

    print(f"\nSummary:")
    print(f"  train: {len(train_features):,} samples  (2× original — original + hflip)")
    print(f"  val:   {len(val_features):,} samples  (unchanged)")
    print(f"  augmentation passes: {AUG_NAMES}")


if __name__ == "__main__":
    main()
