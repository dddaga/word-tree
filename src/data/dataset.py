"""Imagenette dataset with VGG16-standard preprocessing.

Provides ImagenetteDataset wrapping torchvision.datasets.ImageFolder with
canonical class ordering (0-9) mapped from Imagenette folder names.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ---------------------------------------------------------------------------
# Constants: class names, ImageNet indices, folder-to-index mapping
# ---------------------------------------------------------------------------

IMAGENETTE_CLASSES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

IMAGENET_INDICES = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

IMAGENETTE_FOLDER_TO_IDX = {
    "n01440764": 0,  # tench
    "n02102040": 1,  # English springer
    "n02979186": 2,  # cassette player
    "n03000684": 3,  # chain saw
    "n03028079": 4,  # church
    "n03394916": 5,  # French horn
    "n03417042": 6,  # garbage truck
    "n03425413": 7,  # gas pump
    "n03445777": 8,  # golf ball
    "n03888257": 9,  # parachute
}

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ImagenetteDataset(Dataset):
    """Imagenette dataset with canonical 0-9 class ordering.

    Wraps ``torchvision.datasets.ImageFolder`` and remaps its alphabetical
    class indices to the canonical ordering defined by
    ``IMAGENETTE_FOLDER_TO_IDX``.

    Parameters
    ----------
    root : str
        Path to the imagenette2-320 directory (contains train/ and val/).
    split : str
        One of ``"train"`` or ``"val"``.
    transform : callable or None
        Image transform pipeline.  Uses ``DEFAULT_TRANSFORM`` (VGG16
        standard: resize 256, center-crop 224, ImageNet normalize) when
        *None*.
    """

    def __init__(
        self,
        root: str = "data/imagenette2-320",
        split: str = "train",
        transform=None,
    ) -> None:
        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.transform = transform if transform is not None else DEFAULT_TRANSFORM
        self._folder_ds = ImageFolder(str(split_dir), transform=self.transform)
        self._remap = self._build_remap(self._folder_ds.class_to_idx)

    def __len__(self) -> int:
        return len(self._folder_ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, folder_idx = self._folder_ds[idx]
        canonical_idx = self._remap[folder_idx]
        return image, canonical_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_remap(class_to_idx: dict[str, int]) -> dict[int, int]:
        """Map ImageFolder's alphabetical indices to canonical 0-9 ordering.

        ``class_to_idx`` maps folder name -> ImageFolder index.  We need
        ImageFolder index -> canonical index so ``__getitem__`` can remap
        on the fly.
        """
        remap: dict[int, int] = {}
        for folder_name, folder_idx in class_to_idx.items():
            canonical = IMAGENETTE_FOLDER_TO_IDX[folder_name]
            remap[folder_idx] = canonical
        return remap


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def get_dataloader(
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
    root: str = "data/imagenette2-320",
) -> DataLoader:
    """Create a DataLoader wrapping ImagenetteDataset for *split*.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"val"``.
    batch_size : int
        Batch size (default 256, suitable for MPS).
    num_workers : int
        Number of data-loading workers.
    root : str
        Path to imagenette2-320 directory.

    Returns
    -------
    DataLoader
    """
    dataset = ImagenetteDataset(root=root, split=split)
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
