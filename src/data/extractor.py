"""VGG16 feature extractor for Imagenette.

Hooks VGG16's adaptive average pooling layer to capture 25088-dim pre-FC
activations, then selects 10 Imagenette logits from the 1000-class output
and applies T=1 softmax to produce soft labels.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

from src.data.dataset import IMAGENET_INDICES


# ---------------------------------------------------------------------------
# VGGExtractor
# ---------------------------------------------------------------------------

class VGGExtractor:
    """Extract 25088-dim features and 10-class soft labels from VGG16.

    Parameters
    ----------
    device : str or None
        Compute device.  Auto-detects MPS when *None*.
    """

    def __init__(self, device: str | None = None) -> None:
        # Device detection: MPS > CPU (no CUDA path needed)
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        # Load pretrained VGG16, eval mode, frozen
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        # Forward hook: capture pooled activations before classifier
        self._features: Tensor | None = None

        def _hook(module: torch.nn.Module, inp: tuple, output: Tensor) -> None:
            self._features = output.flatten(start_dim=1)

        self.model.avgpool.register_forward_hook(_hook)

        # Indices for selecting 10 Imagenette logits from 1000-class output
        self._imagenet_indices = torch.tensor(
            IMAGENET_INDICES, dtype=torch.long, device=self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_batch(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Extract features and soft labels for a single batch.

        Parameters
        ----------
        images : Tensor
            Shape ``[B, 3, 224, 224]``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(features, soft_labels)`` both on CPU.
            Features: ``[B, 25088]`` float32.
            Soft labels: ``[B, 10]`` float32, rows sum to 1.
        """
        images = images.to(self.device)

        with torch.no_grad():
            logits_1000 = self.model(images)

        features = self._features
        logits_10 = logits_1000[:, self._imagenet_indices]
        soft_labels = torch.softmax(logits_10, dim=1)

        return features.cpu(), soft_labels.cpu()

    def extract_all(
        self,
        dataloader: DataLoader,
        desc: str = "Extracting",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract features, soft labels, and ground-truth labels for an
        entire dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(images, labels)`` batches.
        desc : str
            Progress bar description.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(all_features, all_soft_labels, all_labels)`` on CPU.
        """
        feature_list: list[Tensor] = []
        soft_label_list: list[Tensor] = []
        label_list: list[Tensor] = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=desc):
                features, soft_labels = self.extract_batch(images)
                feature_list.append(features)
                soft_label_list.append(soft_labels)
                label_list.append(labels)

        all_features = torch.cat(feature_list, dim=0)
        all_soft_labels = torch.cat(soft_label_list, dim=0)
        all_labels = torch.cat(label_list, dim=0)

        return all_features, all_soft_labels, all_labels
