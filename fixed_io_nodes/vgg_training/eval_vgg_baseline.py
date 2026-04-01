"""
Evaluate VGG16 (IMAGENET1K_V1) accuracy on the Imagenette validation set.

Uses VGG16's predictions already stored in imagenette_val_data.pt (the 'label' field
is softmax of VGG16's classifier output over the 10 Imagenette classes).
True labels are loaded from the Imagenette dataset (no inference run).
"""

from pathlib import Path

import torch
import torchvision
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
VAL_DATA_PATH = DATA_DIR / "imagenette_val_data.pt"


def main():
    # VGG16's stored predictions: soft targets [N, 10]
    val = torch.load(str(VAL_DATA_PATH), map_location="cpu")
    vgg_preds = val["label"].argmax(dim=1)  # [N]
    print(f"Val samples: {len(vgg_preds)}")

    # True Imagenette labels — load dataset with same order (shuffle=False)
    dataset = torchvision.datasets.Imagenette(
        root=str(DATA_DIR),
        split="val",
        size="320px",
        download=False,
        transform=transforms.ToTensor(),  # minimal transform just to iterate
    )
    true_labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])

    correct = (vgg_preds == true_labels).sum().item()
    total = len(true_labels)
    accuracy = 100.0 * correct / total
    print(f"VGG16 accuracy on Imagenette val set: {correct}/{total} = {accuracy:.2f}%")


if __name__ == "__main__":
    main()
