import os
from pathlib import Path

import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import VGG16_Weights
from tqdm import tqdm


IMAGENETTE_IMAGENET_INDICES = [
    0,
    217,
    482,
    491,
    497,
    566,
    569,
    571,
    574,
    701,
]

THIS_DIR = Path(__file__).resolve().parent


@torch.no_grad()
def prepare_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights = VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights).to(device)
    model.eval()

    feature_extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(),
    )
    classifier = model.classifier

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    data_dir = THIS_DIR / "data"
    os.makedirs(data_dir, exist_ok=True)

    for split in ("train", "val"):
        print(f"\nProcessing '{split}' split...")
        dataset = torchvision.datasets.Imagenette(
            root=str(data_dir),
            split=split,
            size="320px",
            download=True,
            transform=transform,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        all_features = []
        all_targets = []
        for images, _ in tqdm(dataloader, desc=f"Extracting {split}"):
            images = images.to(device)
            features = feature_extractor(images)
            logits_1000 = classifier(features)
            logits_10 = logits_1000[:, IMAGENETTE_IMAGENET_INDICES]
            soft_targets = torch.nn.functional.softmax(logits_10, dim=1)
            all_features.append(features.cpu().to(torch.float32))
            all_targets.append(soft_targets.cpu().to(torch.float32))

        features_tensor = torch.cat(all_features, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        save_path = data_dir / f"imagenette_{split}_data.pt"
        torch.save({"data": features_tensor, "label": targets_tensor}, str(save_path))
        print(f"Saved {split} tensors to {save_path}")
        print(f"Data tensor shape: {features_tensor.shape}")
        print(f"Label tensor shape: {targets_tensor.shape}")


if __name__ == "__main__":
    prepare_dataset()

