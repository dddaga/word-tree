"""Evaluate frozen VGG16 on Imagenette val set.

Produces results/baseline_vgg16.json with top-1 accuracy, mAP,
per-class metrics, FC parameter count, and FC MACs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import json

from src.data.dataset import IMAGENETTE_CLASSES, get_dataloader
from src.data.extractor import VGGExtractor
from src.utils.metrics import compute_all_metrics, count_flops, count_params


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def main() -> None:
    """Run VGG16 baseline evaluation on Imagenette val set."""
    # Load extractor (frozen VGG16, eval mode, MPS)
    ext = VGGExtractor()

    # Val dataloader -- num_workers=0 required on macOS MPS
    loader = get_dataloader(split="val", batch_size=256, num_workers=0)

    # Run inference: discard features, keep soft_labels and labels
    _, soft_labels, labels = ext.extract_all(loader, desc="Evaluating VGG16 on val")

    # Convert to numpy for metrics
    scores = soft_labels.numpy()
    gt = labels.numpy()

    # Compute classification metrics
    metrics = compute_all_metrics(scores, gt, IMAGENETTE_CLASSES)

    # Count FC-only parameters and MACs
    fc_params = count_params(ext.model)
    fc_flops = count_flops(ext.model, (1, 25088))

    # Assemble result
    result = {
        "model": "VGG16 (frozen)",
        "fc_params": fc_params,
        **metrics,
        "flops_fc_per_inference": fc_flops,
        "flops_note": "MACs from thop.profile on model.classifier",
    }

    # Write JSON
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_vgg16.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"Model:         {result['model']}")
    print(f"FC params:     {fc_params:,}")
    print(f"Top-1 acc:     {metrics['top1_accuracy']:.4f}")
    print(f"mAP:           {metrics['mAP']:.4f}")
    print(f"FC MACs:       {fc_flops:,}")
    print(f"Saved to:      {out_path}")


if __name__ == "__main__":
    main()
