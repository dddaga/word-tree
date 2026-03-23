"""Verify stored soft labels match VGG16 direct evaluation.

Loads the val split from HDF5 tensor store, computes argmax accuracy from
stored soft labels, and cross-checks against the direct VGG16 eval result
in baseline_vgg16.json. Also checks entropy (collapsed distribution) and
class balance. Warn-only mode per D-03.

Output: updates results/baseline_vgg16.json with soft_label_accuracy_check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.dataset import IMAGENETTE_CLASSES
from src.data.store import TensorStore


# -------------------------------------------------------------------
# Thresholds (per D-03 in 02-CONTEXT.md)
# -------------------------------------------------------------------

ACCURACY_THRESHOLD = 0.001   # 0.1% accuracy match
ENTROPY_THRESHOLD = 0.01     # nats; warn if mean entropy below this
CLASS_BALANCE_THRESHOLD = 5.0  # percent; warn if any class below this


def verify_accuracy(
    store_acc: float, direct_acc: float
) -> tuple[bool, float]:
    """Check that stored soft-label argmax accuracy matches direct eval."""
    acc_diff = abs(store_acc - direct_acc)
    accuracy_match = acc_diff <= ACCURACY_THRESHOLD
    return accuracy_match, acc_diff


def check_entropy(soft_labels: np.ndarray) -> float:
    """Compute mean entropy of soft label distributions (nats)."""
    eps = 1e-10
    entropy = -np.sum(soft_labels * np.log(soft_labels + eps), axis=1)
    return float(np.mean(entropy))


def check_class_balance(
    labels: np.ndarray, class_names: list[str]
) -> list[str]:
    """Return warning strings for any under-represented class."""
    warnings: list[str] = []
    n_val = len(labels)
    for i, name in enumerate(class_names):
        count = int(np.sum(labels == i))
        pct = count / n_val * 100
        if pct < CLASS_BALANCE_THRESHOLD:
            msg = (
                f"WARNING: Class '{name}' has only {pct:.1f}% "
                f"of val set ({count}/{n_val})"
            )
            warnings.append(msg)
    return warnings


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load baseline results from Plan 01 direct eval
    baseline_path = Path("results/baseline_vgg16.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    direct_acc = baseline["top1_accuracy"]

    # 2. Load stored soft labels from HDF5
    store = TensorStore("data/store.h5")
    _, soft_labels, labels = store.get_split("val")
    print(f"Loaded val split: {soft_labels.shape[0]} samples, "
          f"{soft_labels.shape[1]} classes")

    # 3. Compute argmax accuracy from stored soft labels
    preds = np.argmax(soft_labels, axis=1)
    store_acc = float(np.mean(preds == labels))

    # 4. Check accuracy match (within 0.1% per D-03)
    accuracy_match, acc_diff = verify_accuracy(store_acc, direct_acc)
    print(f"Direct eval accuracy:              {direct_acc:.6f}")
    print(f"Stored soft-label argmax accuracy:  {store_acc:.6f}")
    status = "PASS" if accuracy_match else "FAIL"
    print(f"Difference: {acc_diff:.6f} ({status} <= {ACCURACY_THRESHOLD})")

    # 5. Check entropy (warn if mean < 0.01 nats)
    mean_entropy = check_entropy(soft_labels)
    print(f"Mean soft-label entropy: {mean_entropy:.4f} nats")
    if mean_entropy < ENTROPY_THRESHOLD:
        print("WARNING: Mean entropy < 0.01 nats "
              "-- distribution may be collapsed to near one-hot")

    # 6. Check class balance (warn if any class < 5%)
    balance_warnings = check_class_balance(labels, IMAGENETTE_CLASSES)
    for w in balance_warnings:
        print(w)
    if not balance_warnings:
        print("Class balance: all classes >= 5.0% of val set")

    # 7. Update baseline JSON with check result
    baseline["soft_label_accuracy_check"] = accuracy_match
    baseline["soft_label_accuracy_diff"] = acc_diff
    baseline["soft_label_mean_entropy"] = mean_entropy
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nUpdated {baseline_path} with "
          f"soft_label_accuracy_check={accuracy_match}")
