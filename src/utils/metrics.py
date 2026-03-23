"""Reusable classification metrics, parameter counting, and FLOPs profiling.

Provides three functions used across Phases 2, 4, and 6:
- compute_all_metrics: top-1 accuracy, mAP, per-class P/R/F1/AP
- count_params: FC-only parameter count
- count_flops: FC-only MACs via thop
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)


# -------------------------------------------------------------------
# Classification metrics
# -------------------------------------------------------------------

def compute_all_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute full classification metrics from softmax probabilities.

    Parameters
    ----------
    scores : np.ndarray, shape [N, C]
        Softmax probability outputs (rows sum to ~1). Caller applies
        softmax before passing (per D-04).
    labels : np.ndarray, shape [N]
        Ground-truth integer class labels in range [0, C).
    class_names : list[str]
        Human-readable class names, length C.

    Returns
    -------
    dict
        JSON-serializable metrics dict with keys: top1_accuracy, mAP,
        per_class (each class has accuracy, precision, recall, f1, AP).
    """
    num_classes = len(class_names)
    preds = np.argmax(scores, axis=1)
    top1_acc = float(np.mean(preds == labels))

    # Per-class precision, recall, F1 via sklearn
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    # Per-class accuracy
    per_class_acc = _per_class_accuracy(preds, labels, num_classes)

    # mAP: one-vs-rest binary AP per class (per D-06)
    per_class_ap = _per_class_ap(scores, labels, num_classes)
    mean_ap = float(np.mean(per_class_ap))

    # Assemble per-class dict
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "accuracy": per_class_acc[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "AP": per_class_ap[i],
        }

    return {
        "top1_accuracy": top1_acc,
        "mAP": mean_ap,
        "per_class": per_class,
    }


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> list[float]:
    """Compute per-class accuracy (correct rate within each class)."""
    acc = []
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 0:
            acc.append(float(np.mean(preds[mask] == i)))
        else:
            acc.append(0.0)
    return acc


def _per_class_ap(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> list[float]:
    """Compute per-class average precision using one-vs-rest approach."""
    labels_onehot = np.eye(num_classes, dtype=np.float32)[labels]
    ap_list = []
    for i in range(num_classes):
        ap = float(average_precision_score(labels_onehot[:, i], scores[:, i]))
        ap_list.append(ap)
    return ap_list


# -------------------------------------------------------------------
# Parameter and FLOPs counting
# -------------------------------------------------------------------

def count_params(model: torch.nn.Module) -> int:
    """Count total parameters in model.classifier (FC layers only).

    Counts all parameters in the classifier Sequential, which for VGG16
    includes Linear layers (Dropout/ReLU have zero params).
    """
    return sum(p.numel() for p in model.classifier.parameters())


def count_flops(model: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """Count MACs for model.classifier using thop.

    Profiles only the FC portion of the model, not the full CNN backbone.
    Returns MACs (multiply-accumulate operations), not 2x FLOPs.

    Parameters
    ----------
    model : torch.nn.Module
        Full model (only model.classifier is profiled).
    input_shape : tuple
        Shape of FC input, e.g. (1, 25088).

    Returns
    -------
    int
        MACs for the classifier portion.
    """
    from thop import profile

    # Detect device from model parameters
    device = next(model.classifier.parameters()).device
    dummy = torch.randn(*input_shape, device=device)
    macs, _ = profile(model.classifier, inputs=(dummy,), verbose=False)
    return int(macs)
