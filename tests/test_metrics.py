"""Unit tests for src.utils.metrics module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision.models import VGG16_Weights, vgg16

from src.utils.metrics import compute_all_metrics, count_flops, count_params


# -------------------------------------------------------------------
# compute_all_metrics tests
# -------------------------------------------------------------------

CLASS_NAMES_3 = ["cat", "dog", "bird"]


def test_perfect_predictions() -> None:
    """Perfect scores produce top1_accuracy=1.0 and mAP>0.99."""
    n = 100
    labels = np.array([i % 3 for i in range(n)])
    # Build scores: 0.9 on correct class, 0.05 on others
    scores = np.full((n, 3), 0.05, dtype=np.float32)
    for i in range(n):
        scores[i, labels[i]] = 0.9

    result = compute_all_metrics(scores, labels, CLASS_NAMES_3)

    assert result["top1_accuracy"] == 1.0
    assert result["mAP"] > 0.99
    # Check per_class structure
    for name in CLASS_NAMES_3:
        cls = result["per_class"][name]
        assert "accuracy" in cls
        assert "precision" in cls
        assert "recall" in cls
        assert "f1" in cls
        assert "AP" in cls


def test_mixed_predictions() -> None:
    """Mixed predictions produce 0 < accuracy < 1 and correct dict keys."""
    rng = np.random.RandomState(42)
    n = 100
    labels = np.array([i % 3 for i in range(n)])
    # Random scores (softmax-like via normalization)
    raw = rng.rand(n, 3).astype(np.float32)
    scores = raw / raw.sum(axis=1, keepdims=True)

    result = compute_all_metrics(scores, labels, CLASS_NAMES_3)

    assert 0.0 < result["top1_accuracy"] < 1.0
    assert "mAP" in result
    assert "per_class" in result
    assert len(result["per_class"]) == 3


# -------------------------------------------------------------------
# count_params test
# -------------------------------------------------------------------

def test_count_params_vgg16() -> None:
    """VGG16 classifier has exactly 123,642,856 parameters."""
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    assert count_params(model) == 123642856


# -------------------------------------------------------------------
# count_flops test
# -------------------------------------------------------------------

def test_count_flops_vgg16() -> None:
    """FC FLOPs for VGG16 is a positive integer."""
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    result = count_flops(model, (1, 25088))
    assert isinstance(result, int)
    assert result > 0
