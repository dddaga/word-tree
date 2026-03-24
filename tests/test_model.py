"""Tests for SGNNET nn.Module (Plan 03-02).

Validates three-phase forward pass, C matrix shapes/sparsity,
backward pass gradients, and _last_gate behavior.
"""

from __future__ import annotations

import torch
import pytest

from src.sgnnet.model import SGNNET


# -------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------

def test_construction():
    """SGNNET constructs without error."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    assert model is not None


# -------------------------------------------------------------------
# Forward pass shape
# -------------------------------------------------------------------

def test_forward_shape():
    """Forward returns [batch, N_out] scores."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    x = torch.randn(2, 25088)
    scores = model(x)
    assert scores.shape == (2, 10)


def test_forward_small():
    """Small N_hidden=4, K=1 produces valid scores."""
    model = SGNNET(N_hidden=4, N_out=10, D=4, K=1)
    x = torch.randn(1, 25088)
    scores = model(x)
    assert scores.shape == (1, 10)
    assert torch.isfinite(scores).all()


# -------------------------------------------------------------------
# Backward pass & gradients
# -------------------------------------------------------------------

def test_backward_no_nan():
    """Backward produces no NaN in any parameter gradient."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    x = torch.randn(2, 25088)
    scores = model(x)
    loss = scores.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN in {name}.grad"


def test_w_grad_exists():
    """model.W.grad is not None after backward."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    x = torch.randn(2, 25088)
    scores = model(x)
    loss = scores.sum()
    loss.backward()
    assert model.W.grad is not None, "W.grad should exist after backward"


# -------------------------------------------------------------------
# C matrix shapes
# -------------------------------------------------------------------

def test_c_input_shape():
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    assert model.C_input_values.shape == (25088, 16)


def test_c_hh_shape():
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    assert model.C_hh_values.shape == (16, 16)


def test_c_ho_shape():
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    assert model.C_ho_values.shape == (16, 10)


# -------------------------------------------------------------------
# Sparsity patterns
# -------------------------------------------------------------------

def test_c_input_connectivity():
    """Every input neuron has at least one connection."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    per_row = model.C_input_mask.sum(dim=1)
    assert (per_row >= 1).all(), "Some input neurons have zero connections"


def test_c_input_sparsity():
    """C_input_mask is approximately 90% sparse.

    Threshold lowered to 0.88 because with small N_hidden=16 the
    guaranteed-one-connection-per-row fix adds proportionally more entries.
    """
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    zero_frac = (model.C_input_mask == 0).float().mean().item()
    assert zero_frac >= 0.88, f"C_input not sparse enough: {zero_frac:.3f}"


def test_c_hh_sparsity():
    """C_hh_mask is approximately 90% sparse (0.85 for small N + zero_diag)."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    zero_frac = (model.C_hh_mask == 0).float().mean().item()
    assert zero_frac >= 0.85, f"C_hh not sparse enough: {zero_frac:.3f}"


# -------------------------------------------------------------------
# Spatial coords buffer
# -------------------------------------------------------------------

def test_spatial_coords_buffer():
    """Model has spatial_coords buffer of shape [25088, 3]."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    assert hasattr(model, "spatial_coords")
    assert model.spatial_coords.shape == (25088, 3)


# -------------------------------------------------------------------
# Epsilon safety net
# -------------------------------------------------------------------

def test_scores_epsilon():
    """Scores have epsilon added (all >= epsilon threshold)."""
    torch.manual_seed(42)
    model = SGNNET(N_hidden=4, N_out=10, D=4, K=1)
    # Zero input should produce near-zero activations + epsilon
    x = torch.zeros(1, 25088)
    scores = model(x)
    # epsilon = 1e-4, scores should be >= 1e-4
    assert (scores >= 1e-4 - 1e-6).all(), "Scores missing epsilon safety"


# -------------------------------------------------------------------
# _last_gate behavior
# -------------------------------------------------------------------

def test_last_gate_init_none():
    """_last_gate is None after construction."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    assert model._last_gate is None


def test_last_gate_none_k1():
    """_last_gate is None after forward with K=1 (no hidden iters)."""
    model = SGNNET(N_hidden=4, N_out=10, D=4, K=1)
    model(torch.randn(1, 25088))
    assert model._last_gate is None


def test_last_gate_set_k3():
    """_last_gate is not None after forward with K=3."""
    model = SGNNET(N_hidden=16, N_out=10, D=4, K=3)
    model(torch.randn(1, 25088))
    assert model._last_gate is not None
