"""Tests for geometry primitives: r*, dynamic connectivity hh/ho."""

import torch
import pytest

from src.sgnnet.geometry import (
    personal_volume_radius,
    dynamic_connectivity_hh,
    dynamic_connectivity_ho,
)


# ---------------------------------------------------------------------------
# personal_volume_radius
# ---------------------------------------------------------------------------

class TestPersonalVolumeRadius:

    def test_known_value(self):
        """r* = (1.0/2) / 256^(1/4) = 0.5 / 4.0 = 0.125."""
        r = personal_volume_radius(256, 4, 1.0)
        assert abs(r - 0.125) < 1e-7

    def test_always_within_half_box(self):
        for n in [10, 100, 1000]:
            for d in [2, 4, 8]:
                r = personal_volume_radius(n, d, 1.0)
                assert r <= 0.5 + 1e-9

    def test_scales_with_box_size(self):
        r1 = personal_volume_radius(100, 4, 1.0)
        r2 = personal_volume_radius(100, 4, 2.0)
        assert abs(r2 - 2 * r1) < 1e-7


# ---------------------------------------------------------------------------
# dynamic_connectivity_hh
# ---------------------------------------------------------------------------

class TestDynamicConnectivityHH:

    def test_output_shapes(self):
        batch, n_h, d = 2, 8, 4
        A = torch.randn(batch, n_h, d)
        W = torch.randn(n_h, d)
        contrib, gate = dynamic_connectivity_hh(A, W, n_h, d)
        assert contrib.shape == (batch, n_h, d)
        assert gate.shape == (batch, n_h, n_h)

    def test_diagonal_is_zero(self):
        """No self-routing: gate diagonal must be zero."""
        batch, n_h, d = 2, 8, 4
        A = torch.randn(batch, n_h, d)
        W = torch.randn(n_h, d)
        _, gate = dynamic_connectivity_hh(A, W, n_h, d)
        for b in range(batch):
            diag = torch.diagonal(gate[b])
            assert torch.allclose(diag, torch.zeros(n_h))

    def test_far_apart_zero_contribution(self):
        """Neurons far apart (dist >> r*) produce zero contribution."""
        batch, n_h, d = 1, 4, 4
        # Place activations and weights very far apart
        A = torch.zeros(batch, n_h, d)
        W = torch.ones(n_h, d) * 100.0  # far away
        contrib, gate = dynamic_connectivity_hh(A, W, n_h, d)
        assert torch.allclose(contrib, torch.zeros_like(contrib), atol=1e-6)
        assert gate.sum() == 0.0

    def test_gate_reflects_proximity(self):
        """Neurons within r* have non-zero gate entries."""
        batch, n_h, d = 1, 4, 4
        r_star = personal_volume_radius(n_h, d, 1.0)
        # Place two neurons very close, others far
        W = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [r_star * 0.5, 0.0, 0.0, 0.0],  # within r*
            [10.0, 10.0, 10.0, 10.0],        # far
            [10.0, 10.0, 10.0, 10.1],        # far
        ])
        A = W.unsqueeze(0)  # activations at same positions as weights
        _, gate = dynamic_connectivity_hh(A, W, n_h, d)
        # Neurons 0 and 1 should gate each other
        assert gate[0, 0, 1] > 0
        assert gate[0, 1, 0] > 0
        # Neurons 0 and 2 should not gate
        assert gate[0, 0, 2] == 0.0


# ---------------------------------------------------------------------------
# dynamic_connectivity_ho
# ---------------------------------------------------------------------------

class TestDynamicConnectivityHO:

    def test_output_shape(self):
        batch, n_h, n_out, d = 2, 8, 3, 4
        A = torch.randn(batch, n_h, d)
        W_out = torch.randn(n_out, d)
        contrib = dynamic_connectivity_ho(A, W_out, n_h, d)
        assert contrib.shape == (batch, n_out, d)

    def test_far_apart_zero(self):
        batch, n_h, n_out, d = 1, 4, 3, 4
        A = torch.zeros(batch, n_h, d)
        W_out = torch.ones(n_out, d) * 100.0
        contrib = dynamic_connectivity_ho(A, W_out, n_h, d)
        assert torch.allclose(contrib, torch.zeros_like(contrib), atol=1e-6)
