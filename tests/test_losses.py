"""Tests for SGNNET loss functions and initialization."""

import torch
import torch.nn.functional as F
import pytest

from src.sgnnet.losses import safety_valve_loss, load_balance_loss, total_loss
from src.sgnnet.init import initialize_sgnnet
from src.sgnnet.model import SGNNET


# -------------------------------------------------------------------
# Safety valve loss
# -------------------------------------------------------------------

class TestSafetyValveLoss:
    """Dead-zone Coulomb repulsion tests."""

    def test_well_separated_returns_zero(self):
        """Neurons spread far apart -> loss exactly 0.0."""
        # 4 neurons in D=2, each in a distinct quadrant of [0,1]^2
        W = torch.tensor([
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
        ])
        loss = safety_valve_loss(W, box_size=1.0, N=4, D=2)
        assert loss.item() == 0.0

    def test_colliding_neurons_positive(self):
        """Two neurons at same position -> loss > 0."""
        W = torch.tensor([
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ])
        loss = safety_valve_loss(W, box_size=1.0, N=2, D=4)
        assert loss.item() > 0.0

    def test_near_wall_positive(self):
        """Neuron at position 0.001 (near wall) -> loss > 0."""
        W = torch.tensor([[0.001, 0.5, 0.5, 0.5]])
        loss = safety_valve_loss(W, box_size=1.0, N=1, D=4)
        assert loss.item() > 0.0

    def test_uniform_spread_near_zero(self):
        """N=256, D=4 neurons spread uniformly -> near zero."""
        torch.manual_seed(42)
        N, D = 256, 4
        # Spread on a grid-like pattern in [0.1, 0.9]^D
        W = torch.rand(N, D) * 0.8 + 0.1
        loss = safety_valve_loss(W, box_size=1.0, N=N, D=D)
        # Should be very small (not necessarily exactly 0 for random)
        assert loss.item() < 1.0

    def test_infer_N_D_from_shape(self):
        """N and D inferred from W.shape when not provided."""
        W = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        loss = safety_valve_loss(W, box_size=1.0)
        assert loss.item() > 0.0


# -------------------------------------------------------------------
# Load balance loss
# -------------------------------------------------------------------

class TestLoadBalanceLoss:
    """Selection frequency variance tests."""

    def test_uniform_returns_zero(self):
        """Equal selection counts -> variance 0."""
        counts = torch.tensor([10.0, 10.0, 10.0, 10.0])
        loss = load_balance_loss(counts)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_skewed_positive(self):
        """Skewed counts -> positive variance."""
        counts = torch.tensor([40.0, 0.0, 0.0, 0.0])
        loss = load_balance_loss(counts)
        assert loss.item() > 0.0


# -------------------------------------------------------------------
# Total loss
# -------------------------------------------------------------------

class TestTotalLoss:
    """Combined loss function tests."""

    def test_returns_scalar_with_grad(self):
        """total_loss returns a scalar tensor with grad_fn."""
        scores = torch.randn(2, 10, requires_grad=True)
        targets = F.softmax(torch.randn(2, 10), dim=-1)
        W = torch.randn(20, 4, requires_grad=True)
        loss = total_loss(scores, targets, W, box_size=1.0, N=20, D=4)
        assert loss.dim() == 0
        assert loss.grad_fn is not None

    def test_lambda_safety_zero_ignores_safety(self):
        """With lambda_safety=0, safety valve loss is not added."""
        scores = torch.randn(2, 10, requires_grad=True)
        targets = F.softmax(torch.randn(2, 10), dim=-1)
        # Put neurons at same position (would cause safety > 0)
        W = torch.ones(20, 4, requires_grad=True) * 0.5
        loss_with = total_loss(
            scores, targets, W, lambda_safety=0.5, N=20, D=4
        )
        loss_without = total_loss(
            scores, targets, W, lambda_safety=0.0, N=20, D=4
        )
        # With safety=0, loss should be smaller (only task loss)
        assert loss_without.item() <= loss_with.item()

    def test_uses_kl_divergence(self):
        """Task loss uses KL divergence, not MSE."""
        # Identical distributions -> KL should be ~0
        probs = F.softmax(torch.randn(2, 10), dim=-1)
        W = torch.rand(20, 4) * 0.5 + 0.25
        loss = total_loss(
            probs, probs, W, lambda_safety=0.0, lambda_lb=0.0,
            N=20, D=4,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-4)


# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------

class TestInitializeSGNNET:
    """Initialization utility tests."""

    def test_w_in_box_range(self):
        """After init, W values are in [0, box_size]."""
        model = SGNNET(N_hidden=16, N_out=10, D=4, K=2, sparsity=0.5)
        initialize_sgnnet(model)
        assert model.W.data.min().item() >= 0.0
        assert model.W.data.max().item() <= model.box_size
