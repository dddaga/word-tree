"""Tests for SGNNET loss functions, initialization, and integration."""

import json
import os

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
        # Same logits -> log_softmax identical -> KL should be ~0
        logits = torch.randn(2, 10)
        targets = F.softmax(logits, dim=-1)
        W = torch.rand(20, 4) * 0.5 + 0.25
        loss = total_loss(
            logits, targets, W, lambda_safety=0.0, lambda_lb=0.0,
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


# -------------------------------------------------------------------
# Integration: parameter budget and sparsity
# -------------------------------------------------------------------

VGG16_FC_PARAMS = 123_642_856
BUDGET_1_PERCENT = VGG16_FC_PARAMS * 0.01  # 1,236,428.56


def _count_active_params(model: SGNNET) -> tuple[int, dict]:
    """Count active (non-masked) parameters in SGNNET.

    C matrices use mask buffers: only entries where mask==1 are active.
    W and norm parameters are fully active.
    """
    breakdown = {}
    total = 0

    # C matrices: count only masked-in entries
    for c_name, mask_name in [
        ("C_input_values", "C_input_mask"),
        ("C_hh_values", "C_hh_mask"),
        ("C_ho_values", "C_ho_mask"),
    ]:
        mask = getattr(model, mask_name)
        active = int(mask.sum().item())
        breakdown[c_name] = active
        total += active

    # W: fully active
    breakdown["W"] = model.W.numel()
    total += model.W.numel()

    # Norm: fully active
    norm_count = sum(
        p.numel() for n, p in model.named_parameters() if "norm" in n
    )
    breakdown["norm"] = norm_count
    total += norm_count

    return total, breakdown


class TestParameterBudget:
    """Verify SGNNET stays within 1% of VGG16 FC params."""

    def test_total_params_within_budget(self):
        """Default config (N_hidden=256) active params <= 1% of VGG16 FC."""
        model = SGNNET(N_hidden=256, N_out=10, D=4, sparsity=0.90)
        total, _ = _count_active_params(model)
        assert total <= int(BUDGET_1_PERCENT), (
            f"Active params {total} exceed 1% budget {int(BUDGET_1_PERCENT)}"
        )

    def test_sparsity_all_c_matrices(self):
        """All three C matrices have high sparsity.

        C_ho (256x10) has lower sparsity due to guaranteed-connectivity
        row fix on a small matrix. Threshold 0.85 for C_ho, 0.89 others.
        """
        model = SGNNET(N_hidden=256, N_out=10, D=4, sparsity=0.90)
        thresholds = {"C_input": 0.89, "C_hh": 0.89, "C_ho": 0.85}
        for name, mask in [
            ("C_input", model.C_input_mask),
            ("C_hh", model.C_hh_mask),
            ("C_ho", model.C_ho_mask),
        ]:
            sparsity = (mask == 0).float().mean().item()
            threshold = thresholds[name]
            assert sparsity >= threshold, (
                f"{name} sparsity {sparsity:.4f} < {threshold}"
            )


# -------------------------------------------------------------------
# Integration: toy training loop
# -------------------------------------------------------------------

class TestToyTrainingLoop:
    """End-to-end forward + backward + optimizer step."""

    def test_w_positions_move(self):
        """3 optimizer steps move W positions."""
        torch.manual_seed(7)
        model = SGNNET(N_hidden=16, N_out=10, D=4, K=2, sparsity=0.5)
        initialize_sgnnet(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(4, 25088)
        target = F.softmax(torch.randn(4, 10), dim=-1)

        W_before = model.W.data.clone()
        for _ in range(3):
            scores = model(x)
            gate = model._last_gate
            loss = total_loss(
                scores, target, model.W, gate=gate,
                box_size=model.box_size,
                N=model.N_hidden + model.N_out, D=model.D,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        W_after = model.W.data.clone()
        assert not torch.allclose(W_before, W_after), (
            "W positions did not move during training"
        )


# -------------------------------------------------------------------
# Write sgnnet_config.json (runs as test side effect)
# -------------------------------------------------------------------

class TestWriteConfig:
    """Write results/sgnnet_config.json with parameter accounting."""

    def test_write_sgnnet_config(self):
        """Generate config JSON with parameter breakdown."""
        model = SGNNET(N_hidden=256, N_out=10, D=4, sparsity=0.90)
        total, breakdown = _count_active_params(model)

        # Sparsity
        sparsities = {}
        for name, mask in [
            ("C_input", model.C_input_mask),
            ("C_hh", model.C_hh_mask),
            ("C_ho", model.C_ho_mask),
        ]:
            sparsities[name] = round(
                (mask == 0).float().mean().item(), 4
            )

        # Load VGG16 baseline
        baseline_path = os.path.join("results", "baseline_vgg16.json")
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                vgg = json.load(f)
            vgg_fc = vgg["fc_params"]
        else:
            vgg_fc = VGG16_FC_PARAMS

        config = {
            "model": "SGNNET",
            "N_in": 25088,
            "N_hidden": 256,
            "N_out": 10,
            "D": 4,
            "K": 3,
            "sparsity": 0.90,
            "box_size": 1.0,
            "total_params": total,
            "param_breakdown": breakdown,
            "percent_of_vgg16_fc": round(total / vgg_fc * 100, 4),
            "C_input_sparsity": sparsities["C_input"],
            "C_hh_sparsity": sparsities["C_hh"],
            "C_ho_sparsity": sparsities["C_ho"],
            "vgg16_fc_params": vgg_fc,
        }

        os.makedirs("results", exist_ok=True)
        out_path = os.path.join("results", "sgnnet_config.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

        # Verify written correctly
        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["percent_of_vgg16_fc"] <= 1.0
        assert loaded["total_params"] == total
