"""Tests for SGNNET_Wave model (Plan 04-01).

Validates forward pass shapes, gradient flow, binary C masks,
phasor behavior, and masked normalization across all three stages.
"""

from __future__ import annotations

import torch
import pytest

from src.sgnnet.model_wave import SGNNET_Wave
from src.sgnnet.norm_masked import masked_normalize

# Shared small-model config for fast tests
CFG = dict(N_hidden=16, N_out=10, D=4, N_in=50, K=3, sparsity=0.90)


# -------------------------------------------------------------------
# Forward shape tests
# -------------------------------------------------------------------

class TestStageAForward:
    def test_stage_a_forward_shape(self):
        m = SGNNET_Wave(**CFG, use_proximity=False)
        out = m(torch.randn(2, CFG["N_in"]))
        assert out.shape == (2, CFG["N_out"])

    def test_stage_a_finite_output(self):
        m = SGNNET_Wave(**CFG, use_proximity=False)
        out = m(torch.randn(2, CFG["N_in"]))
        assert torch.isfinite(out).all()


class TestStageBForward:
    def test_stage_b_forward_shape(self):
        m = SGNNET_Wave(**CFG, use_proximity=True)
        out = m(torch.randn(2, CFG["N_in"]))
        assert out.shape == (2, CFG["N_out"])

    def test_stage_b_finite_output(self):
        m = SGNNET_Wave(**CFG, use_proximity=True)
        out = m(torch.randn(2, CFG["N_in"]))
        assert torch.isfinite(out).all()


class TestStageCForward:
    def test_stage_c_forward_shape(self):
        m = SGNNET_Wave(**CFG, use_proximity=True, use_wphase=True)
        out = m(torch.randn(2, CFG["N_in"]))
        assert out.shape == (2, CFG["N_out"])

    def test_stage_c_has_wphase(self):
        m = SGNNET_Wave(**CFG, use_proximity=True, use_wphase=True)
        assert m.W_phase is not None
        assert m.W_phase.shape == (CFG["N_hidden"] + CFG["N_out"], CFG["D"])

    def test_stage_a_no_wphase(self):
        m = SGNNET_Wave(**CFG, use_proximity=False)
        assert m.W_phase is None


# -------------------------------------------------------------------
# C matrix tests
# -------------------------------------------------------------------

class TestCMatrices:
    def test_c_masks_are_buffers(self):
        """No C mask appears in named_parameters."""
        m = SGNNET_Wave(**CFG, use_proximity=False)
        param_names = [name for name, _ in m.named_parameters()]
        for name in param_names:
            assert "mask" not in name, f"Mask is a parameter: {name}"

    def test_c_masks_binary(self):
        """All C mask values are 0.0 or 1.0."""
        m = SGNNET_Wave(**CFG, use_proximity=False)
        for name in ["C_input_mask", "C_hh_mask", "C_ho_mask"]:
            mask = getattr(m, name)
            unique = torch.unique(mask)
            assert all(v in (0.0, 1.0) for v in unique.tolist()), (
                f"{name} has non-binary values: {unique}"
            )

    def test_c_masks_sparsity(self):
        """C masks are approximately 90% sparse (>= 0.85 for small N)."""
        m = SGNNET_Wave(**CFG, use_proximity=False)
        for name in ["C_input_mask", "C_hh_mask", "C_ho_mask"]:
            mask = getattr(m, name)
            zero_frac = (mask == 0).float().mean().item()
            assert zero_frac >= 0.85, f"{name} not sparse enough: {zero_frac:.3f}"

    def test_no_self_connections_chh(self):
        """C_hh_mask diagonal is all zeros."""
        m = SGNNET_Wave(**CFG, use_proximity=False)
        diag = torch.diagonal(m.C_hh_mask)
        assert (diag == 0).all(), "C_hh_mask has self-connections on diagonal"

    def test_param_count_binary_c(self):
        """Only W_pos (and optionally W_phase) are learned parameters."""
        m = SGNNET_Wave(**CFG, use_proximity=False)
        param_names = [n for n, _ in m.named_parameters()]
        assert param_names == ["W_pos"], f"Unexpected params: {param_names}"

        m_c = SGNNET_Wave(**CFG, use_proximity=True, use_wphase=True)
        param_names_c = sorted([n for n, _ in m_c.named_parameters()])
        assert param_names_c == ["W_phase", "W_pos"], (
            f"Stage C unexpected params: {param_names_c}"
        )


# -------------------------------------------------------------------
# Gradient tests
# -------------------------------------------------------------------

class TestGradients:
    def test_w_pos_gradient(self):
        """W_pos.grad is not None after backward."""
        m = SGNNET_Wave(**CFG, use_proximity=False)
        out = m(torch.randn(2, CFG["N_in"]))
        out.sum().backward()
        assert m.W_pos.grad is not None, "W_pos.grad should exist"

    def test_w_phase_gradient(self):
        """In Stage C mode, W_phase.grad is not None after backward."""
        m = SGNNET_Wave(**CFG, use_proximity=True, use_wphase=True)
        out = m(torch.randn(2, CFG["N_in"]))
        out.sum().backward()
        assert m.W_phase.grad is not None, "W_phase.grad should exist"

    def test_w_pos_gradient_stage_b(self):
        """W_pos.grad flows in Stage B (proximity) mode."""
        m = SGNNET_Wave(**CFG, use_proximity=True)
        out = m(torch.randn(2, CFG["N_in"]))
        out.sum().backward()
        assert m.W_pos.grad is not None, "W_pos.grad should exist in Stage B"


# -------------------------------------------------------------------
# Phasor behavior tests
# -------------------------------------------------------------------

class TestPhasorBehavior:
    def test_z_im_zero_after_seed_stageA(self):
        """In Stage A, Z_im remains zero throughout."""
        m = SGNNET_Wave(**CFG, use_proximity=False)
        Z_re, Z_im = m._seed(torch.randn(2, CFG["N_in"]))
        assert (Z_im == 0).all(), "Z_im should be zero after seed"
        Z_re, Z_im = m._iterate_hidden(Z_re, Z_im)
        assert (Z_im == 0).all(), "Z_im should remain zero in Stage A"

    def test_z_im_nonzero_after_proximity(self):
        """In Stage B, Z_im becomes non-zero after proximity iteration."""
        torch.manual_seed(42)
        m = SGNNET_Wave(**CFG, use_proximity=True)
        Z_re, Z_im = m._seed(torch.randn(2, CFG["N_in"]))
        assert (Z_im == 0).all(), "Z_im should be zero after seed"
        Z_re, Z_im = m._iterate_hidden(Z_re, Z_im)
        # After proximity routing, Z_im should have non-zero values
        assert Z_im.abs().max() > 0, "Z_im should be non-zero after proximity"

    def test_masked_norm_excludes_inactive(self):
        """Zero-activation neurons have zero output after normalization."""
        x = torch.zeros(2, 8, 4)
        x[:, :4] = torch.randn(2, 4, 4)  # Only first 4 active
        out = masked_normalize(x)
        assert (out[:, 4:] == 0).all(), "Inactive neurons should be zeroed"
