"""Tests for spatial encoding of VGG16 pool5 features."""

import torch
import pytest

from src.sgnnet.encoding import compute_spatial_encoding


class TestComputeSpatialEncoding:
    """Verify flat-index-to-normalized-coord mapping."""

    def test_output_shape(self):
        coords = compute_spatial_encoding(25088)
        assert coords.shape == (25088, 3)

    def test_first_element(self):
        """Index 0 -> channel 0, row 0, col 0 -> [0, 0, 0]."""
        coords = compute_spatial_encoding(25088)
        assert torch.allclose(coords[0], torch.tensor([0.0, 0.0, 0.0]))

    def test_last_position_first_channel(self):
        """Index 48 -> channel 0, row 6, col 6 -> [0/6, 6/6, 6/6]."""
        coords = compute_spatial_encoding(25088)
        expected = torch.tensor([0.0, 6.0 / 6.0, 6.0 / 6.0])
        assert torch.allclose(coords[48], expected)

    def test_first_position_second_channel(self):
        """Index 49 -> channel 1, row 0, col 0 -> [1/511, 0, 0]."""
        coords = compute_spatial_encoding(25088)
        expected = torch.tensor([1.0 / 511.0, 0.0, 0.0])
        assert torch.allclose(coords[49], expected)

    def test_values_in_unit_range(self):
        coords = compute_spatial_encoding(25088)
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0

    def test_dtype_is_float32(self):
        coords = compute_spatial_encoding(25088)
        assert coords.dtype == torch.float32

    def test_custom_grid_size(self):
        """Small grid: 2 channels, 2x2 -> n_in=8."""
        coords = compute_spatial_encoding(n_in=8, h=2, w=2, c=2)
        assert coords.shape == (8, 3)
        # Index 0: ch=0, row=0, col=0 -> [0, 0, 0]
        assert torch.allclose(coords[0], torch.tensor([0.0, 0.0, 0.0]))
        # Index 4: ch=1, row=0, col=0 -> [1/1, 0, 0] = [1, 0, 0]
        assert torch.allclose(coords[4], torch.tensor([1.0, 0.0, 0.0]))
