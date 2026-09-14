"""Tests for GaussianRBF distance encoding: shape, peak location, smoothness."""

from __future__ import annotations

import torch

from aidd_generative_docking.models.rbf import GaussianRBF


def test_rbf_output_shape() -> None:
    rbf = GaussianRBF(num_centers=16, cutoff=8.0)
    distances = torch.linspace(0.0, 10.0, 25)
    out = rbf(distances)
    assert out.shape == (25, 16)


def test_rbf_output_shape_multi_dim_input() -> None:
    rbf = GaussianRBF(num_centers=8, cutoff=5.0)
    distances = torch.rand(4, 7) * 5.0
    out = rbf(distances)
    assert out.shape == (4, 7, 8)


def test_rbf_peaks_exactly_at_its_own_center() -> None:
    num_centers = 10
    cutoff = 8.0
    rbf = GaussianRBF(num_centers=num_centers, cutoff=cutoff)
    centers = rbf.centers

    # For distance exactly at center k, RBF_k should be 1.0 (the maximum
    # possible value of exp(-x^2/2sigma^2)) and >= every other center's
    # activation at that same distance.
    out = rbf(centers)  # (num_centers, num_centers)
    diag = out.diagonal()
    torch.testing.assert_close(diag, torch.ones(num_centers), atol=1e-6, rtol=1e-6)
    assert (out.amax(dim=-1) == diag).all()


def test_rbf_is_smooth_no_discontinuities() -> None:
    rbf = GaussianRBF(num_centers=16, cutoff=8.0)
    distances = torch.linspace(0.0, 12.0, 2001)  # includes beyond cutoff
    out = rbf(distances)
    # a smooth function sampled this finely should have small step-to-step jumps
    max_jump = (out[1:] - out[:-1]).abs().max().item()
    assert max_jump < 0.01


def test_rbf_values_bounded_in_unit_interval() -> None:
    rbf = GaussianRBF(num_centers=12, cutoff=6.0)
    distances = torch.linspace(-5.0, 20.0, 500)  # exercise well outside nominal range too
    out = rbf(distances)
    assert torch.isfinite(out).all()
    assert (out >= 0.0).all()
    assert (out <= 1.0 + 1e-6).all()


def test_rbf_single_center_does_not_crash() -> None:
    rbf = GaussianRBF(num_centers=1, cutoff=5.0)
    out = rbf(torch.tensor([0.0, 2.5, 5.0]))
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()


def test_rbf_rejects_invalid_config() -> None:
    import pytest

    with pytest.raises(ValueError):
        GaussianRBF(num_centers=0, cutoff=5.0)
    with pytest.raises(ValueError):
        GaussianRBF(num_centers=8, cutoff=0.0)
