"""Unit tests for the Flow Matching linear path and target vector field."""

import torch

from aidd_generative_docking.diffusion.flow_matching import (
    flow_matching_loss,
    linear_path,
    sample_prior,
    target_vector_field,
)


def test_linear_path_endpoints() -> None:
    x0 = torch.tensor([[0.0, 0.0, 0.0]])
    x1 = torch.tensor([[2.0, 4.0, 6.0]])
    torch.testing.assert_close(linear_path(x0, x1, torch.tensor(0.0)), x0)
    torch.testing.assert_close(linear_path(x0, x1, torch.tensor(1.0)), x1)


def test_linear_path_midpoint() -> None:
    x0 = torch.zeros(5, 3)
    x1 = torch.ones(5, 3)
    midpoint = linear_path(x0, x1, torch.tensor(0.5))
    torch.testing.assert_close(midpoint, torch.full((5, 3), 0.5))


def test_target_vector_field_is_displacement() -> None:
    x0 = torch.tensor([[1.0, 1.0, 1.0]])
    x1 = torch.tensor([[3.0, 0.0, -1.0]])
    torch.testing.assert_close(target_vector_field(x0, x1), x1 - x0)


def test_flow_matching_loss_zero_for_perfect_prediction() -> None:
    x0 = torch.randn(10, 3)
    x1 = torch.randn(10, 3)
    perfect_prediction = target_vector_field(x0, x1)
    loss = flow_matching_loss(perfect_prediction, x0, x1)
    assert loss.item() == 0.0


def test_sample_prior_shape() -> None:
    generator = torch.Generator().manual_seed(0)
    x0 = sample_prior((7, 3), generator=generator)
    assert x0.shape == (7, 3)
