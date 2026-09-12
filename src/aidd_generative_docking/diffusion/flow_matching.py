"""Conditional Flow Matching: linear path and target vector field (stage 3-4).

Implements the deterministic linear-interpolation path between a prior
sample ``x0`` and a data sample ``x1``, and the corresponding constant
target vector field, following Lipman et al.'s Conditional Flow Matching
formulation (see ``ARCHITECTURE.md``):

    x_t = (1 - t) * x0 + t * x1
    u_t(x_t | x0, x1) = x1 - x0

These are plain closed-form tensor operations (no learned components), so
they are implemented directly here. The learned vector-field network that
is trained to regress ``u_t`` is a separate, not-yet-implemented component
(see ``models/equivariant.py``).
"""

from __future__ import annotations

import torch


def sample_prior(shape: tuple[int, ...], generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample ``x0`` from the standard Gaussian prior."""
    return torch.randn(shape, generator=generator)


def linear_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute ``x_t = (1 - t) * x0 + t * x1`` for the linear FM path.

    Args:
        x0: ``(..., 3)`` prior sample.
        x1: ``(..., 3)`` data sample, same shape as ``x0``.
        t: flow time in ``[0, 1]``, broadcastable against ``x0``/``x1``
            (typically shape ``(..., 1)`` or a scalar tensor).

    Returns:
        Interpolated coordinates, same shape as ``x0``.
    """
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {x0.shape} vs {x1.shape}")
    return (1.0 - t) * x0 + t * x1


def target_vector_field(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Target conditional vector field for the linear path: ``u_t = x1 - x0``.

    Constant in ``t`` for the linear path, so it does not depend on it.
    """
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {x0.shape} vs {x1.shape}")
    return x1 - x0


def flow_matching_loss(predicted_velocity: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Mean-squared error between a predicted and target velocity field."""
    target = target_vector_field(x0, x1)
    return torch.mean((predicted_velocity - target) ** 2)
