"""Gaussian radial basis function (RBF) distance encoding.

Mathematical definition
------------------------
Given ``num_centers`` K and ``cutoff`` c, K centers are placed evenly over
``[0, c]``::

    mu_k = k / (K - 1) * c,   k = 0, ..., K-1

with a shared width equal to the spacing between adjacent centers
(``sigma = c / (K - 1)``) -- a standard choice giving smooth, overlapping
coverage: each basis function's characteristic width matches the spacing to
its neighbors, so any distance in range always activates more than one
center, with no gaps. The k-th output for a distance ``d`` is::

    RBF_k(d) = exp( -(d - mu_k)^2 / (2 sigma^2) )

This is the same style of encoding used in e.g. SchNet (Schütt et al.,
2017) for continuous-filter message passing on interatomic distances: it
turns a single raw scalar distance into a smooth, differentiable,
higher-dimensional feature that is easier for a small MLP to combine
nonlinearly than the raw scalar would be.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GaussianRBF(nn.Module):
    """Encodes scalar distances as Gaussian radial basis function vectors.

    Args:
        num_centers: number of RBF centers K (>= 1).
        cutoff: distance at which the outermost center is placed; also the
            expected working range of the encoding (distances beyond it are
            still well-defined, just with small activations everywhere).
    """

    def __init__(self, num_centers: int, cutoff: float) -> None:
        super().__init__()
        if num_centers < 1:
            raise ValueError("num_centers must be >= 1")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.num_centers = num_centers
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, num_centers)
        self.register_buffer("centers", centers)
        self.width = cutoff / max(num_centers - 1, 1)

    def forward(self, distances: Tensor) -> Tensor:
        """Encode distances.

        Args:
            distances: ``(...,)`` tensor of raw scalar distances (no
                trailing singleton dimension).

        Returns:
            ``(..., num_centers)`` tensor of RBF activations.
        """
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-(diff**2) / (2 * self.width**2))
