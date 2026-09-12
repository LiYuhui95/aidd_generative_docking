"""Minimal equivariant vector-field network (stage 3).

Defines the interface for the network `v_theta(x_t, t, pocket, ligand_graph)`
used as the Flow Matching vector field (see
``diffusion/flow_matching.py`` and ``ARCHITECTURE.md``). The network body is
intentionally not implemented yet: this stage is scoped to a small,
hand-written equivariant message-passing block (e.g. an EGNN-style update),
not a full architecture port.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EquivariantVectorField(nn.Module):
    """Predicts a per-atom velocity field for ligand coordinates.

    Args:
        hidden_dim: hidden feature dimension for node/edge embeddings.
        num_layers: number of equivariant message-passing layers.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ligand_graph: Any = None,
        pocket_graph: Any = None,
    ) -> torch.Tensor:
        """Predict velocity ``v_theta(x_t, t, ...)`` at coordinates ``x_t``.

        Args:
            x_t: ``(num_atoms, 3)`` ligand atom coordinates at flow time ``t``.
            t: scalar or ``(num_atoms,)`` flow-matching time in ``[0, 1]``.
            ligand_graph: ligand molecular graph (topology + atom features).
            pocket_graph: fixed protein pocket graph (conditioning context).

        Returns:
            ``(num_atoms, 3)`` predicted velocity, same shape as ``x_t``.
        """
        raise NotImplementedError(
            "Stage 3: equivariant vector-field network not yet implemented."
        )
