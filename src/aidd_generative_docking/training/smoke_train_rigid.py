"""Milestone 2 smoke experiment: overfit the rigid-body Flow Matching model
on a handful of real complexes and check whether sampled poses get closer
to the true bound pose than a random rigid initialization.

This is a pipeline sanity check, not a scientific benchmark (see
``README``/``ARCHITECTURE.md``): a handful of complexes trained for many
epochs is expected to memorize, not generalize. It exists to prove the
data -> model -> loss -> sampling loop is wired correctly end to end.

Kept separate from ``training/train.py`` (the config-driven, not-yet-
implemented general entrypoint for a future, larger-scale run) so that
plugging in the real Hydra/YAML config system later doesn't require
touching this direct-argument smoke-test path.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn

from aidd_generative_docking.data.sample import ComplexSample
from aidd_generative_docking.diffusion.rigid_flow_matching import (
    build_training_pair,
    integrate_rigid_pose,
    rigid_flow_matching_loss,
    sample_random_rigid_pose,
)
from aidd_generative_docking.evaluation.metrics import rmsd
from aidd_generative_docking.models.rigid_equivariant import RigidBodyVectorField


def run_smoke_experiment(
    samples: list[ComplexSample],
    num_epochs: int = 500,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    translation_noise_scale: float = 5.0,
    num_integration_steps: int = 50,
    seed: int = 0,
    device: str = "cpu",
    model_factory: Callable[[], nn.Module] | None = None,
) -> dict:
    """Train a rigid-body Flow Matching model on ``samples`` and evaluate pose RMSD.

    Uses one fixed random starting pose per complex (generated once, before
    training) for both the "initial" RMSD baseline and the post-training
    sampling run, so the reported before/after RMSDs are a fair comparison.
    Training itself resamples a fresh random ``(R0, t0, t)`` every epoch per
    complex, as standard for Flow Matching.

    Args:
        samples: real ``ComplexSample`` objects (ligand + pocket graphs).
        num_epochs: full-batch gradient steps over all of ``samples``.
        learning_rate: Adam learning rate.
        hidden_dim, num_layers: only used to construct the default
            ``RigidBodyVectorField`` when ``model_factory`` is ``None``;
            ignored otherwise (the factory owns its own hyperparameters).
        translation_noise_scale: see ``diffusion.rigid_flow_matching``.
        num_integration_steps: Euler steps for sampling.
        seed: seed for model init and the training noise stream.
        device: ``"cuda"`` or ``"cpu"``.
        model_factory: optional zero-argument callable returning an
            uninitialized ``nn.Module`` with the same call signature as
            ``RigidBodyVectorField.forward`` (e.g.
            ``CrossInteractionRigidBodyVectorField``). Defaults to ``None``,
            which reproduces the original Milestone 2 behavior exactly
            (constructs a ``RigidBodyVectorField`` from ``hidden_dim``/
            ``num_layers``) -- existing callers are unaffected.

    Returns:
        Dict with ``loss_history``, ``initial_rmsd`` and ``final_rmsd``
        (both ``{complex_id: float}``), and the trained ``model``.
    """
    torch.manual_seed(seed)
    train_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1000)  # separate stream for the fixed eval starting poses

    example = samples[0]
    if model_factory is None:
        model = RigidBodyVectorField(
            ligand_feature_dim=example.ligand.x.shape[1],
            pocket_feature_dim=example.pocket.x.shape[1],
            bond_feature_dim=example.ligand.edge_attr.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    else:
        model = model_factory()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    eval_starts = {}
    canonical_shapes = {}
    x1_by_id = {}
    for sample in samples:
        x1 = sample.ligand.pos.numpy().astype(np.float64)
        canonical_shape = x1 - x1.mean(axis=0)
        pocket_center = sample.pocket.pos.numpy().astype(np.float64).mean(axis=0)
        r0, t0 = sample_random_rigid_pose(pocket_center, translation_noise_scale, eval_rng)
        eval_starts[sample.complex_id] = (r0, t0)
        canonical_shapes[sample.complex_id] = canonical_shape
        x1_by_id[sample.complex_id] = x1

    initial_rmsd = {}
    for complex_id, (r0, t0) in eval_starts.items():
        x0 = (r0 @ canonical_shapes[complex_id].T).T + t0
        initial_rmsd[complex_id] = rmsd(x0, x1_by_id[complex_id])

    loss_history = []
    for _epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = torch.zeros((), device=device)
        for sample in samples:
            pair = build_training_pair(sample.ligand.pos, sample.pocket.pos, translation_noise_scale, train_rng)
            v_trans_pred, v_rot_pred = model(
                sample.ligand.x.to(device),
                sample.ligand.edge_index.to(device),
                sample.ligand.edge_attr.to(device),
                pair.xt.to(device),
                sample.pocket.x.to(device),
                sample.pocket.pos.to(device),
                torch.tensor([pair.t], dtype=torch.float32, device=device),
            )
            total_loss = total_loss + rigid_flow_matching_loss(
                v_trans_pred, v_rot_pred, pair.v_trans_target.to(device), pair.v_rot_target.to(device)
            )
        total_loss = total_loss / len(samples)
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

    model.eval()
    final_rmsd = {}
    for sample in samples:
        r0, t0 = eval_starts[sample.complex_id]
        canonical_shape = canonical_shapes[sample.complex_id]

        def velocity_fn(xt: np.ndarray, t: float, sample=sample) -> tuple[np.ndarray, np.ndarray]:
            xt_tensor = torch.as_tensor(xt, dtype=torch.float32, device=device)
            with torch.no_grad():
                v_trans, v_rot = model(
                    sample.ligand.x.to(device),
                    sample.ligand.edge_index.to(device),
                    sample.ligand.edge_attr.to(device),
                    xt_tensor,
                    sample.pocket.x.to(device),
                    sample.pocket.pos.to(device),
                    torch.tensor([t], dtype=torch.float32, device=device),
                )
            return v_trans.cpu().numpy().astype(np.float64), v_rot.cpu().numpy().astype(np.float64)

        x_final = integrate_rigid_pose(velocity_fn, canonical_shape, r0, t0, num_integration_steps)
        final_rmsd[sample.complex_id] = rmsd(x_final, x1_by_id[sample.complex_id])

    return {
        "loss_history": loss_history,
        "initial_rmsd": initial_rmsd,
        "final_rmsd": final_rmsd,
        "model": model,
    }
