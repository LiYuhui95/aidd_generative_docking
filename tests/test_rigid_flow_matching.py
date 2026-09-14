"""Tests for rigid-body Flow Matching pose construction (pure math, no model)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aidd_generative_docking.diffusion.rigid_flow_matching import (
    build_training_pair,
    integrate_rigid_pose,
    interpolate_rigid_pose,
    rigid_flow_matching_loss,
    sample_random_rigid_pose,
    sample_random_rotation,
)


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _random_ligand_pos(n_atoms: int, rng: np.random.Generator) -> torch.Tensor:
    return torch.as_tensor(rng.normal(size=(n_atoms, 3)) * 2.0, dtype=torch.float32)


def test_random_rotation_is_orthogonal_proper() -> None:
    rng = np.random.default_rng(0)
    r = sample_random_rotation(rng)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_random_rigid_pose_preserves_pairwise_distances() -> None:
    rng = np.random.default_rng(1)
    ligand_pos = _random_ligand_pos(8, rng)
    canonical = ligand_pos.numpy().astype(np.float64) - ligand_pos.numpy().astype(np.float64).mean(axis=0)
    r0, t0 = sample_random_rigid_pose(pocket_center=np.zeros(3), translation_noise_scale=3.0, rng=rng)
    x0 = (r0 @ canonical.T).T + t0

    np.testing.assert_allclose(_pairwise_distances(x0), _pairwise_distances(canonical), atol=1e-6)


def test_intermediate_pose_preserves_pairwise_distances_at_all_t() -> None:
    rng = np.random.default_rng(2)
    ligand_pos = _random_ligand_pos(10, rng)
    x1_dist = _pairwise_distances((ligand_pos.numpy().astype(np.float64)))

    for _ in range(5):
        pair = build_training_pair(ligand_pos, pocket_pos=torch.zeros(4, 3), translation_noise_scale=4.0, rng=rng)
        xt_dist = _pairwise_distances(pair.xt.numpy().astype(np.float64))
        np.testing.assert_allclose(xt_dist, x1_dist, atol=1e-4)


def test_interpolate_rigid_pose_endpoints_exact() -> None:
    rng = np.random.default_rng(3)
    canonical = _random_ligand_pos(6, rng).numpy().astype(np.float64)
    canonical = canonical - canonical.mean(axis=0)
    r0 = sample_random_rotation(rng)
    t0 = rng.normal(size=3)
    r1 = sample_random_rotation(rng)
    t1 = rng.normal(size=3)

    x0_expected = (r0 @ canonical.T).T + t0
    x1_expected = (r1 @ canonical.T).T + t1

    xt_at_0, v_trans, v_rot = interpolate_rigid_pose(canonical, r0, t0, r1, t1, t=0.0)
    xt_at_1, _, _ = interpolate_rigid_pose(canonical, r0, t0, r1, t1, t=1.0)

    np.testing.assert_allclose(xt_at_0, x0_expected, atol=1e-6)
    np.testing.assert_allclose(xt_at_1, x1_expected, atol=1e-6)
    np.testing.assert_allclose(v_trans, t1 - t0, atol=1e-10)
    assert np.isfinite(v_rot).all()


def test_interpolate_rigid_pose_midpoint_is_still_rigid() -> None:
    rng = np.random.default_rng(30)
    canonical = _random_ligand_pos(9, rng).numpy().astype(np.float64)
    canonical = canonical - canonical.mean(axis=0)
    r0, t0 = sample_random_rotation(rng), rng.normal(size=3)
    r1, t1 = sample_random_rotation(rng), rng.normal(size=3)

    xt, _, _ = interpolate_rigid_pose(canonical, r0, t0, r1, t1, t=0.5)
    np.testing.assert_allclose(_pairwise_distances(xt), _pairwise_distances(canonical), atol=1e-6)


def test_rigid_flow_matching_loss_zero_for_perfect_prediction() -> None:
    target_trans = torch.tensor([1.0, 2.0, 3.0])
    target_rot = torch.tensor([0.1, -0.2, 0.3])
    loss = rigid_flow_matching_loss(target_trans, target_rot, target_trans, target_rot)
    assert loss.item() == 0.0


def test_integrate_rigid_pose_preserves_distances_with_dummy_velocity() -> None:
    rng = np.random.default_rng(4)
    canonical = _random_ligand_pos(7, rng).numpy().astype(np.float64)
    canonical = canonical - canonical.mean(axis=0)
    r0 = sample_random_rotation(rng)
    t0 = rng.normal(size=3)

    def constant_velocity(_xt: np.ndarray, _t: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.5, -0.5, 0.2]), np.array([0.1, 0.0, -0.1])

    final = integrate_rigid_pose(constant_velocity, canonical, r0, t0, num_steps=20)

    np.testing.assert_allclose(_pairwise_distances(final), _pairwise_distances(canonical), atol=1e-4)


def test_build_training_pair_reproducible_with_seeded_rng() -> None:
    ligand_pos = torch.as_tensor(np.random.default_rng(42).normal(size=(5, 3)), dtype=torch.float32)
    pocket_pos = torch.zeros(3, 3)

    pair_a = build_training_pair(ligand_pos, pocket_pos, 2.0, np.random.default_rng(123))
    pair_b = build_training_pair(ligand_pos, pocket_pos, 2.0, np.random.default_rng(123))

    torch.testing.assert_close(pair_a.xt, pair_b.xt)
    torch.testing.assert_close(pair_a.v_trans_target, pair_b.v_trans_target)
    torch.testing.assert_close(pair_a.v_rot_target, pair_b.v_rot_target)
