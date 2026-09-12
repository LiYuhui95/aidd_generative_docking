"""Unit tests for RMSD and pose-success-rate evaluation metrics."""

import numpy as np
import pytest

from aidd_generative_docking.evaluation.metrics import pose_success_rate, rmsd


def test_rmsd_identical_poses_is_zero() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert rmsd(coords, coords) == pytest.approx(0.0)


def test_rmsd_known_value() -> None:
    coords_a = np.array([[0.0, 0.0, 0.0]])
    coords_b = np.array([[3.0, 4.0, 0.0]])
    assert rmsd(coords_a, coords_b) == pytest.approx(5.0)


def test_rmsd_shape_mismatch_raises() -> None:
    coords_a = np.zeros((3, 3))
    coords_b = np.zeros((2, 3))
    with pytest.raises(ValueError):
        rmsd(coords_a, coords_b)


def test_pose_success_rate_threshold() -> None:
    rmsds = np.array([0.5, 1.9, 2.0, 2.1, 5.0])
    assert pose_success_rate(rmsds, threshold_angstrom=2.0) == pytest.approx(3 / 5)


def test_pose_success_rate_empty_raises() -> None:
    with pytest.raises(ValueError):
        pose_success_rate(np.array([]))
