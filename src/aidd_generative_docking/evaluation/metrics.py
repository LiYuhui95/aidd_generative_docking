"""Pose evaluation metrics (stage 5): RMSD and pose-success rate.

These are plain numpy computations with no data-loading or model
dependency, so they are fully implemented and tested now as scaffolding.
"""

from __future__ import annotations

import numpy as np


def rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Root-mean-square deviation between two matching atom coordinate sets.

    Assumes ``coords_a`` and ``coords_b`` are already atom-aligned (same
    atom order, same reference frame) — no rigid superposition is done here.

    Args:
        coords_a: ``(num_atoms, 3)`` array.
        coords_b: ``(num_atoms, 3)`` array, same shape as ``coords_a``.

    Returns:
        Scalar RMSD in the same length units as the input coordinates.
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    if coords_a.shape != coords_b.shape:
        raise ValueError(f"Shape mismatch: {coords_a.shape} vs {coords_b.shape}")
    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))


def pose_success_rate(rmsds: np.ndarray, threshold_angstrom: float = 2.0) -> float:
    """Fraction of poses with RMSD at or below ``threshold_angstrom``.

    Args:
        rmsds: 1D array of per-pose RMSD values.
        threshold_angstrom: success threshold, default 2.0 Å (common
            docking-literature convention).

    Returns:
        Fraction in ``[0, 1]``.
    """
    rmsds = np.asarray(rmsds, dtype=np.float64)
    if rmsds.size == 0:
        raise ValueError("rmsds must be non-empty")
    return float(np.mean(rmsds <= threshold_angstrom))
