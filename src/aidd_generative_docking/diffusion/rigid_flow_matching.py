"""Rigid-body (SE(3)) conditional Flow Matching: pose construction, path, objective, sampling.

Milestone 2, v1 assumptions (see ARCHITECTURE.md / README for the broader
plan): the protein pocket is fixed, ligand *internal* geometry is fixed, and
the ligand only moves via a single rigid-body transform (rotation +
translation) -- no per-atom deformation, no torsions.

Representation
--------------
A ligand pose is parameterized as ``x = R @ L + t`` where ``L`` is the
ligand's canonical shape (its experimental bound-pose coordinates,
recentered to zero centroid -- so by construction the *target* pose
corresponds to ``R1 = I``, ``t1 = centroid(x1)``), ``R`` is a 3x3 rotation
matrix (SO(3)), and ``t`` is a translation vector in R^3.

- **Translation** is linearly interpolated (``tt = (1-t) t0 + t t1``), same
  as vanilla conditional Flow Matching's linear path -- translation lives
  in flat Euclidean space, so there is nothing SE(3)-specific to do here.
  Its velocity is constant in flow-time: ``v_trans = t1 - t0``.
- **Rotation** is interpolated along the SO(3) *geodesic* from R0 to R1,
  ``Rt = expm(t * Omega_hat) @ R0``, where ``Omega`` (a 3D axis-angle /
  rotation vector) is the constant world-frame angular velocity satisfying
  ``expm(Omega_hat) @ R0 = R1``. This is the direct SO(3) analogue of a
  linear path: "constant angular velocity" is to rotations what "constant
  velocity" is to translations, and the resulting ``Rt`` is an exact
  rotation matrix at every ``t``, so ``xt = Rt @ L + tt`` is an exact rigid
  transform of ``L`` at every ``t`` -- ligand internal distances are
  preserved *exactly*, not just approximately.

All SO(3) bookkeeping (random sampling, matrix log/exp for geodesics) is
done with ``scipy.spatial.transform.Rotation`` on plain numpy arrays: this
part is training-*data* construction (building x0, xt, and the targets),
not part of the differentiable model, so there is no need for it to run on
GPU or carry gradients -- keeping it in well-tested scipy code is simpler
and safer than hand-rolling matrix log/exp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def sample_random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Sample a rotation matrix uniformly from SO(3) (Haar measure)."""
    return Rotation.random(random_state=rng).as_matrix()


def sample_random_rigid_pose(
    pocket_center: np.ndarray,
    translation_noise_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random source rigid pose ``(R0, t0)``.

    ``R0`` is uniform over SO(3). ``t0`` is centered on the pocket's own
    centroid plus isotropic Gaussian noise, rather than uniform over all of
    R^3: the model only needs to learn to correct a plausible "somewhere
    near the right pocket but wrong orientation/offset" initialization
    (mirroring blind-docking-style search-box initialization), not to find
    the pocket from anywhere in space -- finding the pocket location itself
    is out of scope for this rigid-body v1.

    Args:
        pocket_center: ``(3,)`` centroid of the protein pocket.
        translation_noise_scale: standard deviation (Angstrom) of the
            isotropic Gaussian offset from ``pocket_center``.
        rng: numpy random generator (explicit, for reproducibility).

    Returns:
        ``(R0, t0)``: a ``(3, 3)`` rotation matrix and a ``(3,)`` translation.
    """
    r0 = sample_random_rotation(rng)
    t0 = pocket_center + rng.normal(scale=translation_noise_scale, size=3)
    return r0, t0


def interpolate_rigid_pose(
    canonical_shape: np.ndarray,
    r0: np.ndarray,
    t0: np.ndarray,
    r1: np.ndarray,
    t1: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the rigid-body geodesic path at flow time ``t``.

    Pure function (no randomness), factored out of ``build_training_pair``
    so the path's endpoint/geodesic behavior can be tested directly.

    Args:
        canonical_shape: ``(N, 3)`` centroid-zero ligand shape.
        r0, t0: source rotation/translation (flow time 0).
        r1, t1: target rotation/translation (flow time 1).
        t: flow time in ``[0, 1]``.

    Returns:
        ``(xt, v_trans, v_rot)``: the interpolated pose ``(N, 3)`` and the
        two constant velocities (translation, world-frame angular) that
        generate this path.
    """
    omega = Rotation.from_matrix(r1 @ r0.T).as_rotvec()
    r_t = Rotation.from_rotvec(t * omega).as_matrix() @ r0
    trans_t = (1.0 - t) * t0 + t * t1

    xt = (r_t @ canonical_shape.T).T + trans_t
    v_trans = t1 - t0
    v_rot = omega
    return xt, v_trans, v_rot


@dataclass
class RigidTrainingPair:
    """One Flow Matching training example for the rigid-body path.

    Attributes:
        canonical_shape: ``(N, 3)`` ligand shape, centroid-zero (the
            experimental pose recentered; this is what stays fixed as the
            ligand "moves" -- only R and t change).
        t: sampled flow time in ``[0, 1]``.
        xt: ``(N, 3)`` intermediate rigid pose at time ``t``.
        v_trans_target: ``(3,)`` target translation velocity, ``t1 - t0``.
        v_rot_target: ``(3,)`` target angular velocity (world-frame
            axis-angle / rotation vector).
    """

    canonical_shape: torch.Tensor
    t: float
    xt: torch.Tensor
    v_trans_target: torch.Tensor
    v_rot_target: torch.Tensor


def build_training_pair(
    ligand_pos: torch.Tensor,
    pocket_pos: torch.Tensor,
    translation_noise_scale: float,
    rng: np.random.Generator,
) -> RigidTrainingPair:
    """Construct one rigid-body Flow Matching training pair from a bound pose.

    Args:
        ligand_pos: ``(N, 3)`` experimental bound-pose ligand coordinates
            (the target, ``x1``), in the same frame as ``pocket_pos``.
        pocket_pos: ``(M, 3)`` pocket residue Cα coordinates, used only to
            center the random source pose nearby (see
            ``sample_random_rigid_pose``).
        translation_noise_scale: see ``sample_random_rigid_pose``.
        rng: numpy random generator.

    Returns:
        A populated ``RigidTrainingPair``.
    """
    x1 = ligand_pos.detach().cpu().numpy().astype(np.float64)
    t1 = x1.mean(axis=0)
    canonical_shape = x1 - t1  # R1 = I by construction

    pocket_center = pocket_pos.detach().cpu().numpy().astype(np.float64).mean(axis=0)
    r0, t0 = sample_random_rigid_pose(pocket_center, translation_noise_scale, rng)

    t = float(rng.uniform(0.0, 1.0))
    r1 = np.eye(3)  # canonical_shape is defined so that R1 = I, t1 = x1's own centroid
    xt, v_trans_target, v_rot_target = interpolate_rigid_pose(canonical_shape, r0, t0, r1, t1, t)

    return RigidTrainingPair(
        canonical_shape=torch.as_tensor(canonical_shape, dtype=torch.float32),
        t=t,
        xt=torch.as_tensor(xt, dtype=torch.float32),
        v_trans_target=torch.as_tensor(v_trans_target, dtype=torch.float32),
        v_rot_target=torch.as_tensor(v_rot_target, dtype=torch.float32),
    )


def rigid_flow_matching_loss(
    v_trans_pred: torch.Tensor,
    v_rot_pred: torch.Tensor,
    v_trans_target: torch.Tensor,
    v_rot_target: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error between predicted and target rigid-body velocities.

    Translation and rotation terms are summed unweighted: with
    ``translation_noise_scale`` chosen on the same order as typical
    rotation-vector magnitudes (angles up to pi), both terms sit in a
    comparable numeric range, so this is a reasonable v1 default rather
    than a carefully tuned loss weighting.
    """
    trans_loss = torch.mean((v_trans_pred - v_trans_target) ** 2)
    rot_loss = torch.mean((v_rot_pred - v_rot_target) ** 2)
    return trans_loss + rot_loss


def integrate_rigid_pose(
    velocity_fn,
    canonical_shape: np.ndarray,
    r0: np.ndarray,
    t0: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """Euler-integrate a learned SE(3) vector field from flow time 0 to 1.

    At each step, ``velocity_fn(xt, t)`` is called with the *current* rigid
    pose's atom coordinates and returns ``(v_trans, v_rot)`` as numpy
    arrays; the rotation is advanced along its own local geodesic
    (``R <- expm(dt * v_rot_hat) @ R``) and the translation by a plain Euler
    step, and a fresh ``xt`` is reconstructed from the updated ``(R, t)``.
    Reconstructing from ``(R, t)`` every step (rather than moving atoms
    independently) is what guarantees the sampled trajectory stays exactly
    rigid throughout integration, not just at the endpoints.

    Args:
        velocity_fn: callable ``(xt: np.ndarray (N,3), t: float) ->
            (v_trans: np.ndarray (3,), v_rot: np.ndarray (3,))``.
        canonical_shape: ``(N, 3)`` centroid-zero ligand shape.
        r0: ``(3, 3)`` initial rotation.
        t0: ``(3,)`` initial translation.
        num_steps: number of Euler integration steps.

    Returns:
        ``(N, 3)`` final predicted ligand pose.
    """
    dt = 1.0 / num_steps
    r_cur = r0.copy()
    trans_cur = t0.copy()

    for step in range(num_steps):
        t_cur = step * dt
        xt = (r_cur @ canonical_shape.T).T + trans_cur
        v_trans, v_rot = velocity_fn(xt, t_cur)
        r_cur = Rotation.from_rotvec(dt * v_rot).as_matrix() @ r_cur
        trans_cur = trans_cur + dt * v_trans

    return (r_cur @ canonical_shape.T).T + trans_cur
