"""Tests for CrossInteractionRigidBodyVectorField: dynamic edges, RBF wiring,
equivariance, rigidity preservation, batching, gradients, and no-NaN cases.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from aidd_generative_docking.diffusion.rigid_flow_matching import integrate_rigid_pose, sample_random_rotation
from aidd_generative_docking.models.cross_interaction_equivariant import CrossInteractionRigidBodyVectorField, predict_batch

LIGAND_FEATURE_DIM = 13
POCKET_FEATURE_DIM = 20
BOND_FEATURE_DIM = 4


def _make_ligand(n_atoms: int, seed: int, spread: float = 2.0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_atoms, LIGAND_FEATURE_DIM, generator=gen)
    pos = torch.randn(n_atoms, 3, generator=gen) * spread
    src = list(range(n_atoms - 1)) + list(range(1, n_atoms))
    dst = list(range(1, n_atoms)) + list(range(n_atoms - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.zeros(len(src), BOND_FEATURE_DIM)
    edge_attr[:, 0] = 1.0
    return {"x": x, "pos": pos, "edge_index": edge_index, "edge_attr": edge_attr}


def _make_pocket(n_residues: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.zeros(n_residues, POCKET_FEATURE_DIM)
    x[torch.arange(n_residues), torch.randint(0, POCKET_FEATURE_DIM, (n_residues,), generator=gen)] = 1.0
    pos = torch.randn(n_residues, 3, generator=gen) * 5.0
    return {"x": x, "pos": pos}


def _make_model(seed: int = 0, cutoff: float = 8.0, num_rbf_centers: int = 16) -> CrossInteractionRigidBodyVectorField:
    torch.manual_seed(seed)
    return CrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM,
        POCKET_FEATURE_DIM,
        BOND_FEATURE_DIM,
        hidden_dim=16,
        num_layers=2,
        cross_edge_cutoff_angstrom=cutoff,
        num_rbf_centers=num_rbf_centers,
    )


def _random_rotation_tensor(seed: int) -> torch.Tensor:
    return torch.as_tensor(Rotation.random(random_state=np.random.default_rng(seed)).as_matrix(), dtype=torch.float32)


def test_forward_output_shapes() -> None:
    model = _make_model()
    ligand = _make_ligand(6, seed=1)
    pocket = _make_pocket(5, seed=2)
    v_trans, v_rot = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], torch.tensor([0.3])
    )
    assert v_trans.shape == (3,)
    assert v_rot.shape == (3,)
    assert torch.isfinite(v_trans).all()
    assert torch.isfinite(v_rot).all()


def test_cross_edges_change_when_ligand_pose_changes() -> None:
    model = _make_model(cutoff=6.0)
    pocket = _make_pocket(10, seed=2)

    pos_a = torch.zeros(5, 3)  # all ligand atoms at the pocket's local origin-ish region
    pos_a = pos_a + pocket["pos"].mean(dim=0, keepdim=True)
    ci_a, cj_a, _, _ = model.build_cross_edges(pos_a, pocket["pos"])

    # move the ligand far away: cross edges should shrink to (possibly) nothing
    pos_b = pos_a + 1000.0
    ci_b, cj_b, _, _ = model.build_cross_edges(pos_b, pocket["pos"])

    assert ci_a.numel() > 0  # near the pocket: should have found some neighbors
    assert ci_b.numel() == 0  # 1000 A away: cutoff excludes everything
    assert ci_a.numel() != ci_b.numel()


def test_cross_edges_zero_case_produces_no_nans() -> None:
    model = _make_model(cutoff=6.0)
    ligand = _make_ligand(4, seed=3)
    pocket = _make_pocket(5, seed=4)
    ligand["pos"] = ligand["pos"] + 10_000.0  # push ligand far outside any cutoff

    v_trans, v_rot = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], torch.tensor([0.5])
    )
    assert torch.isfinite(v_trans).all()
    assert torch.isfinite(v_rot).all()


def test_no_nans_across_various_sizes_and_cutoffs() -> None:
    for n_atoms, n_residues, t, cutoff in [(2, 1, 0.0, 8.0), (3, 4, 1.0, 3.0), (20, 30, 0.5, 100.0)]:
        model = _make_model(cutoff=cutoff)
        ligand = _make_ligand(n_atoms, seed=n_atoms)
        pocket = _make_pocket(n_residues, seed=n_residues)
        v_trans, v_rot = model(
            ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], torch.tensor([t])
        )
        assert torch.isfinite(v_trans).all()
        assert torch.isfinite(v_rot).all()


def test_forward_backward_gradients_are_finite() -> None:
    model = _make_model()
    ligand = _make_ligand(7, seed=3)
    pocket = _make_pocket(6, seed=4)
    v_trans, v_rot = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], torch.tensor([0.7])
    )
    loss = v_trans.pow(2).sum() + v_rot.pow(2).sum()
    loss.backward()

    n_params_with_grad = 0
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        n_params_with_grad += 1
    assert n_params_with_grad > 0


def test_predict_batch_matches_individual_calls() -> None:
    model = _make_model()
    ligands = [_make_ligand(5, seed=10), _make_ligand(8, seed=11)]
    pockets = [_make_pocket(4, seed=20), _make_pocket(7, seed=21)]
    ts = [torch.tensor([0.2]), torch.tensor([0.9])]

    v_trans_batch, v_rot_batch = predict_batch(
        model,
        [lig["x"] for lig in ligands],
        [lig["edge_index"] for lig in ligands],
        [lig["edge_attr"] for lig in ligands],
        [lig["pos"] for lig in ligands],
        [pkt["x"] for pkt in pockets],
        [pkt["pos"] for pkt in pockets],
        ts,
    )
    assert v_trans_batch.shape == (2, 3)
    assert v_rot_batch.shape == (2, 3)
    for i, (lig, pkt, t) in enumerate(zip(ligands, pockets, ts)):
        v_trans_i, v_rot_i = model(lig["x"], lig["edge_index"], lig["edge_attr"], lig["pos"], pkt["x"], pkt["pos"], t)
        torch.testing.assert_close(v_trans_batch[i], v_trans_i)
        torch.testing.assert_close(v_rot_batch[i], v_rot_i)


def test_rotation_and_translation_equivariance() -> None:
    model = _make_model(cutoff=20.0)  # generous cutoff so the transformed pose still has cross edges
    model.eval()
    ligand = _make_ligand(9, seed=42, spread=3.0)
    pocket = _make_pocket(8, seed=43)
    t = torch.tensor([0.4])

    v_trans, v_rot = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], t
    )

    q = _random_rotation_tensor(seed=7)
    d = torch.tensor([3.0, -2.0, 1.5])
    ligand_pos_t = ligand["pos"] @ q.T + d
    pocket_pos_t = pocket["pos"] @ q.T + d

    v_trans_t, v_rot_t = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand_pos_t, pocket["x"], pocket_pos_t, t
    )

    torch.testing.assert_close(v_trans_t, v_trans @ q.T, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(v_rot_t, v_rot @ q.T, atol=1e-4, rtol=1e-3)


def test_rigid_integration_preserves_internal_distances() -> None:
    """End-to-end check with the new model driving diffusion.integrate_rigid_pose:
    ligand internal (pairwise) distances must be exactly preserved throughout
    sampling, exactly as verified for the Milestone 2 model."""
    model = _make_model()
    model.eval()
    rng = np.random.default_rng(5)

    n_atoms = 8
    canonical = (torch.randn(n_atoms, 3, generator=torch.Generator().manual_seed(1)) * 2.0).numpy().astype(np.float64)
    canonical = canonical - canonical.mean(axis=0)
    ligand = _make_ligand(n_atoms, seed=1)
    pocket = _make_pocket(6, seed=2)

    r0 = sample_random_rotation(rng)
    t0 = rng.normal(size=3)

    def velocity_fn(xt: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        xt_tensor = torch.as_tensor(xt, dtype=torch.float32)
        with torch.no_grad():
            v_trans, v_rot = model(
                ligand["x"], ligand["edge_index"], ligand["edge_attr"], xt_tensor, pocket["x"], pocket["pos"], torch.tensor([t])
            )
        return v_trans.numpy().astype(np.float64), v_rot.numpy().astype(np.float64)

    final = integrate_rigid_pose(velocity_fn, canonical, r0, t0, num_steps=30)

    def pdist(c):
        diff = c[:, None, :] - c[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    np.testing.assert_allclose(pdist(final), pdist(canonical), atol=1e-4)
