"""Tests for RigidBodyVectorField: shapes, equivariance, batching, gradients."""

from __future__ import annotations

import numpy as np
import torch

from aidd_generative_docking.models.rigid_equivariant import RigidBodyVectorField, predict_batch

LIGAND_FEATURE_DIM = 13
POCKET_FEATURE_DIM = 20
BOND_FEATURE_DIM = 4


def _make_ligand(n_atoms: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_atoms, LIGAND_FEATURE_DIM, generator=gen)
    pos = torch.randn(n_atoms, 3, generator=gen) * 2.0
    # simple path graph i -> i+1, both directions, so every model gets >=1 bonded neighbor
    src = list(range(n_atoms - 1)) + list(range(1, n_atoms))
    dst = list(range(1, n_atoms)) + list(range(n_atoms - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.zeros(len(src), BOND_FEATURE_DIM)
    edge_attr[:, 0] = 1.0  # all "SINGLE" bonds
    return {"x": x, "pos": pos, "edge_index": edge_index, "edge_attr": edge_attr}


def _make_pocket(n_residues: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.zeros(n_residues, POCKET_FEATURE_DIM)
    x[torch.arange(n_residues), torch.randint(0, POCKET_FEATURE_DIM, (n_residues,), generator=gen)] = 1.0
    pos = torch.randn(n_residues, 3, generator=gen) * 5.0
    return {"x": x, "pos": pos}


def _make_model(seed: int = 0) -> RigidBodyVectorField:
    torch.manual_seed(seed)
    return RigidBodyVectorField(LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2)


def _random_rotation_tensor(seed: int) -> torch.Tensor:
    from scipy.spatial.transform import Rotation

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


def test_no_nans_across_various_sizes() -> None:
    model = _make_model()
    for n_atoms, n_residues, t in [(2, 1, 0.0), (3, 4, 1.0), (20, 30, 0.5)]:
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
    model = _make_model()
    model.eval()
    ligand = _make_ligand(9, seed=42)
    pocket = _make_pocket(8, seed=43)
    t = torch.tensor([0.4])

    v_trans, v_rot = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], t
    )

    q = _random_rotation_tensor(seed=7)
    d = torch.tensor([3.0, -2.0, 1.5])

    ligand_pos_transformed = ligand["pos"] @ q.T + d
    pocket_pos_transformed = pocket["pos"] @ q.T + d

    v_trans_t, v_rot_t = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand_pos_transformed, pocket["x"], pocket_pos_transformed, t
    )

    # translation-invariant, rotation-equivariant: v' = Q v (no dependence on d)
    torch.testing.assert_close(v_trans_t, v_trans @ q.T, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(v_rot_t, v_rot @ q.T, atol=1e-4, rtol=1e-3)
