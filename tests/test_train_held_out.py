"""Tests for the held-out mini-batch training loop (synthetic data, fast)."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from aidd_generative_docking.data.sample import ComplexSample
from aidd_generative_docking.models.soft_cross_interaction_equivariant import SoftCrossInteractionRigidBodyVectorField
from aidd_generative_docking.training.train_held_out import evaluate_val_loss, run_held_out_experiment

LIGAND_FEATURE_DIM = 13
POCKET_FEATURE_DIM = 20
BOND_FEATURE_DIM = 4


def _make_sample(complex_id: str, n_atoms: int, n_residues: int, seed: int) -> ComplexSample:
    gen = torch.Generator().manual_seed(seed)
    ligand_x = torch.randn(n_atoms, LIGAND_FEATURE_DIM, generator=gen)
    ligand_pos = torch.randn(n_atoms, 3, generator=gen) * 2.0
    src = list(range(n_atoms - 1)) + list(range(1, n_atoms))
    dst = list(range(1, n_atoms)) + list(range(n_atoms - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.zeros(len(src), BOND_FEATURE_DIM)
    edge_attr[:, 0] = 1.0
    ligand = Data(x=ligand_x, pos=ligand_pos, edge_index=edge_index, edge_attr=edge_attr)

    pocket_x = torch.zeros(n_residues, POCKET_FEATURE_DIM)
    pocket_x[torch.arange(n_residues), torch.randint(0, POCKET_FEATURE_DIM, (n_residues,), generator=gen)] = 1.0
    pocket_pos = torch.randn(n_residues, 3, generator=gen) * 5.0
    pocket = Data(x=pocket_x, pos=pocket_pos, edge_index=torch.zeros(2, 0, dtype=torch.long))

    return ComplexSample(complex_id=complex_id, ligand_resname="LIG", ligand=ligand, pocket=pocket)


def _make_model(seed: int = 0) -> SoftCrossInteractionRigidBodyVectorField:
    torch.manual_seed(seed)
    return SoftCrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=8, num_layers=1,
        length_scale_angstrom=8.0, num_rbf_centers=4,
    )


def test_run_held_out_experiment_end_to_end_shapes() -> None:
    train_samples = [_make_sample(f"train{i}", n_atoms=6, n_residues=5, seed=i) for i in range(4)]
    val_samples = [_make_sample(f"val{i}", n_atoms=5, n_residues=4, seed=100 + i) for i in range(2)]
    test_samples = [_make_sample(f"test{i}", n_atoms=7, n_residues=6, seed=200 + i) for i in range(3)]

    model = _make_model()
    result = run_held_out_experiment(
        model,
        train_samples,
        val_samples,
        test_samples,
        num_epochs=4,
        batch_size=2,
        learning_rate=1e-3,
        translation_noise_scale=5.0,
        num_integration_steps=5,
        eval_every=2,
        seed=0,
        device="cpu",
    )

    assert len(result.train_loss_history) == 4
    assert all(v == v for v in result.train_loss_history)  # no NaN
    assert len(result.val_loss_history) >= 1
    assert result.best_epoch >= 0
    assert set(result.test_initial_rmsd.keys()) == {"test0", "test1", "test2"}
    assert set(result.test_final_rmsd.keys()) == {"test0", "test1", "test2"}
    for value in list(result.test_initial_rmsd.values()) + list(result.test_final_rmsd.values()):
        assert value == value and value >= 0.0  # finite, non-negative
    assert result.training_time_seconds > 0.0
    assert isinstance(result.model_state_dict, dict)


def test_checkpoint_selection_picks_lowest_val_loss() -> None:
    train_samples = [_make_sample(f"train{i}", n_atoms=6, n_residues=5, seed=i) for i in range(3)]
    val_samples = [_make_sample(f"val{i}", n_atoms=5, n_residues=4, seed=50 + i) for i in range(2)]
    test_samples = [_make_sample("test0", n_atoms=6, n_residues=5, seed=99)]

    model = _make_model(seed=1)
    result = run_held_out_experiment(
        model, train_samples, val_samples, test_samples,
        num_epochs=6, batch_size=3, learning_rate=1e-3, translation_noise_scale=5.0,
        num_integration_steps=5, eval_every=1, seed=1, device="cpu",
    )
    recorded_val_losses = [v for _epoch, v in result.val_loss_history]
    assert result.best_val_loss == min(recorded_val_losses)


def test_evaluate_val_loss_is_reproducible_given_fixed_pairs() -> None:
    from aidd_generative_docking.diffusion.rigid_flow_matching import build_training_pair
    import numpy as np

    val_samples = [_make_sample(f"val{i}", n_atoms=5, n_residues=4, seed=i) for i in range(3)]
    rng = np.random.default_rng(0)
    fixed_pairs = [build_training_pair(s.ligand.pos, s.pocket.pos, 5.0, rng) for s in val_samples]

    model = _make_model()
    loss_a = evaluate_val_loss(model, val_samples, fixed_pairs, device="cpu")
    loss_b = evaluate_val_loss(model, val_samples, fixed_pairs, device="cpu")
    assert loss_a == loss_b
