"""Tests for SoftCrossInteractionRigidBodyVectorField: the smooth envelope
itself, and the "no dead zone" property that motivates this ablation --
compared directly against the unmodified hard-cutoff model.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from aidd_generative_docking.diffusion.rigid_flow_matching import integrate_rigid_pose, sample_random_rotation
from aidd_generative_docking.models.cross_interaction_equivariant import CrossInteractionRigidBodyVectorField
from aidd_generative_docking.models.soft_cross_interaction_equivariant import (
    SoftCrossInteractionRigidBodyVectorField,
    predict_batch,
    smooth_distance_envelope,
)

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


def _make_soft_model(seed: int = 0, length_scale: float = 8.0, num_rbf_centers: int = 16) -> SoftCrossInteractionRigidBodyVectorField:
    torch.manual_seed(seed)
    return SoftCrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2,
        length_scale_angstrom=length_scale, num_rbf_centers=num_rbf_centers,
    )


def _random_rotation_tensor(seed: int) -> torch.Tensor:
    return torch.as_tensor(Rotation.random(random_state=np.random.default_rng(seed)).as_matrix(), dtype=torch.float32)


# --- envelope function itself ---


def test_smooth_distance_envelope_reference_values() -> None:
    d = torch.tensor([0.0, 8.0, 16.0, 800.0, 8_000_000.0])
    w = smooth_distance_envelope(d, length_scale=8.0)
    assert w[0].item() == 1.0
    assert abs(w[1].item() - 0.5) < 1e-6
    assert w[2].item() < w[1].item()
    assert w[3].item() > 0.0
    assert w[4].item() > 0.0  # never exactly zero, however far


def test_smooth_distance_envelope_monotonically_decreasing() -> None:
    d = torch.linspace(0.0, 1000.0, 200)
    w = smooth_distance_envelope(d, length_scale=8.0)
    assert (w[1:] <= w[:-1]).all()


def test_smooth_distance_envelope_always_finite_and_positive() -> None:
    d = torch.tensor([0.0, 1e3, 1e6, 1e9])
    w = smooth_distance_envelope(d, length_scale=8.0)
    assert torch.isfinite(w).all()
    assert (w > 0.0).all()


# --- the actual dead-zone hypothesis test: soft vs. unmodified hard model ---


def test_soft_and_hard_models_have_identical_parameter_count() -> None:
    """The ablation changes masking -> weighting only; architecture (and
    therefore parameter count) should be exactly unchanged."""
    hard_model = CrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2,
        cross_edge_cutoff_angstrom=8.0, num_rbf_centers=16,
    )
    soft_model = SoftCrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2,
        length_scale_angstrom=8.0, num_rbf_centers=16,
    )
    n_hard = sum(p.numel() for p in hard_model.parameters())
    n_soft = sum(p.numel() for p in soft_model.parameters())
    assert n_hard == n_soft


def test_same_seed_gives_identical_initial_weights() -> None:
    """Both classes construct submodules in the same order/shapes, so a
    shared seed should produce bit-identical initializations -- needed for
    a true apples-to-apples forward-output comparison below."""
    torch.manual_seed(123)
    hard_model = CrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2,
        cross_edge_cutoff_angstrom=8.0, num_rbf_centers=16,
    )
    torch.manual_seed(123)
    soft_model = SoftCrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2,
        length_scale_angstrom=8.0, num_rbf_centers=16,
    )
    for (name_h, p_h), (name_s, p_s) in zip(hard_model.state_dict().items(), soft_model.state_dict().items()):
        assert p_h.shape == p_s.shape, f"{name_h} vs {name_s}"
        torch.testing.assert_close(p_h, p_s)


def test_hard_model_output_is_dead_beyond_cutoff_soft_model_is_not() -> None:
    """Direct test of the dead-zone hypothesis: with identical weights, the
    hard-cutoff model's output stops depending on distance at all once
    every pair is beyond its cutoff (a genuine dead zone), while the soft
    model's output keeps changing (however slightly) with distance."""
    torch.manual_seed(123)
    hard_model = CrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2,
        cross_edge_cutoff_angstrom=8.0, num_rbf_centers=16,
    )
    torch.manual_seed(123)
    soft_model = SoftCrossInteractionRigidBodyVectorField(
        LIGAND_FEATURE_DIM, POCKET_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2,
        length_scale_angstrom=8.0, num_rbf_centers=16,
    )

    ligand = _make_ligand(6, seed=1)
    pocket = _make_pocket(5, seed=2)
    t = torch.tensor([0.5])

    # Offsets chosen well within float32's safe precision range (avoiding
    # spurious cdist round-off near coordinates ~1e5-1e6) while still being
    # far beyond the 8 A cutoff either way.
    offsets = (1_000.0, 10_000.0)
    hard_outputs, soft_outputs = [], []
    for offset in offsets:
        far_pos = ligand["pos"] + offset
        cross_i, _, _, _ = hard_model.build_cross_edges(far_pos, pocket["pos"])
        assert cross_i.numel() == 0, "test setup invalid: expected zero hard-model cross edges at this offset"

        with torch.no_grad():
            hv, _ = hard_model(ligand["x"], ligand["edge_index"], ligand["edge_attr"], far_pos, pocket["x"], pocket["pos"], t)
            sv, _ = soft_model(ligand["x"], ligand["edge_index"], ligand["edge_attr"], far_pos, pocket["x"], pocket["pos"], t)
        hard_outputs.append(hv.clone())
        soft_outputs.append(sv.clone())

    # Hard model: with zero cross edges at both offsets, the only path left
    # is the ligand-ligand bond term, which is translation-invariant, so
    # outputs should match up to float32 round-off from differencing
    # large-magnitude coordinates (~1e-5), not a real distance-dependence.
    hard_diff = (hard_outputs[0] - hard_outputs[1]).abs().max().item()
    soft_diff = (soft_outputs[0] - soft_outputs[1]).abs().max().item()

    assert hard_diff < 1e-4, f"hard model should be (numerically) dead beyond cutoff, got diff={hard_diff}"
    # soft model: output still differs meaningfully between the two distances --
    # always some (non-noise-level) signal, an order of magnitude above the
    # float32 noise floor measured on the hard model above.
    assert soft_diff > 10 * hard_diff, f"expected soft model to show real distance-dependence: soft={soft_diff}, hard={hard_diff}"


def test_soft_model_gradient_nonzero_far_from_pocket() -> None:
    """Even when every ligand atom starts far outside the nominal length
    scale, the soft model's cross-interaction parameters must still
    receive a nonzero gradient (the concrete mechanism behind "no dead
    zone": training gets *some* signal regardless of starting distance)."""
    model = _make_soft_model()
    ligand = _make_ligand(5, seed=3)
    pocket = _make_pocket(4, seed=4)
    ligand["pos"] = ligand["pos"] + 5_000.0

    v_trans, v_rot = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], torch.tensor([0.5])
    )
    loss = v_trans.pow(2).sum() + v_rot.pow(2).sum()
    loss.backward()

    cross_param_grad_norms = [
        p.grad.norm().item()
        for name, p in model.named_parameters()
        if "cross" in name and p.grad is not None
    ]
    assert len(cross_param_grad_norms) > 0
    assert any(g > 0.0 for g in cross_param_grad_norms)


# --- standard correctness suite (shapes, no-NaN, gradients, batching, equivariance, rigidity) ---


def test_forward_output_shapes() -> None:
    model = _make_soft_model()
    ligand = _make_ligand(6, seed=1)
    pocket = _make_pocket(5, seed=2)
    v_trans, v_rot = model(
        ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], torch.tensor([0.3])
    )
    assert v_trans.shape == (3,)
    assert v_rot.shape == (3,)
    assert torch.isfinite(v_trans).all()
    assert torch.isfinite(v_rot).all()


def test_no_nans_across_various_sizes_and_distances() -> None:
    for n_atoms, n_residues, t, offset in [(2, 1, 0.0, 0.0), (3, 4, 1.0, 500.0), (20, 30, 0.5, 1e6)]:
        model = _make_soft_model()
        ligand = _make_ligand(n_atoms, seed=n_atoms)
        ligand["pos"] = ligand["pos"] + offset
        pocket = _make_pocket(n_residues, seed=n_residues)
        v_trans, v_rot = model(
            ligand["x"], ligand["edge_index"], ligand["edge_attr"], ligand["pos"], pocket["x"], pocket["pos"], torch.tensor([t])
        )
        assert torch.isfinite(v_trans).all()
        assert torch.isfinite(v_rot).all()


def test_forward_backward_gradients_are_finite() -> None:
    model = _make_soft_model()
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
    model = _make_soft_model()
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
    model = _make_soft_model()
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
    model = _make_soft_model()
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
