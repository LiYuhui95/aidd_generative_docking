"""Held-out generalization experiment: mini-batch training with validation-
based checkpoint selection, and a single test-set evaluation pass.

Additive, new training loop -- distinct from ``smoke_train_rigid.py``
(the 10-complex full-batch overfitting sanity check from Milestones 2/3,
left completely unmodified). This module trains
``SoftCrossInteractionRigidBodyVectorField`` (imported unchanged) on a
real train/val/test split of a few hundred complexes each, using the same
underlying rigid-body Flow Matching machinery
(``diffusion.rigid_flow_matching``, also unmodified).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from aidd_generative_docking.data.sample import ComplexSample
from aidd_generative_docking.diffusion.rigid_flow_matching import (
    build_training_pair,
    integrate_rigid_pose,
    rigid_flow_matching_loss,
    sample_random_rigid_pose,
)
from aidd_generative_docking.evaluation.metrics import rmsd


def _canonical_shape_and_pocket_center(sample: ComplexSample) -> tuple[np.ndarray, np.ndarray]:
    x1 = sample.ligand.pos.numpy().astype(np.float64)
    canonical_shape = x1 - x1.mean(axis=0)
    pocket_center = sample.pocket.pos.numpy().astype(np.float64).mean(axis=0)
    return canonical_shape, pocket_center


def _compute_loss_for_sample(model, sample: ComplexSample, pair, device: str) -> torch.Tensor:
    v_trans_pred, v_rot_pred = model(
        sample.ligand.x.to(device),
        sample.ligand.edge_index.to(device),
        sample.ligand.edge_attr.to(device),
        pair.xt.to(device),
        sample.pocket.x.to(device),
        sample.pocket.pos.to(device),
        torch.tensor([pair.t], dtype=torch.float32, device=device),
    )
    return rigid_flow_matching_loss(v_trans_pred, v_rot_pred, pair.v_trans_target.to(device), pair.v_rot_target.to(device))


@dataclass
class HeldOutResult:
    train_loss_history: list[float]
    val_loss_history: list[tuple[int, float]]  # (epoch, val_loss)
    best_epoch: int
    best_val_loss: float
    test_initial_rmsd: dict[str, float]
    test_final_rmsd: dict[str, float]
    training_time_seconds: float
    peak_gpu_memory_bytes: int | None
    model_state_dict: dict


def evaluate_val_loss(
    model,
    val_samples: list[ComplexSample],
    fixed_val_pairs: list,
    device: str,
) -> float:
    """Average Flow Matching loss over the validation set, at fixed (x0, t)
    pairs generated once (so successive checkpoints are compared fairly)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for sample, pair in zip(val_samples, fixed_val_pairs):
            total += _compute_loss_for_sample(model, sample, pair, device).item()
    model.train()
    return total / len(val_samples)


def run_held_out_experiment(
    model,
    train_samples: list[ComplexSample],
    val_samples: list[ComplexSample],
    test_samples: list[ComplexSample],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    translation_noise_scale: float,
    num_integration_steps: int,
    eval_every: int,
    seed: int,
    device: str,
) -> HeldOutResult:
    """Train ``model`` with mini-batch Flow Matching, select the checkpoint
    with the lowest validation loss, and evaluate the test set exactly once.

    Args:
        model: an already-constructed (untrained) ``nn.Module`` implementing
            the standard ``forward(ligand_x, ligand_edge_index,
            ligand_edge_attr, xt, pocket_x, pocket_pos, t)`` signature
            (e.g. ``SoftCrossInteractionRigidBodyVectorField``, used
            unchanged by this function).
        train_samples, val_samples, test_samples: pre-split ``ComplexSample`` lists.
        num_epochs: passes over the shuffled training set.
        batch_size: complexes per gradient step (a plain Python-loop
            mini-batch, consistent with this project's existing batching
            approach -- see ``models.*.predict_batch``).
        learning_rate: Adam learning rate.
        translation_noise_scale, num_integration_steps: see
            ``diffusion.rigid_flow_matching``.
        eval_every: compute validation loss every this many epochs.
        seed: seed for model-independent randomness (epoch shuffling,
            training pair sampling, fixed val/test starting poses). The
            model's own initialization is whatever state it was
            constructed with by the caller.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        A ``HeldOutResult`` with loss curves, the best epoch/state dict,
        and the one-shot test-set RMSD evaluation.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    shuffle_rng = random.Random(seed)
    train_pair_rng = np.random.default_rng(seed)
    val_pair_rng = np.random.default_rng(seed + 1)
    test_eval_rng = np.random.default_rng(seed + 2)

    # Fixed validation (x0, t) pairs, generated once, reused for every
    # checkpoint comparison during training.
    fixed_val_pairs = [
        build_training_pair(s.ligand.pos, s.pocket.pos, translation_noise_scale, val_pair_rng) for s in val_samples
    ]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_loss_history: list[float] = []
    val_loss_history: list[tuple[int, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = copy.deepcopy(model.state_dict())

    start_time = time.time()
    train_indices = list(range(len(train_samples)))

    for epoch in range(num_epochs):
        shuffle_rng.shuffle(train_indices)
        epoch_loss_total = 0.0
        epoch_loss_count = 0

        for batch_start in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[batch_start : batch_start + batch_size]
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)
            for idx in batch_indices:
                sample = train_samples[idx]
                pair = build_training_pair(sample.ligand.pos, sample.pocket.pos, translation_noise_scale, train_pair_rng)
                batch_loss = batch_loss + _compute_loss_for_sample(model, sample, pair, device)
            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()
            optimizer.step()

            epoch_loss_total += batch_loss.item() * len(batch_indices)
            epoch_loss_count += len(batch_indices)

        train_loss_history.append(epoch_loss_total / epoch_loss_count)

        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            val_loss = evaluate_val_loss(model, val_samples, fixed_val_pairs, device)
            val_loss_history.append((epoch, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

    training_time_seconds = time.time() - start_time
    peak_gpu_memory_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else None

    # Load the best checkpoint before the single test-set evaluation pass.
    model.load_state_dict(best_state_dict)
    model.eval()

    test_initial_rmsd: dict[str, float] = {}
    test_final_rmsd: dict[str, float] = {}
    for sample in test_samples:
        canonical_shape, pocket_center = _canonical_shape_and_pocket_center(sample)
        x1 = sample.ligand.pos.numpy().astype(np.float64)
        r0, t0 = sample_random_rigid_pose(pocket_center, translation_noise_scale, test_eval_rng)

        x0 = (r0 @ canonical_shape.T).T + t0
        test_initial_rmsd[sample.complex_id] = rmsd(x0, x1)

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
        test_final_rmsd[sample.complex_id] = rmsd(x_final, x1)

    return HeldOutResult(
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        test_initial_rmsd=test_initial_rmsd,
        test_final_rmsd=test_final_rmsd,
        training_time_seconds=training_time_seconds,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        model_state_dict=best_state_dict,
    )
