"""Held-out experiment, stage 2: train and evaluate.

Loads the dataset+split prepared by ``prepare_held_out_dataset.py``,
trains ``SoftCrossInteractionRigidBodyVectorField`` (imported and used
UNCHANGED) with mini-batch Flow Matching, selects the checkpoint with the
lowest validation loss, and evaluates the test set exactly once.

This is the project's first *generalization* experiment: unlike the
Milestones 2/3 smoke experiments (10 complexes, thousands of epochs,
explicitly an overfitting sanity check with no held-out data at all), the
640/80/80 train/val/test complexes here are disjoint by construction (see
``data.splitting``), the model never trains on validation or test
complexes, and the test set is touched exactly once, after checkpoint
selection.

Run with the sxt-torch environment, from the repo root:

    conda run -n sxt-torch python notebooks/run_held_out_experiment.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

from aidd_generative_docking.evaluation.metrics import pose_success_rate, rmsd
from aidd_generative_docking.models.soft_cross_interaction_equivariant import SoftCrossInteractionRigidBodyVectorField
from aidd_generative_docking.training.train_held_out import run_held_out_experiment

RAW_DIR = Path("data/raw")
DATASET_PICKLE_PATH = RAW_DIR / ".cache" / "held_out_dataset.pkl"
OUTPUT_DIR = Path("outputs/held_out")
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"

# Model hyperparameters: identical to the Milestone 3 ablation's soft model
# (models.soft_cross_interaction_equivariant.SoftCrossInteractionRigidBodyVectorField
# used UNCHANGED) -- not tuned for this dataset.
HIDDEN_DIM = 64
NUM_LAYERS = 2
LENGTH_SCALE_ANGSTROM = 8.0
NUM_RBF_CENTERS = 16

# Training configuration: a reasonable fixed choice given the calibrated
# ~3s/epoch on this dataset/GPU, not tuned against test performance (the
# test set is not touched until after checkpoint selection, below).
NUM_EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
TRANSLATION_NOISE_SCALE = 5.0  # same as Milestones 2/3
NUM_INTEGRATION_STEPS = 50  # same as Milestones 2/3
EVAL_EVERY = 5
SEED = 0
SUCCESS_THRESHOLDS_ANGSTROM = [2.0, 5.0]


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"device: {device} ({device_name})")

    with open(DATASET_PICKLE_PATH, "rb") as f:
        dataset = pickle.load(f)
    validated = dataset["validated"]
    split_by_id = dataset["split_by_complex_id"]

    train_samples = [v.sample for v in validated if split_by_id[v.complex_id] == "train"]
    val_samples = [v.sample for v in validated if split_by_id[v.complex_id] == "val"]
    test_samples = [v.sample for v in validated if split_by_id[v.complex_id] == "test"]
    print(f"train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    example = train_samples[0]
    torch.manual_seed(SEED)
    model = SoftCrossInteractionRigidBodyVectorField(
        ligand_feature_dim=example.ligand.x.shape[1],
        pocket_feature_dim=example.pocket.x.shape[1],
        bond_feature_dim=example.ligand.edge_attr.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        length_scale_angstrom=LENGTH_SCALE_ANGSTROM,
        num_rbf_centers=NUM_RBF_CENTERS,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    print(f"training for {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}...")
    result = run_held_out_experiment(
        model,
        train_samples,
        val_samples,
        test_samples,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        translation_noise_scale=TRANSLATION_NOISE_SCALE,
        num_integration_steps=NUM_INTEGRATION_STEPS,
        eval_every=EVAL_EVERY,
        seed=SEED,
        device=device,
    )

    print(f"training time: {result.training_time_seconds:.1f}s")
    if result.peak_gpu_memory_bytes is not None:
        print(f"peak GPU memory: {result.peak_gpu_memory_bytes / 1e6:.1f} MB")
    print(f"best epoch: {result.best_epoch} (val loss {result.best_val_loss:.4f})")
    print(f"final train loss: {result.train_loss_history[-1]:.4f} (initial: {result.train_loss_history[0]:.4f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(result.model_state_dict, CHECKPOINT_PATH)
    print(f"saved best checkpoint to {CHECKPOINT_PATH}")

    initial_values = list(result.test_initial_rmsd.values())
    final_values = list(result.test_final_rmsd.values())
    ligand_sizes_by_id = {v.complex_id: v.sample.ligand.x.shape[0] for v in validated}

    success_rates = {
        f"success_rate_at_{thr}A": pose_success_rate(final_values, threshold_angstrom=thr)
        for thr in SUCCESS_THRESHOLDS_ANGSTROM
    }

    per_complex = [
        {
            "complex_id": cid,
            "ligand_atoms": ligand_sizes_by_id[cid],
            "initial_rmsd": result.test_initial_rmsd[cid],
            "final_rmsd": result.test_final_rmsd[cid],
        }
        for cid in result.test_final_rmsd
    ]
    per_complex_sorted = sorted(per_complex, key=lambda r: r["final_rmsd"])
    best_cases = per_complex_sorted[:3]
    worst_cases = per_complex_sorted[-3:]
    median_idx = len(per_complex_sorted) // 2
    median_cases = per_complex_sorted[max(0, median_idx - 1) : median_idx + 2]

    # performance vs. ligand size: bin into terciles by atom count
    by_size = sorted(per_complex, key=lambda r: r["ligand_atoms"])
    n = len(by_size)
    tercile_bounds = [0, n // 3, 2 * n // 3, n]
    size_bins = []
    for lo, hi in zip(tercile_bounds[:-1], tercile_bounds[1:]):
        bucket = by_size[lo:hi]
        if not bucket:
            continue
        size_bins.append(
            {
                "atom_count_range": [bucket[0]["ligand_atoms"], bucket[-1]["ligand_atoms"]],
                "n_complexes": len(bucket),
                "mean_final_rmsd": sum(r["final_rmsd"] for r in bucket) / len(bucket),
                "median_final_rmsd": median([r["final_rmsd"] for r in bucket]),
            }
        )

    summary = {
        "device": device_name,
        "experiment_type": "held-out generalization (disjoint train/val/test, single test pass)",
        "dataset_sizes": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "model_hyperparameters": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "length_scale_angstrom": LENGTH_SCALE_ANGSTROM,
            "num_rbf_centers": NUM_RBF_CENTERS,
        },
        "training_hyperparameters": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "translation_noise_scale": TRANSLATION_NOISE_SCALE,
            "num_integration_steps": NUM_INTEGRATION_STEPS,
            "eval_every": EVAL_EVERY,
            "seed": SEED,
        },
        "trainable_parameters": n_params,
        "training_time_seconds": result.training_time_seconds,
        "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "train_loss_history": result.train_loss_history,
        "val_loss_history": result.val_loss_history,
        "mean_initial_rmsd": sum(initial_values) / len(initial_values),
        "median_initial_rmsd": median(initial_values),
        "mean_final_rmsd": sum(final_values) / len(final_values),
        "median_final_rmsd": median(final_values),
        **success_rates,
        "per_complex": per_complex,
        "best_cases": best_cases,
        "median_cases": median_cases,
        "worst_cases": worst_cases,
        "performance_vs_ligand_size_terciles": size_bins,
    }
    with open(OUTPUT_DIR / "held_out_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved results to {OUTPUT_DIR / 'held_out_results.json'}")

    print("\n=== TEST SET (evaluated once, best checkpoint) ===")
    print(f"mean RMSD  : {summary['mean_initial_rmsd']:.2f} A -> {summary['mean_final_rmsd']:.2f} A")
    print(f"median RMSD: {summary['median_initial_rmsd']:.2f} A -> {summary['median_final_rmsd']:.2f} A")
    for thr in SUCCESS_THRESHOLDS_ANGSTROM:
        print(f"success @ {thr} A: {summary[f'success_rate_at_{thr}A']:.1%}")
    print("\nbest cases:", best_cases)
    print("median cases:", median_cases)
    print("worst cases:", worst_cases)
    print("\nperformance vs ligand size (terciles):")
    for b in size_bins:
        print(f"  atoms {b['atom_count_range']}: n={b['n_complexes']}, mean RMSD={b['mean_final_rmsd']:.2f} A")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(result.train_loss_history, label="train")
        val_epochs = [e for e, _ in result.val_loss_history]
        val_losses = [v for _, v in result.val_loss_history]
        axes[0].plot(val_epochs, val_losses, label="val", marker="o", markersize=3)
        axes[0].axvline(result.best_epoch, color="gray", linestyle="--", linewidth=1, label="best checkpoint")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("rigid flow matching loss")
        axes[0].set_yscale("log")
        axes[0].set_title("train/val loss")
        axes[0].legend()

        axes[1].scatter(initial_values, final_values, s=15, alpha=0.6)
        lims = [0, max(max(initial_values), max(final_values)) * 1.05]
        axes[1].plot(lims, lims, color="gray", linestyle="--", linewidth=1)
        axes[1].axhline(2.0, color="green", linestyle=":", linewidth=1, label="2 A")
        axes[1].axhline(5.0, color="orange", linestyle=":", linewidth=1, label="5 A")
        axes[1].set_xlabel("initial RMSD (A)")
        axes[1].set_ylabel("final RMSD (A)")
        axes[1].set_title("test set: initial vs. final RMSD")
        axes[1].legend()

        sizes = [r["ligand_atoms"] for r in per_complex]
        axes[2].scatter(sizes, final_values, s=15, alpha=0.6)
        axes[2].set_xlabel("ligand size (atoms, incl. H)")
        axes[2].set_ylabel("final RMSD (A)")
        axes[2].set_title("test set: final RMSD vs. ligand size")

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "held_out_plots.png", dpi=150)
        print(f"\nsaved plots to {OUTPUT_DIR / 'held_out_plots.png'}")
    except ImportError:
        print("matplotlib not available; skipped plots")


if __name__ == "__main__":
    main()
