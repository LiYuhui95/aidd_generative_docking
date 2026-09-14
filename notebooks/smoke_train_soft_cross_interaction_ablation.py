"""Milestone 3 ablation: hard-cutoff vs. soft distance-weighted cross interaction.

Tests the hypothesis that CrossInteractionRigidBodyVectorField's hard
distance cutoff creates gradient dead zones when a randomized ligand pose
starts far from the pocket, by comparing it against
SoftCrossInteractionRigidBodyVectorField (dense connectivity + a smooth,
full-support distance envelope) under an IDENTICAL protocol: same 10
complexes, seed, epochs, hidden size, integration steps, optimizer, and
noise settings as Milestone 3. Neither Milestone 2 nor Milestone 3's
implementations are modified by this script.

Pipeline/hypothesis sanity check ONLY (see README.md / ARCHITECTURE.md) --
not a benchmark, and hyperparameters are NOT tuned to make either model
look better; the point is to test the dead-zone hypothesis, not to
optimize performance.

Requires the licensed PDBBind archive at data/raw/P-L.tar.gz. Saves a JSON
comparison summary and plots under outputs/milestone3_soft/.

Run with the sxt-torch environment, from the repo root:

    conda run -n sxt-torch python notebooks/smoke_train_soft_cross_interaction_ablation.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from aidd_generative_docking.data.pdb_io import extract_pdbbind_complexes
from aidd_generative_docking.data.sample import build_training_sample_from_files
from aidd_generative_docking.evaluation.metrics import pose_success_rate
from aidd_generative_docking.models.cross_interaction_equivariant import CrossInteractionRigidBodyVectorField
from aidd_generative_docking.models.soft_cross_interaction_equivariant import SoftCrossInteractionRigidBodyVectorField
from aidd_generative_docking.training.smoke_train_rigid import run_smoke_experiment

RAW_DIR = Path("data/raw")
ARCHIVE_PATH = RAW_DIR / "P-L.tar.gz"
CACHE_DIR = RAW_DIR / ".cache" / "pdbbind_archive"
OUTPUT_DIR = Path("outputs/milestone3_soft")

# Identical to milestone 3's smoke experiment for a controlled ablation.
SAMPLE_COMPLEX_IDS = ["1stp", "1azm", "6cha", "4cts", "6hrp", "2tpi", "1oit", "1b39", "6q36", "6uyy"]
POCKET_RADIUS_ANGSTROM = 10.0
EDGE_CUTOFF_ANGSTROM = 8.0
NUM_EPOCHS = 3000
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2
TRANSLATION_NOISE_SCALE = 5.0
NUM_INTEGRATION_STEPS = 50
SEED = 0
SUCCESS_THRESHOLD_ANGSTROM = 2.0

# Cross-interaction hyperparameters: identical value reused for the hard
# model's cutoff and the soft model's length scale (the one thing that
# differs between the two is masking vs. weighting, not this number).
CROSS_EDGE_CUTOFF_ANGSTROM = 8.0
NUM_RBF_CENTERS = 16


def load_samples():
    paths_by_id = extract_pdbbind_complexes(ARCHIVE_PATH, SAMPLE_COMPLEX_IDS, CACHE_DIR)
    samples = []
    for complex_id in SAMPLE_COMPLEX_IDS:
        paths = paths_by_id[complex_id]
        samples.append(
            build_training_sample_from_files(
                protein_pdb_path=paths["protein"],
                ligand_sdf_path=paths["ligand_sdf"],
                ligand_mol2_path=paths["ligand_mol2"],
                complex_id=complex_id,
                pocket_radius_angstrom=POCKET_RADIUS_ANGSTROM,
                edge_cutoff_angstrom=EDGE_CUTOFF_ANGSTROM,
            )
        )
    return samples


def median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def summarize(name: str, result: dict, elapsed: float) -> dict:
    n_params = sum(p.numel() for p in result["model"].parameters() if p.requires_grad)
    initial = result["initial_rmsd"]
    final = result["final_rmsd"]
    initial_values = list(initial.values())
    final_values = list(final.values())
    return {
        "name": name,
        "trainable_parameters": n_params,
        "training_time_seconds": elapsed,
        "initial_training_loss": result["loss_history"][0],
        "final_training_loss": result["loss_history"][-1],
        "loss_history": result["loss_history"],
        "per_complex_initial_rmsd": initial,
        "per_complex_final_rmsd": final,
        "mean_initial_rmsd": sum(initial_values) / len(initial_values),
        "median_initial_rmsd": median(initial_values),
        "mean_final_rmsd": sum(final_values) / len(final_values),
        "median_final_rmsd": median(final_values),
        "success_rate_at_2A": pose_success_rate(final_values, threshold_angstrom=SUCCESS_THRESHOLD_ANGSTROM),
    }


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['name']} ===")
    print(f"trainable parameters : {summary['trainable_parameters']:,}")
    print(f"training time        : {summary['training_time_seconds']:.1f}s")
    print(f"training loss        : {summary['initial_training_loss']:.4f} -> {summary['final_training_loss']:.4f}")
    print(f"mean final RMSD      : {summary['mean_final_rmsd']:.2f} A (initial {summary['mean_initial_rmsd']:.2f} A)")
    print(f"median final RMSD    : {summary['median_final_rmsd']:.2f} A (initial {summary['median_initial_rmsd']:.2f} A)")
    print(f"success rate @ 2A    : {summary['success_rate_at_2A']:.0%}")
    print(f"{'complex':8s} {'initial':>10s} {'final':>10s}")
    for complex_id in SAMPLE_COMPLEX_IDS:
        print(
            f"{complex_id:8s} {summary['per_complex_initial_rmsd'][complex_id]:10.2f} "
            f"{summary['per_complex_final_rmsd'][complex_id]:10.2f}"
        )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"device: {device} ({device_name})")

    samples = load_samples()
    for s in samples:
        print(f"  loaded {s.complex_id}: {s.ligand.x.shape[0]} ligand atoms, {s.pocket.x.shape[0]} pocket residues")

    common_kwargs = dict(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        translation_noise_scale=TRANSLATION_NOISE_SCALE,
        num_integration_steps=NUM_INTEGRATION_STEPS,
        seed=SEED,
        device=device,
    )

    def hard_model_factory():
        example = samples[0]
        return CrossInteractionRigidBodyVectorField(
            ligand_feature_dim=example.ligand.x.shape[1],
            pocket_feature_dim=example.pocket.x.shape[1],
            bond_feature_dim=example.ligand.edge_attr.shape[1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            cross_edge_cutoff_angstrom=CROSS_EDGE_CUTOFF_ANGSTROM,
            num_rbf_centers=NUM_RBF_CENTERS,
        )

    def soft_model_factory():
        example = samples[0]
        return SoftCrossInteractionRigidBodyVectorField(
            ligand_feature_dim=example.ligand.x.shape[1],
            pocket_feature_dim=example.pocket.x.shape[1],
            bond_feature_dim=example.ligand.edge_attr.shape[1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            length_scale_angstrom=CROSS_EDGE_CUTOFF_ANGSTROM,
            num_rbf_centers=NUM_RBF_CENTERS,
        )

    print("\n--- 1. ORIGINAL hard-cutoff cross-interaction model (Milestone 3, unmodified) ---")
    start = time.time()
    hard_result = run_smoke_experiment(samples, model_factory=hard_model_factory, **common_kwargs)
    hard_elapsed = time.time() - start
    hard_summary = summarize("hard cutoff (Milestone 3)", hard_result, hard_elapsed)
    print_summary(hard_summary)

    print("\n--- 2. SOFT distance-weighted cross-interaction model (this ablation) ---")
    start = time.time()
    soft_result = run_smoke_experiment(samples, model_factory=soft_model_factory, **common_kwargs)
    soft_elapsed = time.time() - start
    soft_summary = summarize("soft distance-weighted (ablation)", soft_result, soft_elapsed)
    print_summary(soft_summary)

    print("\n=== COMPARISON ===")
    print(f"{'metric':28s} {'hard cutoff':>15s} {'soft weighted':>15s}")
    rows = [
        ("trainable parameters", f"{hard_summary['trainable_parameters']:,}", f"{soft_summary['trainable_parameters']:,}"),
        ("training time (s)", f"{hard_summary['training_time_seconds']:.1f}", f"{soft_summary['training_time_seconds']:.1f}"),
        ("final training loss", f"{hard_summary['final_training_loss']:.4f}", f"{soft_summary['final_training_loss']:.4f}"),
        ("mean final RMSD (A)", f"{hard_summary['mean_final_rmsd']:.2f}", f"{soft_summary['mean_final_rmsd']:.2f}"),
        ("median final RMSD (A)", f"{hard_summary['median_final_rmsd']:.2f}", f"{soft_summary['median_final_rmsd']:.2f}"),
        ("success rate @ 2A", f"{hard_summary['success_rate_at_2A']:.0%}", f"{soft_summary['success_rate_at_2A']:.0%}"),
    ]
    for label, hard_val, soft_val in rows:
        print(f"{label:28s} {hard_val:>15s} {soft_val:>15s}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = {
        "device": device_name,
        "hypothesis": (
            "The hard protein-ligand cross-edge cutoff creates gradient dead zones "
            "when the randomized ligand pose starts far from the pocket."
        ),
        "shared_hyperparameters": common_kwargs
        | {
            "pocket_radius_angstrom": POCKET_RADIUS_ANGSTROM,
            "edge_cutoff_angstrom": EDGE_CUTOFF_ANGSTROM,
            "cross_edge_cutoff_or_length_scale_angstrom": CROSS_EDGE_CUTOFF_ANGSTROM,
            "num_rbf_centers": NUM_RBF_CENTERS,
        },
        "hard_model": hard_summary,
        "soft_model": soft_summary,
    }
    with open(OUTPUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nsaved ablation summary to {OUTPUT_DIR / 'ablation_summary.json'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(hard_summary["loss_history"], label="hard cutoff")
        axes[0].plot(soft_summary["loss_history"], label="soft weighted")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("rigid flow matching loss")
        axes[0].set_yscale("log")
        axes[0].set_title("training loss")
        axes[0].legend()

        x = range(len(SAMPLE_COMPLEX_IDS))
        width = 0.35
        hard_final = [hard_summary["per_complex_final_rmsd"][c] for c in SAMPLE_COMPLEX_IDS]
        soft_final = [soft_summary["per_complex_final_rmsd"][c] for c in SAMPLE_COMPLEX_IDS]
        axes[1].bar([i - width / 2 for i in x], hard_final, width, label="hard cutoff")
        axes[1].bar([i + width / 2 for i in x], soft_final, width, label="soft weighted")
        axes[1].axhline(SUCCESS_THRESHOLD_ANGSTROM, color="gray", linestyle="--", linewidth=1, label="2 A success")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(SAMPLE_COMPLEX_IDS, rotation=45, ha="right")
        axes[1].set_ylabel("final RMSD (A)")
        axes[1].set_title("per-complex final RMSD")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "ablation_plots.png", dpi=150)
        print(f"saved ablation plots to {OUTPUT_DIR / 'ablation_plots.png'}")
    except ImportError:
        print("matplotlib not available; skipped plots")


if __name__ == "__main__":
    main()
