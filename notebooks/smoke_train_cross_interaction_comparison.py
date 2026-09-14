"""Milestone 3 smoke experiment: compare the Milestone 2 rigid-body model
against the new dynamic cross-interaction model, under an identical
training/evaluation protocol on the same 10 real complexes.

Pipeline sanity check ONLY (see README.md / ARCHITECTURE.md) -- not a
benchmark, and hyperparameters are intentionally NOT tuned to make the new
model look better: both models use the same hidden_dim, num_layers,
num_epochs, seed, translation_noise_scale, and num_integration_steps. The
new model's cross-interaction-specific hyperparameters
(cross_edge_cutoff_angstrom, num_rbf_centers) are set to standard,
undtuned defaults.

Requires the licensed PDBBind archive at data/raw/P-L.tar.gz (see
milestone 1's data-audit report). Saves a JSON comparison summary and
plots under outputs/milestone3/.

Run with the sxt-torch environment, from the repo root:

    conda run -n sxt-torch python notebooks/smoke_train_cross_interaction_comparison.py
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
from aidd_generative_docking.models.rigid_equivariant import RigidBodyVectorField
from aidd_generative_docking.training.smoke_train_rigid import run_smoke_experiment

RAW_DIR = Path("data/raw")
ARCHIVE_PATH = RAW_DIR / "P-L.tar.gz"
CACHE_DIR = RAW_DIR / ".cache" / "pdbbind_archive"
OUTPUT_DIR = Path("outputs/milestone3")

# Identical to milestone 2's smoke experiment for a controlled comparison.
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

# New-model-only hyperparameters: standard defaults, not tuned.
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


def summarize(name: str, result: dict, elapsed: float) -> dict:
    n_params = sum(p.numel() for p in result["model"].parameters() if p.requires_grad)
    initial = result["initial_rmsd"]
    final = result["final_rmsd"]
    initial_values = list(initial.values())
    final_values = list(final.values())

    def median(values):
        s = sorted(values)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0

    return {
        "name": name,
        "trainable_parameters": n_params,
        "training_time_seconds": elapsed,
        "final_training_loss": result["loss_history"][-1],
        "initial_training_loss": result["loss_history"][0],
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
    print(f"mean initial RMSD    : {summary['mean_initial_rmsd']:.2f} A")
    print(f"mean final RMSD      : {summary['mean_final_rmsd']:.2f} A")
    print(f"median initial RMSD  : {summary['median_initial_rmsd']:.2f} A")
    print(f"median final RMSD    : {summary['median_final_rmsd']:.2f} A")
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

    print("\n--- training OLD model (RigidBodyVectorField, dense pocket conditioning) ---")
    start = time.time()
    old_result = run_smoke_experiment(samples, **common_kwargs)  # model_factory=None -> identical to milestone 2
    old_elapsed = time.time() - start
    old_summary = summarize("old (dense, Milestone 2)", old_result, old_elapsed)
    print_summary(old_summary)

    print("\n--- training NEW model (CrossInteractionRigidBodyVectorField, dynamic RBF cross edges) ---")

    def new_model_factory():
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

    start = time.time()
    new_result = run_smoke_experiment(samples, model_factory=new_model_factory, **common_kwargs)
    new_elapsed = time.time() - start
    new_summary = summarize("new (dynamic cross-interaction, Milestone 3)", new_result, new_elapsed)
    print_summary(new_summary)

    print("\n=== COMPARISON ===")
    print(f"{'metric':28s} {'old (dense)':>15s} {'new (cross-int)':>18s}")
    rows = [
        ("trainable parameters", f"{old_summary['trainable_parameters']:,}", f"{new_summary['trainable_parameters']:,}"),
        ("training time (s)", f"{old_summary['training_time_seconds']:.1f}", f"{new_summary['training_time_seconds']:.1f}"),
        ("final training loss", f"{old_summary['final_training_loss']:.4f}", f"{new_summary['final_training_loss']:.4f}"),
        ("mean initial RMSD (A)", f"{old_summary['mean_initial_rmsd']:.2f}", f"{new_summary['mean_initial_rmsd']:.2f}"),
        ("mean final RMSD (A)", f"{old_summary['mean_final_rmsd']:.2f}", f"{new_summary['mean_final_rmsd']:.2f}"),
        ("median initial RMSD (A)", f"{old_summary['median_initial_rmsd']:.2f}", f"{new_summary['median_initial_rmsd']:.2f}"),
        ("median final RMSD (A)", f"{old_summary['median_final_rmsd']:.2f}", f"{new_summary['median_final_rmsd']:.2f}"),
        ("success rate @ 2A", f"{old_summary['success_rate_at_2A']:.0%}", f"{new_summary['success_rate_at_2A']:.0%}"),
    ]
    for label, old_val, new_val in rows:
        print(f"{label:28s} {old_val:>15s} {new_val:>18s}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = {
        "device": device_name,
        "shared_hyperparameters": common_kwargs | {"pocket_radius_angstrom": POCKET_RADIUS_ANGSTROM, "edge_cutoff_angstrom": EDGE_CUTOFF_ANGSTROM},
        "new_model_only_hyperparameters": {
            "cross_edge_cutoff_angstrom": CROSS_EDGE_CUTOFF_ANGSTROM,
            "num_rbf_centers": NUM_RBF_CENTERS,
        },
        "old_model": old_summary,
        "new_model": new_summary,
    }
    with open(OUTPUT_DIR / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nsaved comparison summary to {OUTPUT_DIR / 'comparison_summary.json'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(old_summary["loss_history"], label="old (dense)")
        axes[0].plot(new_summary["loss_history"], label="new (cross-interaction)")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("rigid flow matching loss")
        axes[0].set_yscale("log")
        axes[0].set_title("training loss")
        axes[0].legend()

        x = range(len(SAMPLE_COMPLEX_IDS))
        width = 0.35
        old_final = [old_summary["per_complex_final_rmsd"][c] for c in SAMPLE_COMPLEX_IDS]
        new_final = [new_summary["per_complex_final_rmsd"][c] for c in SAMPLE_COMPLEX_IDS]
        axes[1].bar([i - width / 2 for i in x], old_final, width, label="old (dense)")
        axes[1].bar([i + width / 2 for i in x], new_final, width, label="new (cross-interaction)")
        axes[1].axhline(SUCCESS_THRESHOLD_ANGSTROM, color="gray", linestyle="--", linewidth=1, label="2 A success")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(SAMPLE_COMPLEX_IDS, rotation=45, ha="right")
        axes[1].set_ylabel("final RMSD (A)")
        axes[1].set_title("per-complex final RMSD")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "comparison_plots.png", dpi=150)
        print(f"saved comparison plots to {OUTPUT_DIR / 'comparison_plots.png'}")
    except ImportError:
        print("matplotlib not available; skipped plots")


if __name__ == "__main__":
    main()
