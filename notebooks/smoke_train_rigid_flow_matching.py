"""Milestone 2 smoke experiment: rigid-body conditional Flow Matching.

Pipeline sanity check ONLY -- 10 complexes trained for many epochs is
expected to memorize, not generalize; this is not a scientific benchmark
(see README.md / ARCHITECTURE.md). It exists to prove that data loading,
random rigid-pose construction, the SE(3)-aware model, the Flow Matching
loss, and Euler-integration sampling are all wired together correctly, and
that the learned field measurably reduces ligand pose RMSD relative to a
random rigid start.

Requires the licensed PDBBind archive already present at
data/raw/P-L.tar.gz (see previous milestone's data-audit report). Saves a
JSON summary and a loss-curve plot under outputs/milestone2/.

Run with the sxt-torch environment, from the repo root:

    conda run -n sxt-torch python notebooks/smoke_train_rigid_flow_matching.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from aidd_generative_docking.data.pdb_io import extract_pdbbind_complexes
from aidd_generative_docking.data.sample import build_training_sample_from_files
from aidd_generative_docking.training.smoke_train_rigid import run_smoke_experiment

RAW_DIR = Path("data/raw")
ARCHIVE_PATH = RAW_DIR / "P-L.tar.gz"
CACHE_DIR = RAW_DIR / ".cache" / "pdbbind_archive"
OUTPUT_DIR = Path("outputs/milestone2")

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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"device: {device} ({device_name})")

    paths_by_id = extract_pdbbind_complexes(ARCHIVE_PATH, SAMPLE_COMPLEX_IDS, CACHE_DIR)
    samples = []
    for complex_id in SAMPLE_COMPLEX_IDS:
        paths = paths_by_id[complex_id]
        sample = build_training_sample_from_files(
            protein_pdb_path=paths["protein"],
            ligand_sdf_path=paths["ligand_sdf"],
            ligand_mol2_path=paths["ligand_mol2"],
            complex_id=complex_id,
            pocket_radius_angstrom=POCKET_RADIUS_ANGSTROM,
            edge_cutoff_angstrom=EDGE_CUTOFF_ANGSTROM,
        )
        samples.append(sample)
        print(f"  loaded {complex_id}: {sample.ligand.x.shape[0]} ligand atoms, {sample.pocket.x.shape[0]} pocket residues")

    start = time.time()
    result = run_smoke_experiment(
        samples,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        translation_noise_scale=TRANSLATION_NOISE_SCALE,
        num_integration_steps=NUM_INTEGRATION_STEPS,
        seed=SEED,
        device=device,
    )
    elapsed = time.time() - start

    model = result["model"]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    initial_rmsd = result["initial_rmsd"]
    final_rmsd = result["final_rmsd"]
    mean_initial = sum(initial_rmsd.values()) / len(initial_rmsd)
    mean_final = sum(final_rmsd.values()) / len(final_rmsd)

    print()
    print(f"trainable parameters: {n_params:,}")
    print(f"training time: {elapsed:.1f}s for {NUM_EPOCHS} epochs over {len(samples)} complexes")
    print(f"final training loss: {result['loss_history'][-1]:.4f} (initial: {result['loss_history'][0]:.4f})")
    print()
    print(f"{'complex':8s} {'initial RMSD':>14s} {'final RMSD':>12s}")
    for complex_id in SAMPLE_COMPLEX_IDS:
        print(f"{complex_id:8s} {initial_rmsd[complex_id]:14.2f} {final_rmsd[complex_id]:12.2f}")
    print(f"{'MEAN':8s} {mean_initial:14.2f} {mean_final:12.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": device_name,
        "num_epochs": NUM_EPOCHS,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "translation_noise_scale": TRANSLATION_NOISE_SCALE,
        "num_integration_steps": NUM_INTEGRATION_STEPS,
        "seed": SEED,
        "trainable_parameters": n_params,
        "training_time_seconds": elapsed,
        "loss_history": result["loss_history"],
        "initial_rmsd": initial_rmsd,
        "final_rmsd": final_rmsd,
        "mean_initial_rmsd": mean_initial,
        "mean_final_rmsd": mean_final,
    }
    with open(OUTPUT_DIR / "smoke_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary to {OUTPUT_DIR / 'smoke_experiment_summary.json'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(result["loss_history"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("rigid flow matching loss")
        ax.set_title("Milestone 2 smoke experiment: training loss")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
        print(f"saved loss curve to {OUTPUT_DIR / 'loss_curve.png'}")
    except ImportError:
        print("matplotlib not available; skipped loss curve plot")


if __name__ == "__main__":
    main()
