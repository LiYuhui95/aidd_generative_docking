"""Held-out experiment, stage 1: build, validate, and split the dataset.

Builds a candidate pool from the full PDBBind v2020.R1 archive, validates
every candidate (documenting exclusion reasons), caps the dataset to a
manageable size, computes the deterministic protein/ligand-similarity-aware
train/val/test split, reports ligand-size and pocket-size distributions,
and pickles the result for the training stage
(``run_held_out_experiment.py``) to consume without repeating this
(several-minute) step.

Run with the sxt-torch environment, from the repo root:

    conda run -n sxt-torch python notebooks/prepare_held_out_dataset.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from aidd_generative_docking.data.dataset_builder import (
    build_validated_dataset,
    select_candidate_ids,
    summarize_exclusions_by_family,
)
from aidd_generative_docking.data.pdb_io import read_pdbbind_general_index
from aidd_generative_docking.data.splitting import assign_similarity_aware_split

RAW_DIR = Path("data/raw")
ARCHIVE_PATH = RAW_DIR / "P-L.tar.gz"
INDEX_PATH = RAW_DIR / ".cache" / "index" / "index" / "INDEX_general_PL.2020R1.lst"
CACHE_DIR = RAW_DIR / ".cache" / "pdbbind_archive"
OUTPUT_DIR = Path("outputs/held_out")
DATASET_PICKLE_PATH = RAW_DIR / ".cache" / "held_out_dataset.pkl"

CANDIDATE_POOL_SIZE = 1200
FINAL_DATASET_SIZE = 800
SEED = 0
MAX_LIGAND_ATOMS = 100
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
LIGAND_SIMILARITY_THRESHOLD = 0.4


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(int(p / 100 * len(s)), len(s) - 1)
    return s[idx]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading PDBBind index...")
    index_entries = read_pdbbind_general_index(INDEX_PATH)
    print(f"  {len(index_entries)} total protein-ligand complexes in PDBBind v2020.R1")

    candidate_ids = select_candidate_ids(index_entries, pool_size=CANDIDATE_POOL_SIZE, seed=SEED, exclude_covalent=True)
    print(f"selected {len(candidate_ids)} candidates (seed={SEED}, covalent-flagged entries excluded upfront)")

    print("validating candidates (extract + featurize + exclude malformed)...")
    start = time.time()
    validated, report = build_validated_dataset(
        ARCHIVE_PATH, candidate_ids, CACHE_DIR, max_ligand_atoms=MAX_LIGAND_ATOMS
    )
    elapsed = time.time() - start
    print(f"  done in {elapsed:.1f}s: {report.num_valid}/{report.num_candidates} valid")

    families = summarize_exclusions_by_family(report)
    print("exclusion reasons:")
    for reason, count in sorted(families.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")

    if len(validated) > FINAL_DATASET_SIZE:
        validated = validated[:FINAL_DATASET_SIZE]  # deterministic: candidate order was already seeded+sorted
    print(f"final dataset size: {len(validated)} complexes")

    complex_ids = [v.complex_id for v in validated]
    protein_sequences = [v.protein_sequence for v in validated]
    ligand_fingerprints = [v.ligand_fingerprint for v in validated]

    print("computing deterministic protein/ligand-similarity-aware split...")
    split_result = assign_similarity_aware_split(
        complex_ids,
        protein_sequences,
        ligand_fingerprints,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        ligand_similarity_threshold=LIGAND_SIMILARITY_THRESHOLD,
    )
    split_counts = {"train": 0, "val": 0, "test": 0}
    for split in split_result.split_by_complex_id.values():
        split_counts[split] += 1
    print(f"  {split_result.num_groups} leakage groups found among {len(validated)} complexes")
    print(f"  split counts: {split_counts}")

    ligand_sizes = [v.sample.ligand.x.shape[0] for v in validated]
    pocket_sizes = [v.sample.pocket.x.shape[0] for v in validated]

    def dist_summary(values: list[float]) -> dict:
        return {
            "min": min(values),
            "p25": percentile(values, 25),
            "median": percentile(values, 50),
            "p75": percentile(values, 75),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    ligand_size_dist = dist_summary(ligand_sizes)
    pocket_size_dist = dist_summary(pocket_sizes)
    print(f"ligand size distribution (atoms, incl. H): {ligand_size_dist}")
    print(f"pocket size distribution (residues): {pocket_size_dist}")

    # Save the full validated-complex list + split assignment for the
    # training stage (avoids repeating the several-minute build step).
    with open(DATASET_PICKLE_PATH, "wb") as f:
        pickle.dump(
            {
                "validated": validated,
                "split_by_complex_id": split_result.split_by_complex_id,
                "group_by_complex_id": split_result.group_by_complex_id,
            },
            f,
        )
    print(f"saved dataset + split to {DATASET_PICKLE_PATH}")

    report_summary = {
        "pdbbind_total_complexes": len(index_entries),
        "candidate_pool_size": len(candidate_ids),
        "seed": SEED,
        "num_valid_before_cap": report.num_valid,
        "final_dataset_size": len(validated),
        "exclusion_families": families,
        "exclusion_examples": report.excluded_examples,
        "num_leakage_groups": split_result.num_groups,
        "split_counts": split_counts,
        "ligand_size_distribution": ligand_size_dist,
        "pocket_size_distribution": pocket_size_dist,
        "max_ligand_atoms_cap": MAX_LIGAND_ATOMS,
        "train_val_test_target_fractions": [TRAIN_FRAC, VAL_FRAC, TEST_FRAC],
        "ligand_similarity_threshold": LIGAND_SIMILARITY_THRESHOLD,
    }
    with open(OUTPUT_DIR / "dataset_report.json", "w") as f:
        json.dump(report_summary, f, indent=2)
    print(f"saved dataset report to {OUTPUT_DIR / 'dataset_report.json'}")


if __name__ == "__main__":
    main()
