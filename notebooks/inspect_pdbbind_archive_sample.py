"""End-to-end smoke test on 10 real PDBBind v2020.R1 complexes.

Requires the user's own licensed PDBBind download already in place:
``data/raw/P-L.tar.gz`` (protein-ligand structures) and
``data/raw/index.tar.gz`` (affinity index). Neither archive is modified;
only the 10 complexes below are extracted, into a gitignored cache under
``data/raw/.cache/pdbbind_archive/``.

Run with the ``sxt-torch`` environment, from the repo root:

    conda run -n sxt-torch python notebooks/inspect_pdbbind_archive_sample.py
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import torch

from aidd_generative_docking.data.pdb_io import extract_pdbbind_complexes, read_pdbbind_general_index
from aidd_generative_docking.data.sample import build_training_sample_from_files

RAW_DIR = Path("data/raw")
ARCHIVE_PATH = RAW_DIR / "P-L.tar.gz"
INDEX_ARCHIVE_PATH = RAW_DIR / "index.tar.gz"
CACHE_DIR = RAW_DIR / ".cache" / "pdbbind_archive"
INDEX_CACHE_DIR = RAW_DIR / ".cache" / "index"

# 10 hand-picked complexes covering: tiny/large ligands, aromatic rings,
# charged ligands (carboxylate, phosphate, zwitterion), a covalent-flagged
# entry, an inequality-affinity entry, and all 3 year buckets.
SAMPLE_COMPLEX_IDS = ["1stp", "1azm", "6cha", "4cts", "6hrp", "2tpi", "1oit", "1b39", "6q36", "6uyy"]

POCKET_RADIUS_ANGSTROM = 10.0
EDGE_CUTOFF_ANGSTROM = 8.0


def _load_index() -> dict[str, str]:
    if not any(INDEX_CACHE_DIR.glob("INDEX_general_PL*.lst")):
        with tarfile.open(INDEX_ARCHIVE_PATH, "r:gz") as archive:
            archive.extractall(INDEX_CACHE_DIR, filter="data")  # index archive is tiny (~0.5 MB)
    index_path = next((INDEX_CACHE_DIR / "index").glob("INDEX_general_PL*.lst"))
    entries = read_pdbbind_general_index(index_path)
    return {e.complex_id: f"{e.affinity_measure}{e.affinity_operator}{e.affinity_value}{e.affinity_unit}" for e in entries}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print()

    affinities = _load_index()
    paths_by_id = extract_pdbbind_complexes(ARCHIVE_PATH, SAMPLE_COMPLEX_IDS, CACHE_DIR)

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

        n_aromatic = int(sample.ligand.x[:, -2].sum().item())  # aromatic flag column, see featurize_ligand
        n_charged = int((sample.ligand.x[:, -3] != 0).sum().item())  # formal charge column
        ligand_centroid = sample.ligand.pos.mean(dim=0)
        min_ca_dist = torch.cdist(ligand_centroid[None], sample.pocket.pos).min().item()

        print(f"=== {complex_id} (affinity: {affinities.get(complex_id, 'unknown')}) ===")
        print(f"  ligand : {tuple(sample.ligand.x.shape)} atoms, {n_aromatic} aromatic, {n_charged} charged")
        print(f"  bonds  : edge_attr {tuple(sample.ligand.edge_attr.shape)}")
        print(f"  pocket : {tuple(sample.pocket.x.shape)} residues, edges {tuple(sample.pocket.edge_index.shape)}")
        print(f"  min ligand-centroid to nearest pocket CA: {min_ca_dist:.1f} A (bound-pose sanity check)")
        print()


if __name__ == "__main__":
    main()
