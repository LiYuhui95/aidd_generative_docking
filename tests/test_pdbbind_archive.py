"""Tests for the real PDBBind v2020.R1 archive pipeline.

These tests require the user's own licensed PDBBind download at
``data/raw/P-L.tar.gz`` / ``data/raw/index.tar.gz`` and are skipped
entirely when that data isn't present locally (e.g. a fresh clone,
or CI without access to the licensed archive).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aidd_generative_docking.data.featurize import load_ligand_mol
from aidd_generative_docking.data.pdb_io import extract_pdbbind_complexes
from aidd_generative_docking.data.sample import build_training_sample_from_files

RAW_DIR = Path("data/raw")
ARCHIVE_PATH = RAW_DIR / "P-L.tar.gz"

# Same 10 complexes used in notebooks/inspect_pdbbind_archive_sample.py.
SAMPLE_COMPLEX_IDS = ["1stp", "1azm", "6cha", "4cts", "6hrp", "2tpi", "1oit", "1b39", "6q36", "6uyy"]

pytestmark = pytest.mark.skipif(not ARCHIVE_PATH.exists(), reason="PDBBind archive not present under data/raw/")


@pytest.fixture(scope="session")
def pdbbind_archive_paths(tmp_path_factory) -> dict[str, dict[str, Path]]:
    cache_dir = tmp_path_factory.mktemp("pdbbind_archive_cache")
    return extract_pdbbind_complexes(ARCHIVE_PATH, SAMPLE_COMPLEX_IDS, cache_dir)


def test_extract_pdbbind_complexes_returns_all_four_files(pdbbind_archive_paths) -> None:
    for complex_id in SAMPLE_COMPLEX_IDS:
        paths = pdbbind_archive_paths[complex_id]
        assert set(paths.keys()) == {"protein", "pocket", "ligand_sdf", "ligand_mol2"}
        for path in paths.values():
            assert path.exists()
            assert path.stat().st_size > 0


def test_extract_pdbbind_complexes_unknown_id_raises(tmp_path) -> None:
    with pytest.raises(KeyError):
        extract_pdbbind_complexes(ARCHIVE_PATH, ["zzzz"], tmp_path)


def test_ligand_bond_order_and_aromaticity_are_real(pdbbind_archive_paths) -> None:
    # 1azm (acetazolamide) has a known aromatic thiadiazole ring: CONECT-only
    # parsing (the legacy RCSB path) cannot perceive this -- see featurize.py.
    paths = pdbbind_archive_paths["1azm"]
    mol = load_ligand_mol(paths["ligand_sdf"], paths["ligand_mol2"])
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    bond_types = {str(bond.GetBondType()) for bond in mol.GetBonds()}
    assert aromatic_atoms > 0
    assert "AROMATIC" in bond_types
    assert "DOUBLE" in bond_types  # not every bond collapsed to SINGLE


def test_ligand_formal_charge_is_real(pdbbind_archive_paths) -> None:
    # 1stp (biotin) is deposited with a deprotonated carboxylate: net charge -1.
    paths = pdbbind_archive_paths["1stp"]
    mol = load_ligand_mol(paths["ligand_sdf"], paths["ligand_mol2"])
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    assert total_charge == -1


def test_ligand_mol2_fallback_used_when_sdf_is_corrupt(tmp_path, pdbbind_archive_paths) -> None:
    paths = pdbbind_archive_paths["1stp"]
    corrupt_sdf = tmp_path / "corrupt_ligand.sdf"
    corrupt_sdf.write_text("this is not a valid molfile\n")

    mol = load_ligand_mol(corrupt_sdf, paths["ligand_mol2"])
    assert mol.GetNumAtoms() > 0


def test_ligand_mol2_fallback_raises_when_both_invalid(tmp_path) -> None:
    corrupt_sdf = tmp_path / "corrupt_ligand.sdf"
    corrupt_mol2 = tmp_path / "corrupt_ligand.mol2"
    corrupt_sdf.write_text("not a molfile\n")
    corrupt_mol2.write_text("not a mol2 file\n")

    with pytest.raises(ValueError):
        load_ligand_mol(corrupt_sdf, corrupt_mol2)


def test_bound_pose_coordinates_share_reference_frame_with_protein(pdbbind_archive_paths) -> None:
    # If ligand coordinates were re-centered/idealized rather than the true
    # bound pose, the ligand centroid would not sit close to any pocket atom.
    for complex_id in SAMPLE_COMPLEX_IDS:
        paths = pdbbind_archive_paths[complex_id]
        sample = build_training_sample_from_files(
            protein_pdb_path=paths["protein"],
            ligand_sdf_path=paths["ligand_sdf"],
            ligand_mol2_path=paths["ligand_mol2"],
            complex_id=complex_id,
            pocket_radius_angstrom=10.0,
            edge_cutoff_angstrom=8.0,
        )
        ligand_centroid = sample.ligand.pos.mean(dim=0, keepdim=True)
        min_dist = torch.cdist(ligand_centroid, sample.pocket.pos).min().item()
        assert min_dist < 15.0  # generous bound; true poses are typically ~4-10 A


def test_build_training_sample_from_files_shapes(pdbbind_archive_paths) -> None:
    paths = pdbbind_archive_paths["1azm"]
    sample = build_training_sample_from_files(
        protein_pdb_path=paths["protein"],
        ligand_sdf_path=paths["ligand_sdf"],
        ligand_mol2_path=paths["ligand_mol2"],
        complex_id="1azm",
        pocket_radius_angstrom=10.0,
        edge_cutoff_angstrom=8.0,
    )
    assert sample.complex_id == "1azm"
    assert sample.ligand.x.shape[0] == sample.ligand.pos.shape[0] > 0
    assert sample.ligand.edge_index.shape[1] == sample.ligand.edge_attr.shape[0] > 0
    assert sample.pocket.x.shape[0] == sample.pocket.pos.shape[0] > 0
    assert not torch.isnan(sample.ligand.pos).any()
    assert not torch.isnan(sample.pocket.pos).any()
