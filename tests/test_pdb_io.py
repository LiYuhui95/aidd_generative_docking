"""Tests for low-level PDB parsing: ligand block extraction and residue parsing."""

from __future__ import annotations

import pytest

from aidd_generative_docking.data.constants import STANDARD_AMINO_ACIDS
from aidd_generative_docking.data.pdb_io import extract_ligand_pdb_block, parse_protein_residues


def test_extract_ligand_pdb_block_has_matching_atoms_and_bonds(pdbbind_subset) -> None:
    path = pdbbind_subset.ensure_downloaded("3PTB")
    text = path.read_text()
    block = extract_ligand_pdb_block(text, "BEN")
    het_lines = [line for line in block.splitlines() if line.startswith("HETATM")]
    conect_lines = [line for line in block.splitlines() if line.startswith("CONECT")]
    assert len(het_lines) == 9  # benzamidine: 9 heavy atoms
    assert len(conect_lines) > 0


def test_extract_ligand_pdb_block_missing_resname_raises(pdbbind_subset) -> None:
    path = pdbbind_subset.ensure_downloaded("3PTB")
    text = path.read_text()
    with pytest.raises(ValueError):
        extract_ligand_pdb_block(text, "ZZZ")


def test_parse_protein_residues_only_standard_amino_acids(pdbbind_subset) -> None:
    path = pdbbind_subset.ensure_downloaded("3PTB")
    residues = parse_protein_residues(path)
    assert len(residues) > 100  # beta-trypsin has ~223 residues
    resnames = {r.resname for r in residues}
    assert resnames.issubset(set(STANDARD_AMINO_ACIDS))


def test_parse_protein_residues_have_finite_ca_coords(pdbbind_subset) -> None:
    path = pdbbind_subset.ensure_downloaded("3PTB")
    residues = parse_protein_residues(path)
    for residue in residues:
        assert len(residue.ca_coord) == 3
        assert all(coord == coord for coord in residue.ca_coord)  # not NaN
