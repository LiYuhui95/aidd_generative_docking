"""Tests for ligand-graph and protein-pocket-graph featurization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rdkit import Chem

from aidd_generative_docking.data.featurize import featurize_ligand, featurize_protein_pocket, mol_from_ligand_block
from aidd_generative_docking.data.pdb_io import extract_ligand_pdb_block, parse_protein_residues


def _benzamidine_ligand_graph(pdbbind_subset):
    path = pdbbind_subset.ensure_downloaded("3PTB")
    text = path.read_text()
    block = extract_ligand_pdb_block(text, "BEN")
    mol = mol_from_ligand_block(block)
    return featurize_ligand(mol)


def test_ligand_graph_shapes(pdbbind_subset) -> None:
    graph = _benzamidine_ligand_graph(pdbbind_subset)
    num_atoms = 9
    assert graph.x.shape[0] == num_atoms
    assert graph.pos.shape == (num_atoms, 3)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == graph.edge_attr.shape[0]
    assert graph.edge_index.max().item() < num_atoms
    assert graph.edge_index.min().item() >= 0


def test_ligand_graph_has_no_nan_or_empty_content(pdbbind_subset) -> None:
    graph = _benzamidine_ligand_graph(pdbbind_subset)
    assert not torch.isnan(graph.x).any()
    assert not torch.isnan(graph.pos).any()
    assert graph.x.shape[0] > 0
    assert graph.edge_index.shape[1] > 0


def test_featurize_ligand_empty_mol_raises() -> None:
    empty_mol = Chem.RWMol()
    with pytest.raises(ValueError):
        featurize_ligand(empty_mol)


def test_pocket_graph_residues_within_radius(pdbbind_subset) -> None:
    path = pdbbind_subset.ensure_downloaded("3PTB")
    ligand_graph = _benzamidine_ligand_graph(pdbbind_subset)
    residues = parse_protein_residues(path)
    pocket = featurize_protein_pocket(
        residues, ligand_graph.pos.numpy(), pocket_radius_angstrom=10.0, edge_cutoff_angstrom=8.0
    )

    assert 0 < pocket.x.shape[0] < len(residues)  # a strict, non-trivial subset
    assert pocket.pos.shape == (pocket.x.shape[0], 3)
    assert pocket.edge_index.shape[0] == 2
    assert pocket.edge_index.shape[1] > 0

    dists = np.linalg.norm(
        pocket.pos.numpy()[:, None, :] - ligand_graph.pos.numpy()[None, :, :], axis=-1
    ).min(axis=1)
    assert (dists <= 10.0).all()


def test_pocket_graph_empty_ligand_positions_raises(pdbbind_subset) -> None:
    path = pdbbind_subset.ensure_downloaded("3PTB")
    residues = parse_protein_residues(path)
    far_away_ligand = np.array([[1e6, 1e6, 1e6]])
    with pytest.raises(ValueError):
        featurize_protein_pocket(residues, far_away_ligand, pocket_radius_angstrom=10.0, edge_cutoff_angstrom=8.0)


def test_pocket_graph_no_residues_raises() -> None:
    with pytest.raises(ValueError):
        featurize_protein_pocket([], np.zeros((1, 3)), pocket_radius_angstrom=10.0, edge_cutoff_angstrom=8.0)
