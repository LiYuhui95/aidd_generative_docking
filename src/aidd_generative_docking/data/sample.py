"""Assembles one (protein pocket, ligand) training sample from a raw complex file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch_geometric.data import Data

from aidd_generative_docking.data.featurize import (
    featurize_ligand,
    featurize_protein_pocket,
    load_ligand_mol,
    mol_from_ligand_block,
)
from aidd_generative_docking.data.pdb_io import extract_ligand_pdb_block, parse_protein_residues


@dataclass
class ComplexSample:
    """One protein-pocket / ligand training example.

    Attributes:
        complex_id: identifier of the source complex (e.g. a PDB code).
        ligand_resname: 3-character residue name of the ligand used.
        ligand: PyTorch Geometric graph with ``x``, ``pos``, ``edge_index``,
            ``edge_attr`` (see ``featurize.featurize_ligand``).
        pocket: PyTorch Geometric graph with ``x``, ``pos``, ``edge_index``
            (see ``featurize.featurize_protein_pocket``).
    """

    complex_id: str
    ligand_resname: str
    ligand: Data
    pocket: Data


def build_training_sample(
    pdb_path: Path | str,
    complex_id: str,
    ligand_resname: str,
    pocket_radius_angstrom: float,
    edge_cutoff_angstrom: float,
) -> ComplexSample:
    """Build one ``ComplexSample`` from a raw protein-ligand complex PDB file.

    Args:
        pdb_path: path to a structure file containing both the protein and
            the bound ligand (as deposited by RCSB).
        complex_id: identifier to attach to the resulting sample.
        ligand_resname: 3-character HETATM residue name of the ligand.
        pocket_radius_angstrom: see ``featurize.featurize_protein_pocket``.
        edge_cutoff_angstrom: see ``featurize.featurize_protein_pocket``.

    Returns:
        A populated ``ComplexSample``.
    """
    pdb_path = Path(pdb_path)
    pdb_text = pdb_path.read_text()

    ligand_block = extract_ligand_pdb_block(pdb_text, ligand_resname)
    ligand_mol = mol_from_ligand_block(ligand_block)
    ligand_graph = featurize_ligand(ligand_mol)

    residues = parse_protein_residues(pdb_path)
    pocket_graph = featurize_protein_pocket(
        residues,
        ligand_positions=ligand_graph.pos.numpy(),
        pocket_radius_angstrom=pocket_radius_angstrom,
        edge_cutoff_angstrom=edge_cutoff_angstrom,
    )

    return ComplexSample(
        complex_id=complex_id,
        ligand_resname=ligand_resname,
        ligand=ligand_graph,
        pocket=pocket_graph,
    )


def build_training_sample_from_files(
    protein_pdb_path: Path | str,
    ligand_sdf_path: Path | str,
    complex_id: str,
    pocket_radius_angstrom: float,
    edge_cutoff_angstrom: float,
    ligand_mol2_path: Path | str | None = None,
    ligand_resname: str = "UNK",
) -> ComplexSample:
    """Build one ``ComplexSample`` from PDBBind's real, separate per-complex files.

    Unlike ``build_training_sample`` (legacy RCSB path), the ligand is read
    from its own SDF/MOL2 file rather than guessed from PDB CONECT records,
    so bond order/aromaticity/formal charge are correct (see
    ``featurize.load_ligand_mol``). The protein file
    (``<id>_protein.pdb``) contains only the protein (+ water), no ligand
    HETATM records, so ``parse_protein_residues`` and
    ``featurize_protein_pocket`` are reused unchanged.

    Args:
        protein_pdb_path: path to PDBBind's ``<id>_protein.pdb``.
        ligand_sdf_path: path to PDBBind's ``<id>_ligand.sdf``.
        complex_id: identifier to attach to the resulting sample.
        pocket_radius_angstrom: see ``featurize.featurize_protein_pocket``.
        edge_cutoff_angstrom: see ``featurize.featurize_protein_pocket``.
        ligand_mol2_path: path to PDBBind's ``<id>_ligand.mol2``, used only
            as a fallback if the SDF fails to parse.
        ligand_resname: optional display label for the ligand (PDBBind's
            per-complex files are already isolated, so this is metadata
            only, not used to locate or filter atoms).

    Returns:
        A populated ``ComplexSample``.
    """
    ligand_mol = load_ligand_mol(ligand_sdf_path, ligand_mol2_path)
    ligand_graph = featurize_ligand(ligand_mol)

    residues = parse_protein_residues(protein_pdb_path)
    pocket_graph = featurize_protein_pocket(
        residues,
        ligand_positions=ligand_graph.pos.numpy(),
        pocket_radius_angstrom=pocket_radius_angstrom,
        edge_cutoff_angstrom=edge_cutoff_angstrom,
    )

    return ComplexSample(
        complex_id=complex_id,
        ligand_resname=ligand_resname,
        ligand=ligand_graph,
        pocket=pocket_graph,
    )
