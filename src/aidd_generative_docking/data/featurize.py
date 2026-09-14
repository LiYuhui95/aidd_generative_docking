"""Ligand and protein-pocket graph featurization (stage 2, real implementation).

Ligand: molecular graph built with RDKit (atoms as nodes, covalent bonds as
edges), using the actual bound-pose 3D coordinates from the crystal
structure. Two ways to obtain the RDKit ``Mol`` are provided:

- ``load_ligand_mol`` (preferred): reads PDBBind's own per-complex ligand
  SDF file (MOL2 fallback if the SDF fails to parse -- a known issue for a
  meaningful fraction of PDBBind ligand SDFs). Bond order, aromaticity, and
  formal charge all come from the file as deposited, verified against 10
  representative complexes (see ``notebooks/inspect_pdbbind_archive_sample.py``).
- ``mol_from_ligand_block`` (legacy): parses a CONECT-only PDB block from
  the original RCSB smoke-test path (``pdbbind.py``'s 3-complex manifest).
  CONECT records carry connectivity but not bond order, so every bond comes
  out as a formal single bond and aromaticity is not perceived -- kept only
  for backward compatibility with that older manifest and its tests.

Protein pocket: residue-level graph using Cα coordinates (see
``ARCHITECTURE.md`` and ``pdb_io.parse_protein_residues``), restricted to
residues near the ligand. Unchanged by the switch to real PDBBind data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from aidd_generative_docking.data.constants import BOND_TYPE_NAMES, LIGAND_ELEMENTS, STANDARD_AMINO_ACIDS
from aidd_generative_docking.data.pdb_io import ProteinResidue

_BOND_TYPES = [getattr(Chem.BondType, name) for name in BOND_TYPE_NAMES]


def mol_from_ligand_block(pdb_block: str) -> Chem.Mol:
    """Parse a ligand PDB block (HETATM + CONECT lines) into an RDKit Mol.

    Args:
        pdb_block: output of ``pdb_io.extract_ligand_pdb_block``.

    Returns:
        An RDKit ``Mol`` with one 3D conformer (the bound pose).

    Raises:
        ValueError: if RDKit cannot parse the block, or no conformer results.
    """
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=False)
    if mol is None:
        raise ValueError("RDKit failed to parse ligand PDB block")
    if mol.GetNumConformers() == 0:
        raise ValueError("Parsed ligand molecule has no 3D conformer")
    return mol


def load_ligand_mol(sdf_path: Path | str, mol2_path: Path | str | None = None) -> Chem.Mol:
    """Load a ligand from PDBBind's own SDF file, with a MOL2 fallback.

    PDBBind ships each ligand as both ``<id>_ligand.sdf`` and
    ``<id>_ligand.mol2``, both carrying real bond orders, aromaticity, and
    formal charge (unlike CONECT-only PDB parsing -- see module docstring).
    A published data-quality check found a meaningful fraction of PDBBind
    refined-set ligand SDFs fail to parse in RDKit, so MOL2 is used as a
    fallback rather than raising immediately.

    Args:
        sdf_path: path to ``<id>_ligand.sdf``.
        mol2_path: path to ``<id>_ligand.mol2``, used only if SDF parsing
            fails. If ``None``, no fallback is attempted.

    Returns:
        An RDKit ``Mol`` with one 3D conformer (the true bound pose).

    Raises:
        ValueError: if neither file could be parsed into a valid molecule.
    """
    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True, removeHs=False)
    if mol is not None and mol.GetNumConformers() > 0:
        return mol

    if mol2_path is not None:
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
        if mol is not None and mol.GetNumConformers() > 0:
            return mol

    raise ValueError(f"RDKit failed to parse ligand from '{sdf_path}'" + (f" or '{mol2_path}'" if mol2_path else ""))


def _one_hot_element(symbol: str) -> list[float]:
    vec = [0.0] * (len(LIGAND_ELEMENTS) + 1)
    if symbol in LIGAND_ELEMENTS:
        vec[LIGAND_ELEMENTS.index(symbol)] = 1.0
    else:
        vec[-1] = 1.0  # "other" bucket
    return vec


def featurize_ligand(mol: Chem.Mol) -> Data:
    """Build a ligand molecular graph: atoms as nodes, bonds as edges.

    Node features (per atom): one-hot element over ``LIGAND_ELEMENTS`` plus
    an "other" bucket, formal charge, aromaticity flag, heavy-atom degree.
    Edge features (per bond, both directions): one-hot bond type over
    ``BOND_TYPE_NAMES`` (see module docstring for the current single-bond
    limitation of this data source).

    Args:
        mol: an RDKit Mol with a 3D conformer, e.g. from
            ``mol_from_ligand_block``.

    Returns:
        A PyTorch Geometric ``Data`` with ``x``, ``pos``, ``edge_index``,
        ``edge_attr``.

    Raises:
        ValueError: if the molecule has no atoms or no bonds.
    """
    if mol.GetNumAtoms() == 0:
        raise ValueError("Ligand molecule has no atoms")

    conf = mol.GetConformer()
    node_features = []
    positions = []
    for atom in mol.GetAtoms():
        element_oh = _one_hot_element(atom.GetSymbol())
        node_features.append(
            [*element_oh, float(atom.GetFormalCharge()), float(atom.GetIsAromatic()), float(atom.GetDegree())]
        )
        p = conf.GetAtomPosition(atom.GetIdx())
        positions.append([p.x, p.y, p.z])

    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_oh = [1.0 if bond.GetBondType() == bt else 0.0 for bt in _BOND_TYPES]
        edge_index.append([i, j])
        edge_attr.append(bond_oh)
        edge_index.append([j, i])
        edge_attr.append(bond_oh)

    if not edge_index:
        raise ValueError("Ligand molecule has no bonds")

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        pos=torch.tensor(positions, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )


def _one_hot_residue(resname: str) -> list[float]:
    vec = [0.0] * len(STANDARD_AMINO_ACIDS)
    vec[STANDARD_AMINO_ACIDS.index(resname)] = 1.0
    return vec


def featurize_protein_pocket(
    residues: list[ProteinResidue],
    ligand_positions: np.ndarray,
    pocket_radius_angstrom: float,
    edge_cutoff_angstrom: float,
) -> Data:
    """Build a residue-level pocket graph near the ligand.

    Cutoff rationale:
        - ``pocket_radius_angstrom`` (default 10 A, see
          ``configs/baseline.yaml``): a residue is kept if its Cα atom is
          within this distance of *any* ligand (heavy) atom. 10 A from Cα
          reliably captures first- and second-shell contact residues for a
          small-molecule pocket -- side chains reach several Angstroms
          further than their Cα -- while keeping the pocket graph small.
          Using Cα (rather than an all-atom minimum distance) is a
          deliberate simplification consistent with the coarse,
          residue-level pocket representation described in
          ``ARCHITECTURE.md``; it can under-count residues whose side
          chain, but not backbone, reaches into the pocket, which the
          generous 10 A radius largely compensates for.
        - ``edge_cutoff_angstrom`` (default 8 A): two pocket residues are
          connected if their Cα atoms are within this distance, a standard
          local-neighborhood radius for residue-level graphs -- large
          enough to give every residue several geometric neighbors, small
          enough not to collapse the pocket into a near-complete graph.

    Args:
        residues: full list of protein residues (e.g. from
            ``pdb_io.parse_protein_residues``); pocket selection filters
            this down internally.
        ligand_positions: ``(num_ligand_atoms, 3)`` array of ligand atom
            coordinates, in the same reference frame as the residues.
        pocket_radius_angstrom: residue-selection cutoff, see above.
        edge_cutoff_angstrom: residue-residue edge cutoff, see above.

    Returns:
        A PyTorch Geometric ``Data`` with ``x`` (residue one-hot), ``pos``
        (Cα coordinates), ``edge_index`` (radius graph, both directions).

    Raises:
        ValueError: if no residues are provided, none fall within the
            pocket radius, or the resulting pocket graph has no edges.
    """
    if len(residues) == 0:
        raise ValueError("No protein residues provided")

    ca_coords = np.array([r.ca_coord for r in residues], dtype=np.float64)
    ligand_positions = np.asarray(ligand_positions, dtype=np.float64)

    dists_to_ligand = np.linalg.norm(ca_coords[:, None, :] - ligand_positions[None, :, :], axis=-1).min(axis=1)
    pocket_mask = dists_to_ligand <= pocket_radius_angstrom
    pocket_residues = [r for r, keep in zip(residues, pocket_mask) if keep]

    if len(pocket_residues) == 0:
        raise ValueError(
            f"No residues found within {pocket_radius_angstrom} Angstrom of the ligand; "
            "check ligand coordinates / cutoff."
        )

    pocket_ca = torch.tensor([r.ca_coord for r in pocket_residues], dtype=torch.float32)
    node_features = torch.tensor([_one_hot_residue(r.resname) for r in pocket_residues], dtype=torch.float32)

    pairwise = torch.cdist(pocket_ca, pocket_ca)
    within_cutoff = (pairwise <= edge_cutoff_angstrom) & (pairwise > 0)
    edge_index = within_cutoff.nonzero(as_tuple=False).t().contiguous()

    if edge_index.numel() == 0:
        raise ValueError(
            f"Pocket graph has no edges within {edge_cutoff_angstrom} Angstrom; "
            "cutoff may be too small for this pocket."
        )

    return Data(x=node_features, pos=pocket_ca, edge_index=edge_index)
