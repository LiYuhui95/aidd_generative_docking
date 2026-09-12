"""Protein pocket and ligand graph featurization (stage 2).

RDKit is imported lazily inside functions rather than at module scope, so
that the rest of the package remains importable in environments where
RDKit has not been installed yet (see README: RDKit is a declared but not
yet installed dependency in ``sxt-torch``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def featurize_ligand(ligand_sdf_path: Path | str) -> Any:
    """Parse a ligand file into an atom/bond feature graph.

    Intended to return a PyTorch Geometric ``Data`` object with atomic
    number, formal charge, hybridization, and bond-order features, plus
    3D conformer coordinates. Not yet implemented (stage 2).
    """
    from rdkit import Chem  # noqa: F401  (lazy import; see module docstring)

    raise NotImplementedError("Stage 2: ligand featurization not yet implemented.")


def featurize_protein_pocket(protein_pdb_path: Path | str, pocket_radius_angstrom: float) -> Any:
    """Extract a residue-level pocket graph around the bound ligand.

    Intended to return a PyTorch Geometric ``Data`` object with residue
    identity features and Cα coordinates. Not yet implemented (stage 2).
    """
    raise NotImplementedError("Stage 2: protein pocket featurization not yet implemented.")
