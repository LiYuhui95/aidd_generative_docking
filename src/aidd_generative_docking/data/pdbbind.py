"""PDBBind subset loading and preprocessing (stage 1).

This module defines the intended interface for building a small PDBBind
subset into PyTorch Geometric-ready examples. No data is downloaded or read
by this module yet; methods raise ``NotImplementedError`` until stage 1/2
are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDBBindExample:
    """A single protein-pocket / ligand complex, pre-featurization.

    Attributes:
        complex_id: PDBBind identifier (e.g. a 4-character PDB code).
        protein_pdb_path: path to the protein structure file.
        ligand_sdf_path: path to the ligand structure file (RDKit-readable).
    """

    complex_id: str
    protein_pdb_path: Path
    ligand_sdf_path: Path


class PDBBindSubset:
    """Loads a small, locally-provided subset of PDBBind complexes.

    This class does not download anything: ``root`` must already contain
    the raw structure files. It exists to define the dataset interface
    that later preprocessing (pocket extraction, RDKit parsing) will build
    on top of.

    Args:
        root: directory containing raw PDBBind subset files
            (expected layout: ``data/raw/pdbbind_subset/<complex_id>/``).
        pocket_radius_angstrom: residue selection radius around the ligand.
    """

    def __init__(self, root: Path | str, pocket_radius_angstrom: float = 10.0) -> None:
        self.root = Path(root)
        self.pocket_radius_angstrom = pocket_radius_angstrom

    def list_complex_ids(self) -> list[str]:
        """Return complex identifiers found under ``root``."""
        raise NotImplementedError("Stage 1: dataset curation not yet implemented.")

    def load_example(self, complex_id: str) -> PDBBindExample:
        """Load raw file paths for a single complex."""
        raise NotImplementedError("Stage 1: dataset curation not yet implemented.")
