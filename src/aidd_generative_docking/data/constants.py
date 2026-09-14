"""Shared vocabularies for protein/ligand featurization.

Kept dependency-free (no RDKit/Biopython imports) so these lists can be
reused by any module without pulling in extra imports.
"""

from __future__ import annotations

STANDARD_AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

# Common heavy elements in drug-like ligands; anything else falls into an
# "other" bucket appended at runtime (see featurize.py).
LIGAND_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]

# Bond-type vocabulary for ligand edge features, as RDKit bond-type names.
BOND_TYPE_NAMES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
