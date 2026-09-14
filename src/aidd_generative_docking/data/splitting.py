"""Deterministic, protein/ligand-similarity-aware train/val/test splitting.

Additive module for the final held-out experiment; does not modify any
existing preprocessing interface (``pdb_io.py``, ``featurize.py``,
``sample.py``, ``pdbbind.py`` are all untouched).

Why similarity-aware grouping, and what "aware" means here
-------------------------------------------------------------
PDBBind contains many near-duplicate entries: the same protein target
solved dozens of times with different ligands (e.g. HIV protease,
thrombin, many kinases), and the same ligand scaffold bound to several
related targets. A naive random split puts near-identical
protein/ligand pairs on both sides of train and test, which inflates
apparent generalization. Full sequence-alignment or scaffold-fingerprint
clustering at PDBBind's full scale normally uses external tools
(MMseqs2, CD-HIT) that are not part of this project's dependency set;
this module instead uses two cheap, deterministic, dependency-free (given
what's already installed -- Biopython, RDKit) proxies that catch the most
common and severe leakage case in PDBBind:

1. **Protein grouping by exact sequence match.** Complexes whose pocket's
   parent protein has the *exact same* residue sequence (extracted from
   ``ATOM`` records) are grouped together. This directly catches "same
   target, many co-crystal structures with different ligands," the
   single most common redundancy pattern in PDBBind.
2. **Ligand grouping by Tanimoto similarity clustering.** Morgan
   fingerprints (radius 2, 1024 bits) + RDKit's Butina clustering group
   near-identical ligands (e.g. an analog series against one target).

Any two complexes sharing a protein group OR a ligand group are merged
(via union-find) into one combined "leakage group," and whole groups --
never individual complexes -- are assigned to train/val/test, using a
fully deterministic (no RNG) greedy proportional bin-packing: groups are
processed largest-first, each going to whichever split is currently
furthest below its target share. This is a real but intentionally modest
leakage-control measure, not a full-scale reproduction of a published
protocol (e.g. LP-PDBBind) -- see README/ARCHITECTURE for the honest
framing of this project's held-out experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import PDBParser
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

# Minimal 3-letter -> 1-letter mapping for the 20 standard amino acids,
# defined locally rather than relying on Biopython's own (version-unstable)
# three_to_one helper.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def extract_protein_sequence(pdb_path: Path | str) -> str:
    """Extract the 1-letter amino-acid sequence of the first chain with any
    standard residues, from a PDBBind ``<id>_protein.pdb`` file.

    Used only as a similarity-grouping key, not for any modeling purpose.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    model = next(iter(structure))
    residues = []
    for chain in model:
        for residue in chain:
            hetfield, _seq_id, _icode = residue.id
            if hetfield.strip():
                continue
            one_letter = _THREE_TO_ONE.get(residue.resname)
            if one_letter is not None:
                residues.append(one_letter)
        if residues:
            break  # first chain with standard residues is enough for a grouping key
    return "".join(residues)


def compute_ligand_fingerprint(mol: Chem.Mol):
    """Morgan fingerprint (radius 2, 1024 bits) for ligand-similarity grouping."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)


class UnionFind:
    """Minimal union-find (disjoint-set) for merging protein/ligand groups."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def cluster_by_exact_match(keys: list[str]) -> list[int]:
    """Assign a cluster id to each item, grouping identical keys together."""
    seen: dict[str, int] = {}
    cluster_ids = []
    for key in keys:
        if key not in seen:
            seen[key] = len(seen)
        cluster_ids.append(seen[key])
    return cluster_ids


def cluster_ligands_by_similarity(fingerprints: list, similarity_threshold: float = 0.4) -> list[int]:
    """Cluster ligand fingerprints with RDKit's Butina algorithm.

    Args:
        fingerprints: list of RDKit bit vectors (see ``compute_ligand_fingerprint``).
        similarity_threshold: Tanimoto similarity above which two ligands
            are considered "the same scaffold" for grouping purposes.
            Butina takes a *distance* threshold, so this is converted as
            ``distance_threshold = 1 - similarity_threshold``.

    Returns:
        A cluster id per fingerprint (deterministic given fixed input order).
    """
    n = len(fingerprints)
    if n == 0:
        return []
    distance_threshold = 1.0 - similarity_threshold
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, distance_threshold, isDistData=True)

    cluster_ids = [-1] * n
    for cluster_id, members in enumerate(clusters):
        for member_idx in members:
            cluster_ids[member_idx] = cluster_id
    return cluster_ids


@dataclass
class SplitAssignment:
    """Deterministic group-aware split result."""

    split_by_complex_id: dict[str, str]  # complex_id -> "train" | "val" | "test"
    group_by_complex_id: dict[str, int]  # complex_id -> leakage-group id
    num_groups: int


def assign_similarity_aware_split(
    complex_ids: list[str],
    protein_sequences: list[str],
    ligand_fingerprints: list,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    ligand_similarity_threshold: float = 0.4,
) -> SplitAssignment:
    """Assign complexes to train/val/test by whole leakage-group, deterministically.

    Args:
        complex_ids: complex identifiers, in a fixed canonical order (this
            order is part of what makes the result deterministic).
        protein_sequences: parallel list, from ``extract_protein_sequence``.
        ligand_fingerprints: parallel list, from ``compute_ligand_fingerprint``.
        train_frac, val_frac, test_frac: target proportions (must sum to ~1).
        ligand_similarity_threshold: see ``cluster_ligands_by_similarity``.

    Returns:
        A ``SplitAssignment`` with a split label and leakage-group id per
        complex, and the total number of leakage groups found.
    """
    n = len(complex_ids)
    protein_clusters = cluster_by_exact_match(protein_sequences)
    ligand_clusters = cluster_ligands_by_similarity(ligand_fingerprints, ligand_similarity_threshold)

    # merge protein-cluster-id-space and ligand-cluster-id-space into one
    # union-find over complexes: two complexes merge if they share EITHER.
    uf = UnionFind(n)
    first_seen_protein: dict[int, int] = {}
    first_seen_ligand: dict[int, int] = {}
    for idx in range(n):
        pc = protein_clusters[idx]
        if pc in first_seen_protein:
            uf.union(idx, first_seen_protein[pc])
        else:
            first_seen_protein[pc] = idx

        lc = ligand_clusters[idx]
        if lc in first_seen_ligand:
            uf.union(idx, first_seen_ligand[lc])
        else:
            first_seen_ligand[lc] = idx

    group_of: dict[int, int] = {}
    members_of_group: dict[int, list[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        if root not in group_of:
            group_of[root] = len(group_of)
        gid = group_of[root]
        members_of_group.setdefault(gid, []).append(idx)

    # Deterministic, RNG-free greedy proportional bin-packing: largest
    # groups first, each to whichever split is furthest below its target
    # share of complexes assigned so far.
    groups_sorted = sorted(members_of_group.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    targets = {"train": train_frac, "val": val_frac, "test": test_frac}
    counts = {"train": 0, "val": 0, "test": 0}
    split_by_idx: dict[int, str] = {}

    for _gid, member_indices in groups_sorted:
        total_assigned = sum(counts.values()) or 1
        deficits = {split: targets[split] - counts[split] / total_assigned for split in targets}
        # Break ties (common when e.g. val_frac == test_frac) by preferring
        # the split with fewer members so far, rather than dict/insertion
        # order -- otherwise one split could be starved indefinitely.
        chosen_split = max(targets, key=lambda split: (deficits[split], -counts[split]))
        for idx in member_indices:
            split_by_idx[idx] = chosen_split
        counts[chosen_split] += len(member_indices)

    split_by_complex_id = {complex_ids[idx]: split_by_idx[idx] for idx in range(n)}
    group_by_complex_id = {complex_ids[idx]: uf.find(idx) for idx in range(n)}

    return SplitAssignment(
        split_by_complex_id=split_by_complex_id,
        group_by_complex_id=group_by_complex_id,
        num_groups=len(members_of_group),
    )
