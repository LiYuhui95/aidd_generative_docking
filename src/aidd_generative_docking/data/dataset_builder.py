"""Build and validate a manageable (500-1000 complex) dataset from the full
PDBBind v2020.R1 archive, for the final held-out generalization experiment.

Additive module: does not modify ``pdb_io.py``, ``featurize.py``,
``sample.py``, or ``pdbbind.py`` (the 3-complex RCSB smoke-test manifest in
``pdbbind.py`` is untouched and unrelated to this larger, real pipeline).
"""

from __future__ import annotations

import random
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from aidd_generative_docking.data.pdb_io import PDBBindIndexEntry, extract_pdbbind_complexes, read_pdbbind_general_index
from aidd_generative_docking.data.sample import ComplexSample, build_training_sample_from_files
from aidd_generative_docking.data.splitting import compute_ligand_fingerprint, extract_protein_sequence

DEFAULT_POCKET_RADIUS_ANGSTROM = 10.0
DEFAULT_EDGE_CUTOFF_ANGSTROM = 8.0
DEFAULT_MAX_LIGAND_ATOMS = 100  # total atoms incl. explicit H; excludes large peptide-like outliers


@dataclass
class ValidatedComplex:
    complex_id: str
    sample: ComplexSample
    protein_sequence: str
    ligand_fingerprint: object


@dataclass
class DatasetBuildReport:
    """Bookkeeping for the "validate every complex, document exclusions" requirement."""

    num_candidates: int = 0
    num_valid: int = 0
    exclusion_counts: dict[str, int] = field(default_factory=dict)
    excluded_examples: dict[str, list[str]] = field(default_factory=dict)  # reason -> a few example ids

    def record_exclusion(self, complex_id: str, reason: str) -> None:
        self.exclusion_counts[reason] = self.exclusion_counts.get(reason, 0) + 1
        examples = self.excluded_examples.setdefault(reason, [])
        if len(examples) < 5:
            examples.append(complex_id)


def select_candidate_ids(
    index_entries: list[PDBBindIndexEntry],
    pool_size: int,
    seed: int,
    exclude_covalent: bool = True,
) -> list[str]:
    """Deterministically sample a candidate pool of complex ids from the index.

    Args:
        index_entries: parsed ``INDEX_general_PL.2020R1.lst`` entries.
        pool_size: number of candidates to sample (before validation).
        seed: RNG seed (Python's ``random``, seeded explicitly for
            reproducibility -- this is the only randomized step in dataset
            construction; the similarity-aware split itself is RNG-free).
        exclude_covalent: skip entries whose index notes mention "covalent"
            (our rigid, non-covalent binding assumption -- see
            ARCHITECTURE.md -- does not apply to covalent complexes).

    Returns:
        A sorted list of ``pool_size`` complex ids (or fewer, if the
        filtered index has fewer entries than requested).
    """
    candidates = [e.complex_id for e in index_entries if not (exclude_covalent and "covalent" in e.notes.lower())]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:pool_size])


def build_validated_dataset(
    archive_path: Path | str,
    candidate_ids: list[str],
    cache_dir: Path | str,
    pocket_radius_angstrom: float = DEFAULT_POCKET_RADIUS_ANGSTROM,
    edge_cutoff_angstrom: float = DEFAULT_EDGE_CUTOFF_ANGSTROM,
    max_ligand_atoms: int = DEFAULT_MAX_LIGAND_ATOMS,
) -> tuple[list[ValidatedComplex], DatasetBuildReport]:
    """Extract, featurize, and validate every candidate complex.

    Each candidate is featurized independently; failures are caught and
    recorded with a specific reason rather than aborting the whole build.

    Args:
        archive_path: path to ``P-L.tar.gz``.
        candidate_ids: complex ids to attempt (see ``select_candidate_ids``).
        cache_dir: local extraction cache (gitignored).
        pocket_radius_angstrom, edge_cutoff_angstrom: unchanged from
            ``featurize.featurize_protein_pocket`` defaults used throughout
            this project.
        max_ligand_atoms: exclusion cap on total ligand atom count
            (including explicit hydrogens); see module docstring.

    Returns:
        ``(validated, report)``: the list of successfully validated
        complexes (with their protein sequence and ligand fingerprint,
        precomputed here for the similarity-aware split) and a
        ``DatasetBuildReport`` documenting every exclusion.
    """
    report = DatasetBuildReport(num_candidates=len(candidate_ids))
    paths_by_id = extract_pdbbind_complexes(archive_path, candidate_ids, cache_dir)

    validated: list[ValidatedComplex] = []
    for complex_id in candidate_ids:
        paths = paths_by_id[complex_id]
        try:
            sample = build_training_sample_from_files(
                protein_pdb_path=paths["protein"],
                ligand_sdf_path=paths["ligand_sdf"],
                ligand_mol2_path=paths["ligand_mol2"],
                complex_id=complex_id,
                pocket_radius_angstrom=pocket_radius_angstrom,
                edge_cutoff_angstrom=edge_cutoff_angstrom,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: documenting *why* each complex failed
            reason = f"{type(exc).__name__}: {exc}"
            report.record_exclusion(complex_id, reason)
            continue

        if sample.ligand.x.shape[0] > max_ligand_atoms:
            report.record_exclusion(complex_id, f"ligand too large (> {max_ligand_atoms} atoms)")
            continue

        try:
            protein_sequence = extract_protein_sequence(paths["protein"])
            if len(protein_sequence) == 0:
                report.record_exclusion(complex_id, "no standard-residue protein sequence extracted")
                continue

            # Re-parse the ligand mol purely for fingerprinting (kept separate
            # from featurize_ligand's internal Data-object construction to
            # avoid changing that function's return type).
            from aidd_generative_docking.data.featurize import load_ligand_mol

            mol = load_ligand_mol(paths["ligand_sdf"], paths["ligand_mol2"])
            fingerprint = compute_ligand_fingerprint(mol)
        except Exception as exc:  # noqa: BLE001
            report.record_exclusion(complex_id, f"grouping-feature extraction failed: {type(exc).__name__}: {exc}")
            continue

        validated.append(
            ValidatedComplex(
                complex_id=complex_id, sample=sample, protein_sequence=protein_sequence, ligand_fingerprint=fingerprint
            )
        )

    report.num_valid = len(validated)
    return validated, report


def summarize_exclusions_by_family(report: DatasetBuildReport) -> dict[str, int]:
    """Collapse the (very specific) exclusion reasons into coarse families
    for a readable report, keeping the fine-grained detail available on
    ``report.exclusion_counts`` / ``report.excluded_examples``.
    """
    families: dict[str, int] = {}
    for reason, count in report.exclusion_counts.items():
        if "ligand too large" in reason:
            family = "ligand too large"
        elif "No HETATM records" in reason:
            family = "ligand resname not found in structure"
        elif "RDKit failed to parse ligand" in reason or "grouping-feature extraction failed" in reason:
            family = "RDKit could not parse ligand (SDF and MOL2 both failed)"
        elif "no atoms" in reason or "no bonds" in reason:
            family = "ligand has no atoms/bonds after parsing"
        elif "No residues found within" in reason:
            family = "no pocket residues within radius of ligand"
        elif "no edges within" in reason:
            family = "pocket graph has no edges at this cutoff"
        elif "no standard-residue protein sequence" in reason:
            family = "protein has no standard-residue chain"
        else:
            family = "other"
        families[family] = families.get(family, 0) + count
    return families
