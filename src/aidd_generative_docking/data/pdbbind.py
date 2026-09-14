"""A small public-PDB "PDBBind-format" subset for smoke-testing the pipeline.

The officially licensed PDBBind dataset (pdbbind.org.cn) requires a
registration agreement that has not been obtained for this project. PDBBind
itself is curated directly from PDB co-crystal structures, so this module
substitutes a small, hand-picked set of public, unrestricted RCSB PDB
entries with the same structural content (one protein chain plus one bound
small-molecule ligand, from a single experimental structure) as a stand-in.
This exercises the full data pipeline -- parsing, pocket extraction,
featurization -- against real 3D structures now. Swapping in the licensed
PDBBind refined/core set later (stage 6) only requires changing
``DEFAULT_MANIFEST`` and the raw-file layout; ``sample.py`` and
``featurize.py`` do not need to change.

One structural difference from licensed PDBBind is worth noting: RCSB
deposits the protein and the bound ligand together in one file, whereas
PDBBind ships the ligand separately as an SDF/MOL2 file with correct bond
orders. Here, both protein and ligand come from the same downloaded file,
which is why ligand bond order is not chemically perceived -- see the
``featurize.py`` module docstring for the full explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aidd_generative_docking.data.pdb_io import download_pdb_file
from aidd_generative_docking.data.sample import ComplexSample, build_training_sample

DEFAULT_ROOT = Path("data/raw/pdbbind_subset")


@dataclass(frozen=True)
class ManifestEntry:
    """One curated complex: which PDB entry, and which residue is the ligand."""

    complex_id: str
    ligand_resname: str
    description: str


DEFAULT_MANIFEST: list[ManifestEntry] = [
    ManifestEntry(
        "3PTB", "BEN", "Beta-trypsin + benzamidine: tiny ligand (9 heavy atoms), classic protease pocket."
    ),
    ManifestEntry(
        "1STP", "BTN", "Streptavidin + biotin: small ligand, extremely high-affinity buried pocket."
    ),
    ManifestEntry(
        "1AZM", "AZM", "Human carbonic anhydrase II + acetazolamide: drug-like sulfonamide ligand."
    ),
]


class PDBBindSubset:
    """Loads / downloads the small public smoke-test subset defined above.

    Args:
        root: local directory holding downloaded structure files
            (layout: ``<root>/<complex_id>/<complex_id>.pdb``).
        manifest: which complexes to include (defaults to
            ``DEFAULT_MANIFEST``).
        pocket_radius_angstrom: passed through to
            ``featurize.featurize_protein_pocket``.
        edge_cutoff_angstrom: passed through to
            ``featurize.featurize_protein_pocket``.
    """

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        manifest: list[ManifestEntry] | None = None,
        pocket_radius_angstrom: float = 10.0,
        edge_cutoff_angstrom: float = 8.0,
    ) -> None:
        self.root = Path(root)
        self.manifest = {entry.complex_id: entry for entry in (manifest or DEFAULT_MANIFEST)}
        self.pocket_radius_angstrom = pocket_radius_angstrom
        self.edge_cutoff_angstrom = edge_cutoff_angstrom

    def list_complex_ids(self) -> list[str]:
        """Return the complex identifiers in this subset's manifest."""
        return list(self.manifest.keys())

    def pdb_path(self, complex_id: str) -> Path:
        """Local path where ``complex_id``'s structure file is (or will be) stored."""
        return self.root / complex_id / f"{complex_id}.pdb"

    def ensure_downloaded(self, complex_id: str) -> Path:
        """Download the complex's structure file from RCSB if not already local."""
        path = self.pdb_path(complex_id)
        if not path.exists():
            download_pdb_file(complex_id, path)
        return path

    def build_sample(self, complex_id: str) -> ComplexSample:
        """Download (if needed) and featurize one complex into a training sample.

        Raises:
            KeyError: if ``complex_id`` is not in this subset's manifest.
        """
        if complex_id not in self.manifest:
            raise KeyError(f"Unknown complex_id '{complex_id}'; not in manifest")
        entry = self.manifest[complex_id]
        path = self.ensure_downloaded(complex_id)
        return build_training_sample(
            pdb_path=path,
            complex_id=complex_id,
            ligand_resname=entry.ligand_resname,
            pocket_radius_angstrom=self.pocket_radius_angstrom,
            edge_cutoff_angstrom=self.edge_cutoff_angstrom,
        )
