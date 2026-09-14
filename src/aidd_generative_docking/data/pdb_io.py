"""Low-level structure I/O.

Two ligand/protein sources are supported side by side:

1. **Legacy RCSB smoke test** (``extract_ligand_pdb_block`` +
   ``download_pdb_file``): plain public RCSB PDB entries used before the
   real, licensed PDBBind archive was available locally. Kept for backward
   compatibility with the original 3-complex manifest in ``pdbbind.py`` and
   its tests; bond order is not chemically perceived from these files (see
   ``featurize.py``).
2. **Real PDBBind v2020.R1 archive** (``extract_pdbbind_complexes`` +
   ``read_pdbbind_general_index``): the officially licensed, registered
   download the user obtained from pdbbind-plus.org.cn, stored as
   ``data/raw/P-L.tar.gz`` (19,037 complexes) and ``data/raw/index.tar.gz``.
   Ligand bond order/aromaticity/charge come from PDBBind's own ligand SDF
   (MOL2 fallback), not from PDB CONECT guessing. See ``pdbbind.py`` module
   docstring for license/version/redistribution details.
"""

from __future__ import annotations

import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import PDBParser

from aidd_generative_docking.data.constants import STANDARD_AMINO_ACIDS

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

PDBBIND_YEAR_BUCKETS = ("1981-2000", "2001-2010", "2011-2019")
PDBBIND_LIGAND_FILE_KINDS = ("protein.pdb", "pocket.pdb", "ligand.sdf", "ligand.mol2")


def download_pdb_file(pdb_id: str, dest_path: Path | str) -> Path:
    """Download one structure file from RCSB PDB (public, unauthenticated).

    Args:
        pdb_id: 4-character PDB accession code (e.g. ``"3PTB"``).
        dest_path: local file path to write the downloaded structure to.

    Returns:
        ``dest_path``, for convenient chaining.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    with urllib.request.urlopen(url, timeout=30) as response:
        dest_path.write_bytes(response.read())
    return dest_path


def extract_ligand_pdb_block(pdb_text: str, resname: str) -> str:
    """Return a minimal PDB block (HETATM + CONECT lines) for one ligand residue.

    Connectivity comes from the file's own CONECT records as deposited by
    RCSB, which is more reliable than RDKit's distance-based bond guessing
    and preserves the experimentally observed atom graph. Note that PDB
    CONECT records do not encode bond *order* -- see ``featurize.py`` for
    how that limitation is handled and documented.

    Args:
        pdb_text: full contents of a PDB structure file.
        resname: 3-character ligand residue name (e.g. ``"BEN"``).

    Returns:
        A small PDB-format text block containing just this ligand's
        HETATM and CONECT records, terminated with ``END``.

    Raises:
        ValueError: if no HETATM records match ``resname``.
    """
    lines = pdb_text.splitlines()
    het_lines = [line for line in lines if line.startswith("HETATM") and line[17:20].strip() == resname]
    if not het_lines:
        raise ValueError(f"No HETATM records found for ligand residue '{resname}'")
    serials = {line[6:11].strip() for line in het_lines}
    conect_lines = [line for line in lines if line.startswith("CONECT") and line[6:11].strip() in serials]
    return "\n".join([*het_lines, *conect_lines, "END"])


@dataclass
class ProteinResidue:
    """One standard-amino-acid residue, reduced to its Cα geometry."""

    resname: str
    chain_id: str
    seq_id: int
    ca_coord: tuple[float, float, float]


def parse_protein_residues(pdb_path: Path | str) -> list[ProteinResidue]:
    """Parse standard amino-acid residues with a resolved Cα atom.

    Water, ligands, ions, and non-standard/modified residues (i.e. anything
    not in ``constants.STANDARD_AMINO_ACIDS``, and any residue missing a
    resolved CA atom) are skipped. Only the first model is used (X-ray
    structures are single-model; NMR-style multi-model files would need
    different handling, out of scope here).

    Args:
        pdb_path: path to a PDB structure file.

    Returns:
        List of ``ProteinResidue``, in file order.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    residues: list[ProteinResidue] = []
    model = next(iter(structure))
    for chain in model:
        for residue in chain:
            hetfield, seq_id, _icode = residue.id
            if hetfield.strip():
                continue  # hetero / water residue, not a standard protein residue
            if residue.resname not in STANDARD_AMINO_ACIDS:
                continue
            if "CA" not in residue:
                continue
            ca = residue["CA"].coord
            residues.append(
                ProteinResidue(
                    resname=residue.resname,
                    chain_id=chain.id,
                    seq_id=seq_id,
                    ca_coord=(float(ca[0]), float(ca[1]), float(ca[2])),
                )
            )
    return residues


def extract_pdbbind_complexes(
    archive_path: Path | str,
    complex_ids: list[str],
    dest_dir: Path | str,
) -> dict[str, dict[str, Path]]:
    """Extract a small number of complexes from the PDBBind ``P-L.tar.gz`` archive.

    Deliberately does **not** extract the whole (~3.1 GB, 19,037-complex)
    archive: it streams through the ``.tar.gz`` once, pulling out only the
    four files (``protein.pdb``, ``pocket.pdb``, ``ligand.sdf``,
    ``ligand.mol2``) for the requested ``complex_ids``, and stops as soon as
    every requested complex has been found. Already-extracted complexes
    (all 4 files present under ``dest_dir``) are skipped entirely -- no
    archive scan happens if everything requested is already cached. The
    archive itself is opened read-only and is never modified.

    A full streaming pass over the archive costs roughly 30-40s regardless
    of how many complexes are requested (gzip is not randomly seekable), so
    this function is meant to be called once with the *entire* list of
    complex_ids needed for a run, not once per complex.

    Args:
        archive_path: path to the downloaded ``P-L.tar.gz``.
        complex_ids: lowercase 4-character PDBBind complex identifiers.
        dest_dir: local cache directory; files land at
            ``<dest_dir>/<complex_id>/<complex_id>_<kind>``.

    Returns:
        Mapping ``complex_id -> {"protein": Path, "pocket": Path,
        "ligand_sdf": Path, "ligand_mol2": Path}`` for every complex found.

    Raises:
        KeyError: if any requested complex_id is not found in the archive.
    """
    dest_dir = Path(dest_dir)
    remaining = set(complex_ids)
    result: dict[str, dict[str, Path]] = {}

    for complex_id in list(remaining):
        complex_dir = dest_dir / complex_id
        paths = {
            "protein": complex_dir / f"{complex_id}_protein.pdb",
            "pocket": complex_dir / f"{complex_id}_pocket.pdb",
            "ligand_sdf": complex_dir / f"{complex_id}_ligand.sdf",
            "ligand_mol2": complex_dir / f"{complex_id}_ligand.mol2",
        }
        if all(p.exists() for p in paths.values()):
            result[complex_id] = paths
            remaining.discard(complex_id)

    if not remaining:
        return result

    wanted_members = {
        f"P-L/{bucket}/{cid}/{cid}_{kind}": cid
        for cid in remaining
        for bucket in PDBBIND_YEAR_BUCKETS
        for kind in PDBBIND_LIGAND_FILE_KINDS
    }

    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name not in wanted_members:
                continue
            complex_id = wanted_members[member.name]
            complex_dir = dest_dir / complex_id
            complex_dir.mkdir(parents=True, exist_ok=True)
            dest_path = complex_dir / Path(member.name).name
            with archive.extractfile(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

            complex_dir_full = {
                "protein": complex_dir / f"{complex_id}_protein.pdb",
                "pocket": complex_dir / f"{complex_id}_pocket.pdb",
                "ligand_sdf": complex_dir / f"{complex_id}_ligand.sdf",
                "ligand_mol2": complex_dir / f"{complex_id}_ligand.mol2",
            }
            if all(p.exists() for p in complex_dir_full.values()):
                result[complex_id] = complex_dir_full
                remaining.discard(complex_id)
                if not remaining:
                    break

    if remaining:
        raise KeyError(f"Complex id(s) not found in archive: {sorted(remaining)}")

    return result


@dataclass
class PDBBindIndexEntry:
    """One row of a PDBBind ``INDEX_general_PL*.lst`` file."""

    complex_id: str
    resolution: str
    release_year: int
    affinity_measure: str  # "Kd", "Ki", or "IC50"
    affinity_operator: str  # "=", "<", ">", "~", "<=", ">="
    affinity_value: float
    affinity_unit: str  # e.g. "nM", "uM", "mM", "fM", "pM"
    ligand_name: str
    notes: str  # free-text after the ligand name, e.g. "covalent complex"


def read_pdbbind_general_index(index_path: Path | str) -> list[PDBBindIndexEntry]:
    """Parse a PDBBind ``INDEX_general_PL*.lst`` file.

    Format (whitespace-delimited, one complex per non-comment line)::

        <pdb_id>  <resolution>  <year>  <measure><op><value><unit>  // <ref>.pdf (<ligand_name>) <notes>

    e.g. ``1azm  2.00  1994  Ki=0.8uM       // 1azm.pdf (AZM) ligand is compound 2``.

    Args:
        index_path: path to an ``INDEX_general_PL*.lst`` file.

    Returns:
        Parsed entries, skipping comment lines (starting with ``#``).
    """
    pattern = re.compile(
        r"^(?P<pdb_id>\S+)\s+(?P<resolution>\S+)\s+(?P<year>\d+)\s+"
        r"(?P<measure>Kd|Ki|IC50)(?P<op><=|>=|[=<>~])(?P<value>[\d.]+)(?P<unit>[a-zA-Z]+)"
        r"\s*//\s*\S+\s*\((?P<ligand_name>[^)]*)\)\s*(?P<notes>.*)$"
    )
    entries: list[PDBBindIndexEntry] = []
    with open(index_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            match = pattern.match(line)
            if match is None:
                continue  # a small number of malformed/free-text lines are skipped
            entries.append(
                PDBBindIndexEntry(
                    complex_id=match["pdb_id"],
                    resolution=match["resolution"],
                    release_year=int(match["year"]),
                    affinity_measure=match["measure"],
                    affinity_operator=match["op"],
                    affinity_value=float(match["value"]),
                    affinity_unit=match["unit"],
                    ligand_name=match["ligand_name"],
                    notes=match["notes"].strip(),
                )
            )
    return entries
