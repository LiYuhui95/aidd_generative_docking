"""Tests for candidate selection and validation against the real PDBBind
archive. Skipped entirely if the licensed archive isn't present locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aidd_generative_docking.data.dataset_builder import (
    build_validated_dataset,
    select_candidate_ids,
    summarize_exclusions_by_family,
)
from aidd_generative_docking.data.pdb_io import read_pdbbind_general_index

RAW_DIR = Path("data/raw")
ARCHIVE_PATH = RAW_DIR / "P-L.tar.gz"
INDEX_PATH = RAW_DIR / ".cache" / "index" / "index" / "INDEX_general_PL.2020R1.lst"

pytestmark = pytest.mark.skipif(
    not (ARCHIVE_PATH.exists() and INDEX_PATH.exists()), reason="PDBBind archive/index not present under data/raw/"
)


def test_select_candidate_ids_is_deterministic_and_excludes_covalent() -> None:
    entries = read_pdbbind_general_index(INDEX_PATH)
    ids_a = select_candidate_ids(entries, pool_size=20, seed=42)
    ids_b = select_candidate_ids(entries, pool_size=20, seed=42)
    assert ids_a == ids_b
    assert len(ids_a) == 20

    covalent_ids = {e.complex_id for e in entries if "covalent" in e.notes.lower()}
    assert not (set(ids_a) & covalent_ids)


def test_select_candidate_ids_different_seed_differs() -> None:
    entries = read_pdbbind_general_index(INDEX_PATH)
    ids_a = select_candidate_ids(entries, pool_size=20, seed=1)
    ids_b = select_candidate_ids(entries, pool_size=20, seed=2)
    assert ids_a != ids_b


def test_build_validated_dataset_documents_exclusions(tmp_path) -> None:
    entries = read_pdbbind_general_index(INDEX_PATH)
    ids = select_candidate_ids(entries, pool_size=15, seed=0)

    validated, report = build_validated_dataset(ARCHIVE_PATH, ids, tmp_path, max_ligand_atoms=100)

    assert report.num_candidates == 15
    assert report.num_valid == len(validated)
    assert report.num_valid + sum(report.exclusion_counts.values()) == report.num_candidates
    for v in validated:
        assert v.sample.ligand.x.shape[0] <= 100
        assert len(v.protein_sequence) > 0
        assert v.ligand_fingerprint is not None

    families = summarize_exclusions_by_family(report)
    assert sum(families.values()) == sum(report.exclusion_counts.values())
