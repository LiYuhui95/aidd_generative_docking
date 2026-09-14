"""Shared fixtures for data-pipeline tests.

Tests that need real structures use the ``pdbbind_subset`` fixture, which
downloads (and locally caches under ``data/raw/pdbbind_subset/``, which is
gitignored) a few small public PDB files the first time it runs. If no
network is reachable, dependent tests are skipped rather than failed.
"""

from __future__ import annotations

import pytest

from aidd_generative_docking.data.pdbbind import PDBBindSubset


@pytest.fixture(scope="session")
def pdbbind_subset() -> PDBBindSubset:
    subset = PDBBindSubset()
    try:
        for complex_id in subset.list_complex_ids():
            subset.ensure_downloaded(complex_id)
    except OSError as exc:
        pytest.skip(f"No network access to download PDB test data: {exc}")
    return subset
