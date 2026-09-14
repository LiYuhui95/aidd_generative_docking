"""Tests for building complete training samples from the PDBBind-format subset."""

from __future__ import annotations

import pytest
import torch

from aidd_generative_docking.data.pdbbind import PDBBindSubset


def test_build_sample_for_every_manifest_complex(pdbbind_subset: PDBBindSubset) -> None:
    for complex_id in pdbbind_subset.list_complex_ids():
        sample = pdbbind_subset.build_sample(complex_id)

        assert sample.complex_id == complex_id

        assert sample.ligand.x.shape[0] > 0
        assert sample.ligand.pos.shape[0] == sample.ligand.x.shape[0]
        assert sample.ligand.edge_index.shape[1] > 0
        assert not torch.isnan(sample.ligand.x).any()
        assert not torch.isnan(sample.ligand.pos).any()

        assert sample.pocket.x.shape[0] > 0
        assert sample.pocket.pos.shape[0] == sample.pocket.x.shape[0]
        assert sample.pocket.edge_index.shape[1] > 0
        assert not torch.isnan(sample.pocket.x).any()
        assert not torch.isnan(sample.pocket.pos).any()


def test_unknown_complex_id_raises(pdbbind_subset: PDBBindSubset) -> None:
    with pytest.raises(KeyError):
        pdbbind_subset.build_sample("NOTAREALCOMPLEX")


def test_list_complex_ids_matches_manifest(pdbbind_subset: PDBBindSubset) -> None:
    ids = pdbbind_subset.list_complex_ids()
    assert set(ids) == {"3PTB", "1STP", "1AZM"}
    assert len(ids) == len(set(ids))  # no duplicates
