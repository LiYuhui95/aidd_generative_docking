"""Tests for deterministic, protein/ligand-similarity-aware splitting."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from aidd_generative_docking.data.splitting import (
    UnionFind,
    assign_similarity_aware_split,
    cluster_by_exact_match,
    cluster_ligands_by_similarity,
)


def _fp(smiles: str):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=1024)


def test_union_find_basic() -> None:
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) != uf.find(0)


def test_cluster_by_exact_match_groups_duplicates() -> None:
    keys = ["AAA", "BBB", "AAA", "CCC", "BBB"]
    clusters = cluster_by_exact_match(keys)
    assert clusters[0] == clusters[2]
    assert clusters[1] == clusters[4]
    assert clusters[0] != clusters[1] != clusters[3]


def test_cluster_ligands_by_similarity_groups_identical_ligands() -> None:
    fps = [_fp("CCO"), _fp("CCO"), _fp("c1ccccc1"), _fp("CCCl")]
    clusters = cluster_ligands_by_similarity(fps, similarity_threshold=0.4)
    assert clusters[0] == clusters[1]  # identical SMILES -> identical cluster
    assert clusters[2] != clusters[0]


def test_cluster_ligands_by_similarity_empty_input() -> None:
    assert cluster_ligands_by_similarity([], similarity_threshold=0.4) == []


def test_assign_similarity_aware_split_no_shared_protein_across_splits() -> None:
    ids = [f"c{i}" for i in range(20)]
    # complexes 0 and 10 share a protein sequence; everything else unique
    seqs = ["SEQUENCE_A" if i in (0, 10) else f"UNIQUESEQ{i}" for i in range(20)]
    fps = [_fp("CCO") if i % 7 == 0 else _fp(f"{'C' * (i % 5 + 1)}N") for i in range(20)]

    result = assign_similarity_aware_split(ids, seqs, fps, train_frac=0.6, val_frac=0.2, test_frac=0.2)

    assert result.split_by_complex_id["c0"] == result.split_by_complex_id["c10"]
    assert result.group_by_complex_id["c0"] == result.group_by_complex_id["c10"]


def test_assign_similarity_aware_split_no_shared_ligand_across_splits() -> None:
    ids = [f"c{i}" for i in range(15)]
    seqs = [f"UNIQUESEQ{i}" for i in range(15)]
    fps = [_fp("c1ccccc1O") if i in (2, 9) else _fp(f"{'C' * (i % 4 + 1)}Cl") for i in range(15)]

    result = assign_similarity_aware_split(ids, seqs, fps, train_frac=0.6, val_frac=0.2, test_frac=0.2)

    assert result.split_by_complex_id["c2"] == result.split_by_complex_id["c9"]


def test_assign_similarity_aware_split_is_deterministic() -> None:
    ids = [f"c{i}" for i in range(30)]
    seqs = [f"SEQ{i % 8}" for i in range(30)]
    fps = [_fp(f"{'C' * (i % 6 + 1)}O") for i in range(30)]

    result_a = assign_similarity_aware_split(ids, seqs, fps)
    result_b = assign_similarity_aware_split(ids, seqs, fps)

    assert result_a.split_by_complex_id == result_b.split_by_complex_id
    assert result_a.group_by_complex_id == result_b.group_by_complex_id


def test_assign_similarity_aware_split_covers_all_complexes_exactly_once() -> None:
    ids = [f"c{i}" for i in range(25)]
    seqs = [f"SEQ{i % 6}" for i in range(25)]
    fps = [_fp(f"{'C' * (i % 5 + 1)}N") for i in range(25)]

    result = assign_similarity_aware_split(ids, seqs, fps)

    assert set(result.split_by_complex_id.keys()) == set(ids)
    assert all(v in {"train", "val", "test"} for v in result.split_by_complex_id.values())


_DIVERSE_SCAFFOLDS = [
    "CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1", "C1CCCCC1", "CC(C)Cc1ccccc1", "c1ccc2ccccc2c1",
    "CC(=O)Nc1ccccc1", "c1ccsc1", "C1COCCN1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc(cc1)S(=O)(=O)N",
    "C1CCNCC1", "c1cnc2[nH]ccc2c1", "CC1=CC(=O)CC(C)(C)C1", "c1ccc(cc1)Br", "CC(C)(C)OC(=O)N",
    "c1ccc(cc1)C#N", "C1CC1", "c1ccc2[nH]ccc2c1", "CC(=O)N1CCCCC1", "c1ccc(cc1)F",
    "CCN(CC)CC", "c1ccc(cc1)OC", "CC(C)N",
]


def test_assign_similarity_aware_split_roughly_matches_target_proportions() -> None:
    # 25 mutually-dissimilar scaffolds, cycled -> many small groups, so
    # proportions should be able to track targets reasonably closely.
    ids = [f"c{i}" for i in range(100)]
    seqs = [f"SEQ{i}" for i in range(100)]
    fps = [_fp(_DIVERSE_SCAFFOLDS[i % len(_DIVERSE_SCAFFOLDS)]) for i in range(100)]

    result = assign_similarity_aware_split(ids, seqs, fps, train_frac=0.8, val_frac=0.1, test_frac=0.1)
    counts = {"train": 0, "val": 0, "test": 0}
    for split in result.split_by_complex_id.values():
        counts[split] += 1

    assert 60 <= counts["train"] <= 95
    assert counts["val"] >= 1
    assert counts["test"] >= 1
