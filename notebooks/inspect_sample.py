"""Inspection script: build and print one real protein-ligand training sample.

Run with the ``sxt-torch`` environment, from the repo root:

    conda run -n sxt-torch python notebooks/inspect_sample.py

Downloads (and caches under ``data/raw/pdbbind_subset/``, gitignored) the
small public PDB subset defined in ``aidd_generative_docking.data.pdbbind``
the first time it runs, then featurizes and prints every complex.
"""

from __future__ import annotations

import torch

from aidd_generative_docking.data.pdbbind import PDBBindSubset


def describe_sample(subset: PDBBindSubset, complex_id: str) -> None:
    sample = subset.build_sample(complex_id)
    entry = subset.manifest[complex_id]

    print(f"=== {complex_id} ===")
    print(f"description: {entry.description}")
    print(f"ligand resname: {sample.ligand_resname}")
    print()
    print("ligand graph:")
    print(f"  x          : {tuple(sample.ligand.x.shape)}  (element one-hot + charge + aromatic + degree)")
    print(f"  pos        : {tuple(sample.ligand.pos.shape)}  (3D bound-pose coordinates, Angstrom)")
    print(f"  edge_index : {tuple(sample.ligand.edge_index.shape)}")
    print(f"  edge_attr  : {tuple(sample.ligand.edge_attr.shape)}  (bond type one-hot)")
    print(f"  centroid   : {sample.ligand.pos.mean(dim=0).tolist()}")
    print()
    print("pocket graph:")
    print(f"  x          : {tuple(sample.pocket.x.shape)}  (20-way residue one-hot)")
    print(f"  pos        : {tuple(sample.pocket.pos.shape)}  (C-alpha coordinates, Angstrom)")
    print(f"  edge_index : {tuple(sample.pocket.edge_index.shape)}")
    print(f"  num_residues within {subset.pocket_radius_angstrom} A of ligand: {sample.pocket.x.shape[0]}")
    print()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device available for downstream stages: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    subset = PDBBindSubset()
    for complex_id in subset.list_complex_ids():
        describe_sample(subset, complex_id)


if __name__ == "__main__":
    main()
