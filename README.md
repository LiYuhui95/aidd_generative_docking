# aidd_generative_docking

Research-grade, Windows-native prototype for **protein-ligand generative
docking**: given a fixed protein pocket and a ligand's molecular graph,
generate 3D binding poses with a conditional **Flow Matching** model built
on equivariant geometric deep learning, then evaluate poses against
crystallographic ground truth.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the scientific design (input/
output, protein/ligand representation, the Flow Matching objective, and
evaluation metrics).

## Status

Early scaffolding. Package structure, config, and interfaces are in place;
the equivariant vector-field network, data pipeline, and training loop are
not yet implemented (see stage list below and `ARCHITECTURE.md`). No data
has been downloaded and no model has been trained.

## Project stages

1. Small PDBBind subset + preprocessing
2. Protein / ligand geometric representation (PyTorch Geometric graphs)
3. Minimal equivariant Flow Matching model
4. Ligand pose generation (vector-field integration)
5. RMSD and pose-success evaluation
6. Later: larger dataset, binding affinity prediction, virtual screening

## Environment

- **Platform:** Windows-native. No Linux/WSL dependency is introduced by
  this project; any dependency that requires Linux-only kernels is out of
  scope.
- **Conda env:** uses the existing `sxt-torch` environment
  (PyTorch + CUDA + PyTorch Geometric already present). No new environment
  is created and no packages are installed by this commit.
- **Compute:** GPU-first (CUDA), with CPU as a fallback for a device that
  lacks a GPU.
- **Core stack:** PyTorch, PyTorch Geometric, RDKit (RDKit is declared as a
  dependency but not yet installed in `sxt-torch`; ligand-parsing code that
  needs it imports it lazily so the rest of the package stays importable).

## Repository layout

```
configs/                      experiment configs (YAML)
src/aidd_generative_docking/
    data/                     dataset curation + preprocessing (stage 1-2)
    models/                   equivariant encoders / vector-field network (stage 3)
    diffusion/                Flow Matching path + objective (stage 3-4)
    training/                 training loop entrypoint (stage 3)
    evaluation/               RMSD / pose-success metrics (stage 5)
tests/                        placeholder import + unit tests
notebooks/                    exploratory notebooks
```

## Running tests

From the repo root, using the `sxt-torch` environment's Python directly
(no install step required — `pyproject.toml` sets `pythonpath = ["src"]`
for pytest):

```
conda run -n sxt-torch pytest
```

## Non-goals right now

This commit intentionally does **not**: download PDBBind or any dataset,
download model checkpoints, install new packages, or implement/train the
full model. It only establishes the project skeleton and interfaces.
