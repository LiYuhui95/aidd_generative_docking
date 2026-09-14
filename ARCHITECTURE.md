# Architecture

## Scientific question

Given a protein binding pocket and a ligand's molecular graph, can a
conditional generative model produce a 3D ligand binding pose close to the
experimentally observed pose? This project studies a deliberately narrowed
version of that question: **can rigid-body conditional Flow Matching**
(translate + rotate a ligand of fixed shape, no internal deformation)
**move a randomly placed ligand toward its true bound pose**, using an
E(3)-aware geometric neural network conditioned on the protein pocket. It
does not attempt full flexible docking, binding affinity prediction, or
virtual screening (see Limitations and Future Work below).

## Input / output

- **Input:** a fixed protein pocket (a residue-level graph, Cα coordinates)
  and a ligand (an RDKit-derived atom/bond graph with its known internal
  3D geometry), placed at a random rigid-body pose.
- **Output:** a rigid-body transform (rotation + translation) applied to
  the ligand's fixed shape, producing predicted 3D atom coordinates for
  the bound pose. Internal ligand geometry (bond lengths/angles, all
  pairwise atom distances) is preserved exactly by construction.

The protein pocket is always fixed context; ligand internal geometry is
also fixed (v1 assumption, see Limitations). Only the ligand's rigid-body
placement is generated.

## Protein representation

The protein is reduced to a **pocket graph**: standard-amino-acid residues
within `pocket_radius_angstrom` (10 Å) of any ligand atom are kept as
nodes, one node per residue via its Cα coordinate. Node features are a
20-way one-hot residue identity. Edges connect residue pairs within
`edge_cutoff_angstrom` (8 Å) of each other (a fixed radius graph, computed
once at preprocessing time — the pocket does not move or update during
generation). See `data/featurize.py::featurize_protein_pocket` for the
full cutoff rationale.

## Ligand representation

The ligand is a **molecular graph** parsed with RDKit directly from
PDBBind's own `<id>_ligand.sdf` file (MOL2 fallback on parse failure):
atoms are nodes (element one-hot, formal charge, aromaticity, degree),
covalent bonds are edges (bond-type one-hot: single/double/triple/
aromatic), and atom positions are the true experimental bound-pose
coordinates, in the same reference frame as the pocket. See
`data/featurize.py` and `data/dataset_builder.py`.

## Flow Matching objective: rigid-body SE(3) path

A ligand pose is parameterized as `x = R @ L + t`, where `L` is the
ligand's canonical shape (its bound pose recentered to zero centroid, so
by construction the *target* corresponds to `R1 = I`, `t1 = centroid(x1)`),
`R` a 3×3 rotation matrix, `t` a translation. This replaces the earlier,
simpler per-atom linear-interpolation sketch (`diffusion/flow_matching.py`,
kept as historical scaffolding, not used by the trained models below) once
the project moved to a genuinely rigid-body formulation:

- **Translation** is linearly interpolated (`tt = (1-t)·t0 + t·t1`);
  constant velocity `v_trans = t1 - t0` — flat Euclidean space, nothing
  SE(3)-specific needed.
- **Rotation** follows the SO(3) *geodesic* `Rt = expm(t·Ω̂) @ R0`, where
  `Ω` is the constant world-frame angular velocity with
  `expm(Ω̂) @ R0 = R1`. This is the direct SO(3) analogue of "linear path ⇒
  constant velocity," and `xt = Rt @ L + tt` is an exact rigid transform of
  `L` at *every* `t` — ligand internal distances are preserved exactly
  throughout training and sampling, not approximately (verified in tests
  to float64 machine precision).

See `diffusion/rigid_flow_matching.py` for the full implementation
(`build_training_pair`, `interpolate_rigid_pose`, `integrate_rigid_pose`).

### Model: E(3)-aware rigid-body vector field

Three model variants were built and compared (all in `models/`), each
predicting a single global `(v_trans, v_rot)` pair per complex (not a
per-atom field) via a shared reduction:

```
per-atom "force" f_i  -->  v_trans = mean_i(f_i)          (net force)
                      -->  v_rot   = mean_i(r_i × f_i)     (net torque about centroid)
```

The cross product makes `v_rot` a genuine SO(3) pseudovector (equivariant
under proper rotations); `v_trans` is built purely from relative vectors,
so it is translation-invariant and rotation-equivariant — exactly the
properties required of a Flow Matching velocity field on rigid poses.

1. **`RigidBodyVectorField`** (Milestone 2): dense pocket conditioning —
   every ligand atom exchanges messages with every pocket residue, using
   raw scalar distance.
2. **`CrossInteractionRigidBodyVectorField`** (Milestone 3): dynamic
   protein-ligand cross *edges*, rebuilt every forward call from the
   current pose, connecting only atom/residue pairs within a hard distance
   cutoff (8 Å), with distances encoded via Gaussian RBFs
   (`models/rbf.py`) instead of a raw scalar.
3. **`SoftCrossInteractionRigidBodyVectorField`** (Milestone 3 ablation,
   the model used for the final held-out experiment): same RBF-based
   pathway, but connectivity is dense (no hard mask) and every pair is
   weighted by a smooth, strictly-positive envelope
   `w(d) = 1/(1+(d/length_scale)²)`. An ablation directly confirmed the
   hypothesis motivating this variant: model (2)'s hard cutoff creates a
   genuine gradient **dead zone** when a randomized starting pose places
   every ligand atom beyond the cutoff of every pocket residue (verified
   in tests: with identical weights, model (2)'s output becomes
   completely insensitive to distance beyond the cutoff, while model (3)'s
   does not) — and under an otherwise identical training protocol, model
   (3) reached a training loss of 11.4 vs. model (2)'s 38.5 (see
   Experiments table).

Ligand-ligand covalent bonds are always handled by a separate, fixed
message-passing branch (never confused with the dynamic protein-ligand
cross terms) using the same scalar-gated-relative-vector construction.

## Sampling

Starting from a random rigid pose (`R0` uniform over SO(3) via
`scipy.spatial.transform.Rotation.random`, `t0` = pocket centroid +
Gaussian noise), the learned vector field is Euler-integrated over 50
steps from flow time 0 to 1, reconstructing the rigid pose from
`(R, t)` at every step (`diffusion/rigid_flow_matching.py::
integrate_rigid_pose`) — this is what guarantees the sampled trajectory,
not just its endpoints, stays exactly rigid.

## Evaluation metrics

- **RMSD** between generated and crystallographic ligand pose (atoms
  already correspondence-matched and co-frame with the pocket, so no
  additional alignment step; `evaluation/metrics.py::rmsd`).
- **Pose success rate** at a given RMSD threshold
  (`evaluation/metrics.py::pose_success_rate`); this project reports both
  the strict 2 Å docking-literature convention and a looser 5 Å threshold.
- Training/validation loss curves and checkpoint selection by validation
  loss (held-out experiment only; the smoke experiments have no held-out
  split at all).

## Dataset and split (final held-out experiment)

Built from the user's own licensed PDBBind v2020.R1 download
(`data/raw/P-L.tar.gz`, 19,037 complexes + `index.tar.gz`; see
`data/dataset_builder.py`, `data/splitting.py`):

1. A candidate pool (1,200 complexes) is drawn deterministically (seeded
   shuffle) from the full index, excluding index-flagged covalent
   complexes (this project's rigid, non-covalent binding assumption does
   not apply to them).
2. Every candidate is independently validated: ligand/protein parsed,
   featurized, and checked against a ligand-size cap (100 atoms
   including explicit hydrogens, to exclude large peptide-like outliers
   and keep training compute manageable); failures are recorded with a
   specific, documented reason rather than silently dropped.
3. The dataset is capped at 800 complexes (all failures in practice were
   the size cap — see the Experiments table for the exact count).
4. **Similarity-aware split** (`data/splitting.py`): proteins are grouped
   by exact sequence match (catches "same target, many co-crystal
   structures" — the dominant PDBBind redundancy pattern); ligands are
   grouped by RDKit Butina clustering on Morgan fingerprints (Tanimoto
   ≥ 0.4). Any two complexes sharing a protein group *or* a ligand group
   are merged (union-find) into one leakage group, and whole groups —
   never individual complexes — are assigned to train/val/test via a
   deterministic (RNG-free), largest-group-first, proportional bin-packing.
   This is a real but intentionally modest leakage-control measure — a
   cheaper proxy for full-scale published protocols (e.g. LP-PDBBind's
   sequence-alignment + full scaffold clustering), not a reproduction of
   one; see Limitations.

## Experiments

| | Milestones 2/3 (smoke tests) | Final held-out experiment |
|---|---|---|
| Purpose | pipeline sanity check / overfitting | **generalization** |
| Data | 10 hand-picked complexes | 800 complexes (640/80/80 train/val/test) |
| Train/val/test | **none** — same 10 complexes trained and "evaluated" | disjoint by construction (similarity-aware) |
| Epochs | 3,000 (full-batch) | 150 (mini-batch, size 16) |
| Model | `RigidBodyVectorField` / `CrossInteractionRigidBodyVectorField` / `SoftCrossInteractionRigidBodyVectorField` | `SoftCrossInteractionRigidBodyVectorField` (unchanged) |
| Checkpoint selection | n/a (report final epoch) | lowest validation loss (epoch 104/150) |
| Test-set touches | n/a | **exactly once**, after checkpoint selection |
| Final training loss | 4.58 (dense) / 38.5 (hard cutoff) / 11.4 (soft, same epoch budget) | 13.70 (train), 12.50 (best val) |
| Mean RMSD, initial→final | 10.67→5.77 Å (dense, best of the three) | 10.29→7.14 Å |
| Median RMSD, initial→final | 11.22→3.51 Å (dense) | 10.24→6.75 Å |
| Success @ 2 Å | 20% (dense, 10/10 complexes) | 0% (80 held-out complexes) |
| Success @ 5 Å | not reported at this scale | 25% |

The gap between the smoke tests (a handful of complexes, thousands of
epochs — expected to memorize) and the held-out experiment (hundreds of
never-seen complexes, 150 epochs) is exactly the expected direction and
magnitude for a small, non-flexible, briefly-trained rigid-body model: it
learns a real, measurable pull toward the pocket (confirmed by the
initial→final RMSD reduction and the fact smaller ligands generalize
better — see `outputs/held_out/held_out_results.json`'s ligand-size
terciles), but is far from precise pose prediction. **No benchmark claims
are made anywhere in this project.**

## Limitations

- **Rigid ligand, rigid pocket.** No torsional degrees of freedom and no
  protein flexibility; real binding often involves both.
- **Small model, brief training.** `hidden_dim=64`, 2 message-passing
  layers, 150 epochs over 640 training complexes — orders of magnitude
  below typical published docking models' scale.
- **Leakage control is a proxy, not a full protocol.** Exact-sequence
  protein grouping catches the dominant redundancy pattern but not
  partial-identity homologs; ligand grouping uses one fingerprint/
  threshold choice. A published, full-scale leak-proof split (e.g.
  LP-PDBBind) was not reproduced here.
- **No affinity signal used anywhere.** Training and evaluation are
  purely structural (pose RMSD); binding affinity values in the PDBBind
  index are not used.
- **Single train/eval configuration.** Hyperparameters were chosen once
  from a compute-time calibration, not tuned against validation or test
  performance, and are not claimed to be optimal.
- **Held-out generalization is measured, not "solved."** 0% success at
  the strict 2 Å threshold on the test set indicates this v1 pipeline is
  a working proof of concept, not a competitive pose predictor.

## Future work (not implemented in this project)

- **Ligand torsional flexibility**: allow internal conformational change
  (rotatable-bond torsion angles) instead of a single rigid transform.
- **Protein-pocket flexibility**: allow side-chain (or backbone) motion
  instead of treating the pocket as entirely fixed.
- **Affinity/ranking heads**: a model head predicting binding affinity or
  ranking candidate poses/ligands, using the affinity labels already
  present in the PDBBind index (unused so far).
- **Larger-scale virtual screening**: applying a trained model to rank or
  filter large candidate ligand libraries against a target pocket.
