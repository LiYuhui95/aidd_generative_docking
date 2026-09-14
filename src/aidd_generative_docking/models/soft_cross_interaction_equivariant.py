"""Soft distance-weighted cross-interaction rigid-body vector field.

Milestone 3 ablation, additive only: does not modify
``CrossInteractionRigidBodyVectorField`` (hard cutoff),
``RigidBodyVectorField`` (dense), or any preprocessing/diffusion code.

Hypothesis under test
----------------------
``CrossInteractionRigidBodyVectorField`` connects a ligand atom to a
pocket residue only if their current distance is within
``cross_edge_cutoff_angstrom``. If a randomized starting pose places
*every* ligand atom beyond that cutoff of *every* pocket residue, the
cross-edge set is empty: the cross-interaction term contributes exactly
zero force and exactly zero gradient that step, regardless of exactly how
far away the ligand is. This is a "dead zone" -- training gets no signal
at all telling it which direction the pocket is.

What changes here
-------------------
Same message-passing pathway as the hard-cutoff model (RBF-encoded
distance -> MLP -> scalar gate on the equivariant relative vector ->
aggregated into per-atom force), with exactly one change:

1. **No hard connectivity mask.** Every ligand atom is connected to every
   pocket residue (dense, as in the original Milestone 2 model) -- there
   is no distance at which a pair is dropped entirely.
2. **A smooth, strictly-positive, full-support distance envelope**

       w(d) = 1 / (1 + (d / length_scale)^2)

   multiplies every cross message and force contribution. ``w(0) = 1``,
   ``w(length_scale) = 0.5``, and ``w`` decreases monotonically toward
   (but never reaches) 0 as ``d -> infinity``. However far the ligand
   starts, the cross-interaction term is always nonzero and always
   (slightly) distance-dependent -- there is no radius beyond which it
   goes to exactly zero, so there is no dead zone.

``length_scale_angstrom`` plays the same role ``cross_edge_cutoff_angstrom``
played in the hard model (it is reused, unchanged, for both the RBF's own
``cutoff`` and the envelope's decay length), so the module construction
here mirrors ``CrossInteractionRigidBodyVectorField`` layer-for-layer,
parameter-for-parameter -- the only difference is masking vs. weighting.

Equivariance: ``w(d)`` is a function of an invariant scalar (distance), so
multiplying it onto the same scalar-gated relative-vector construction
already used (and proven equivariant) in
``CrossInteractionRigidBodyVectorField`` does not change the equivariance
of the result. The final rigid-body reduction (mean force -> translation,
mean r x force -> rotation) is copied unchanged from that model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from aidd_generative_docking.models.rbf import GaussianRBF


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def smooth_distance_envelope(distance: Tensor, length_scale: float) -> Tensor:
    """Smooth, strictly-positive, full-support distance decay weight.

    ``w(d) = 1 / (1 + (d / length_scale)^2)``

    Unlike a hard cutoff (or even a smoothly-vanishing compact-support
    cutoff such as a cosine window), this has no finite radius beyond
    which the weight -- and hence the gradient -- becomes exactly zero;
    it only decays asymptotically.
    """
    return 1.0 / (1.0 + (distance / length_scale) ** 2)


class SoftCrossInteractionRigidBodyVectorField(nn.Module):
    """Rigid-body vector field with dense, smoothly distance-weighted cross interactions.

    Args:
        ligand_feature_dim: dimension of ligand atom node features.
        pocket_feature_dim: dimension of pocket residue node features.
        bond_feature_dim: dimension of ligand bond edge features.
        hidden_dim: shared hidden width for all embeddings/MLPs.
        num_layers: number of scalar-only message-passing rounds before the
            final equivariant read-out.
        length_scale_angstrom: characteristic decay distance for the smooth
            envelope, and the ``cutoff`` used to place the RBF centers.
            Reused at the same value (8 A) as the hard model's
            ``cross_edge_cutoff_angstrom`` for a controlled ablation; not
            tuned for performance.
        num_rbf_centers: number of Gaussian RBF centers (see
            ``models.rbf.GaussianRBF``). Same default as the hard model.
    """

    def __init__(
        self,
        ligand_feature_dim: int,
        pocket_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        length_scale_angstrom: float = 8.0,
        num_rbf_centers: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.length_scale_angstrom = length_scale_angstrom

        self.ligand_embed = nn.Linear(ligand_feature_dim, hidden_dim)
        self.pocket_embed = nn.Linear(pocket_feature_dim, hidden_dim)
        self.time_embed = _mlp(1, hidden_dim, hidden_dim)
        self.rbf = GaussianRBF(num_rbf_centers, length_scale_angstrom)

        bond_msg_in_dim = 2 * hidden_dim + bond_feature_dim + 1
        cross_msg_in_dim = 2 * hidden_dim + num_rbf_centers

        self.bond_scalar_mlps = nn.ModuleList([_mlp(bond_msg_in_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.cross_scalar_mlps = nn.ModuleList(
            [_mlp(cross_msg_in_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.node_update_mlps = nn.ModuleList(
            [_mlp(3 * hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.readout_bond_mlp = _mlp(bond_msg_in_dim, hidden_dim, hidden_dim)
        self.readout_bond_gate = nn.Linear(hidden_dim, 1)
        self.readout_cross_mlp = _mlp(cross_msg_in_dim, hidden_dim, hidden_dim)
        self.readout_cross_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        ligand_x: Tensor,
        ligand_edge_index: Tensor,
        ligand_edge_attr: Tensor,
        ligand_pos: Tensor,
        pocket_x: Tensor,
        pocket_pos: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict rigid-body velocity for one complex. Same call signature
        as ``CrossInteractionRigidBodyVectorField.forward``."""
        n_atoms, n_pocket = ligand_pos.shape[0], pocket_pos.shape[0]

        h_lig = self.ligand_embed(ligand_x) + self.time_embed(t.reshape(1, 1)).squeeze(0)
        h_pocket = self.pocket_embed(pocket_x)  # static context, never updated

        src, dst = ligand_edge_index[0], ligand_edge_index[1]
        bond_diff = ligand_pos[src] - ligand_pos[dst]
        bond_dist = bond_diff.norm(dim=-1, keepdim=True)

        # Dense: every ligand atom <-> every pocket residue, no hard mask.
        cross_diff = ligand_pos.unsqueeze(1) - pocket_pos.unsqueeze(0)  # (N, M, 3)
        cross_dist = cross_diff.norm(dim=-1)  # (N, M)
        cross_rbf = self.rbf(cross_dist)  # (N, M, K)
        envelope = smooth_distance_envelope(cross_dist, self.length_scale_angstrom).unsqueeze(-1)  # (N, M, 1)
        h_pocket_expanded = h_pocket.unsqueeze(0).expand(n_atoms, -1, -1)

        for layer in range(self.num_layers):
            bond_msg_in = torch.cat([h_lig[src], h_lig[dst], ligand_edge_attr, bond_dist], dim=-1)
            e_bond = self.bond_scalar_mlps[layer](bond_msg_in)
            agg_bond = torch.zeros(n_atoms, self.hidden_dim, device=h_lig.device, dtype=h_lig.dtype)
            agg_bond.index_add_(0, src, e_bond)

            cross_msg_in = torch.cat([h_lig.unsqueeze(1).expand(-1, n_pocket, -1), h_pocket_expanded, cross_rbf], dim=-1)
            e_cross = self.cross_scalar_mlps[layer](cross_msg_in)  # (N, M, H)
            agg_cross = (e_cross * envelope).sum(dim=1)  # (N, H); envelope-weighted, always well-defined

            h_lig = h_lig + self.node_update_mlps[layer](torch.cat([h_lig, agg_bond, agg_cross], dim=-1))

        bond_msg_in = torch.cat([h_lig[src], h_lig[dst], ligand_edge_attr, bond_dist], dim=-1)
        e_bond_ro = self.readout_bond_mlp(bond_msg_in)
        gate_bond = self.readout_bond_gate(e_bond_ro)
        f_bond = torch.zeros(n_atoms, 3, device=h_lig.device, dtype=h_lig.dtype)
        f_bond.index_add_(0, src, gate_bond * bond_diff)

        cross_msg_in = torch.cat([h_lig.unsqueeze(1).expand(-1, n_pocket, -1), h_pocket_expanded, cross_rbf], dim=-1)
        e_cross_ro = self.readout_cross_mlp(cross_msg_in)
        gate_cross = self.readout_cross_gate(e_cross_ro)  # (N, M, 1)
        f_cross = (gate_cross * envelope * cross_diff).sum(dim=1)  # (N, 3)

        f = f_bond + f_cross

        centroid = ligand_pos.mean(dim=0, keepdim=True)
        r = ligand_pos - centroid

        v_trans = f.mean(dim=0)
        v_rot = torch.cross(r, f, dim=-1).mean(dim=0)

        return v_trans, v_rot


def predict_batch(
    model: SoftCrossInteractionRigidBodyVectorField,
    ligand_x_list: list[Tensor],
    ligand_edge_index_list: list[Tensor],
    ligand_edge_attr_list: list[Tensor],
    ligand_pos_list: list[Tensor],
    pocket_x_list: list[Tensor],
    pocket_pos_list: list[Tensor],
    t_list: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Loop ``model`` over a list of complexes and stack outputs. See
    ``models.rigid_equivariant.predict_batch`` for the batching rationale.
    """
    v_trans_out, v_rot_out = [], []
    for args in zip(
        ligand_x_list, ligand_edge_index_list, ligand_edge_attr_list, ligand_pos_list, pocket_x_list, pocket_pos_list, t_list
    ):
        v_trans, v_rot = model(*args)
        v_trans_out.append(v_trans)
        v_rot_out.append(v_rot)
    return torch.stack(v_trans_out, dim=0), torch.stack(v_rot_out, dim=0)
