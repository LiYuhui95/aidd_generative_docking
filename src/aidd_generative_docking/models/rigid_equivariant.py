"""Minimal SE(3)-aware rigid-body vector-field network (Milestone 2).

Unlike a typical per-atom EGNN (which outputs one vector per atom),
``RigidBodyVectorField`` outputs a single global velocity pair per complex:
a translation velocity ``v_trans`` (R^3) and an angular velocity ``v_rot``
(R^3, axis-angle), matching the rigid-body Flow Matching path in
``diffusion/rigid_flow_matching.py``.

Design (plain torch, no e3nn dependency -- "minimal and inspectable"):

1. Embed ligand atom features and pocket residue features into a shared
   hidden dimension. The pocket embedding is computed once and never
   updated by message passing: per the v1 assumption that the pocket is
   fixed, it is pure conditioning context, not something the model needs to
   reason about jointly with itself.
2. For ``num_layers`` rounds, refine ligand atom scalar features via
   message passing over (a) the ligand's own bond graph and (b) dense
   ligand<->pocket cross edges, using only *invariant* inputs (pairwise
   distances, bond/residue features) -- so the resulting scalar embeddings
   are rotation- and translation-invariant.
3. A final equivariant read-out layer computes a per-atom "force" vector
   ``f_i`` as a learned scalar gate times the raw relative-position vector
   (the standard EGNN trick: scalar-gated relative vectors are automatically
   E(3)-equivariant, since the gate depends only on invariant features).
4. Pool per-atom forces into the two rigid-body outputs, by analogy with
   classical rigid-body mechanics:
   - ``v_trans = mean_i(f_i)``       (net force -> translation of the whole body)
   - ``v_rot   = mean_i(r_i x f_i)`` (net torque about the current centroid,
     ``r_i`` = atom position relative to centroid -> angular velocity)
   The cross product makes ``v_rot`` a genuine SO(3) pseudovector: for any
   proper rotation ``Q`` (``det(Q) = 1``), ``(Q r_i) x (Q f_i) = Q (r_i x f_i)``,
   so ``v_rot`` rotates correctly with the input. ``v_trans`` is built from
   sums of relative vectors only, so it is translation-invariant and
   rotation-equivariant, exactly the properties required of a Flow Matching
   velocity field on rigid poses.

Batching: this module's ``forward`` handles **one complex at a time**
(clearer to write and test for a v1 model). ``predict_batch`` below loops
over a Python list of complexes -- deliberately not a fused batched-graph
op, since batch sizes in this milestone are tiny (a handful of complexes,
tens of atoms each) and a plain loop is far easier to verify correct than
custom bipartite batched message passing. Revisit if this ever needs to
scale to large batches.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class RigidBodyVectorField(nn.Module):
    """Predicts a global rigid-body velocity ``(v_trans, v_rot)`` for one ligand pose.

    Args:
        ligand_feature_dim: dimension of ligand atom node features
            (``ComplexSample.ligand.x``).
        pocket_feature_dim: dimension of pocket residue node features
            (``ComplexSample.pocket.x``).
        bond_feature_dim: dimension of ligand bond edge features
            (``ComplexSample.ligand.edge_attr``).
        hidden_dim: shared hidden width for all embeddings/MLPs.
        num_layers: number of scalar-only message-passing rounds before the
            final equivariant read-out.
    """

    def __init__(
        self,
        ligand_feature_dim: int,
        pocket_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.ligand_embed = nn.Linear(ligand_feature_dim, hidden_dim)
        self.pocket_embed = nn.Linear(pocket_feature_dim, hidden_dim)
        self.time_embed = _mlp(1, hidden_dim, hidden_dim)

        bond_msg_in_dim = 2 * hidden_dim + bond_feature_dim + 1
        pocket_msg_in_dim = 2 * hidden_dim + 1

        self.bond_scalar_mlps = nn.ModuleList([_mlp(bond_msg_in_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.pocket_scalar_mlps = nn.ModuleList(
            [_mlp(pocket_msg_in_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.node_update_mlps = nn.ModuleList(
            [_mlp(3 * hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.readout_bond_mlp = _mlp(bond_msg_in_dim, hidden_dim, hidden_dim)
        self.readout_bond_gate = nn.Linear(hidden_dim, 1)
        self.readout_pocket_mlp = _mlp(pocket_msg_in_dim, hidden_dim, hidden_dim)
        self.readout_pocket_gate = nn.Linear(hidden_dim, 1)

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
        """Predict rigid-body velocity for one complex.

        Args:
            ligand_x: ``(N, ligand_feature_dim)`` ligand atom features.
            ligand_edge_index: ``(2, E)`` ligand bond graph (both directions).
            ligand_edge_attr: ``(E, bond_feature_dim)`` bond one-hot features.
            ligand_pos: ``(N, 3)`` current ligand pose ``xt``.
            pocket_x: ``(M, pocket_feature_dim)`` pocket residue features.
            pocket_pos: ``(M, 3)`` pocket residue Cα coordinates (fixed).
            t: scalar flow time, shape ``(1,)`` or ``()``.

        Returns:
            ``(v_trans, v_rot)``: two ``(3,)`` tensors.
        """
        h_lig = self.ligand_embed(ligand_x) + self.time_embed(t.reshape(1, 1)).squeeze(0)
        h_pocket = self.pocket_embed(pocket_x)  # static context, never updated

        src, dst = ligand_edge_index[0], ligand_edge_index[1]
        bond_diff = ligand_pos[src] - ligand_pos[dst]  # (E, 3)
        bond_dist = bond_diff.norm(dim=-1, keepdim=True)  # (E, 1)

        cross_diff = ligand_pos.unsqueeze(1) - pocket_pos.unsqueeze(0)  # (N, M, 3)
        cross_dist = cross_diff.norm(dim=-1, keepdim=True)  # (N, M, 1)

        n_atoms, n_pocket = ligand_pos.shape[0], pocket_pos.shape[0]
        h_pocket_expanded = h_pocket.unsqueeze(0).expand(n_atoms, -1, -1)  # (N, M, H)

        for layer in range(self.num_layers):
            bond_msg_in = torch.cat([h_lig[src], h_lig[dst], ligand_edge_attr, bond_dist], dim=-1)
            e_bond = self.bond_scalar_mlps[layer](bond_msg_in)  # (E, H)
            agg_bond = torch.zeros(n_atoms, self.hidden_dim, device=h_lig.device, dtype=h_lig.dtype)
            agg_bond.index_add_(0, src, e_bond)

            pocket_msg_in = torch.cat([h_lig.unsqueeze(1).expand(-1, n_pocket, -1), h_pocket_expanded, cross_dist], dim=-1)
            e_pocket = self.pocket_scalar_mlps[layer](pocket_msg_in)  # (N, M, H)
            agg_pocket = e_pocket.mean(dim=1)  # (N, H)

            h_lig = h_lig + self.node_update_mlps[layer](torch.cat([h_lig, agg_bond, agg_pocket], dim=-1))

        bond_msg_in = torch.cat([h_lig[src], h_lig[dst], ligand_edge_attr, bond_dist], dim=-1)
        e_bond_ro = self.readout_bond_mlp(bond_msg_in)
        gate_bond = self.readout_bond_gate(e_bond_ro)  # (E, 1)
        f_bond = torch.zeros(n_atoms, 3, device=h_lig.device, dtype=h_lig.dtype)
        f_bond.index_add_(0, src, gate_bond * bond_diff)

        pocket_msg_in = torch.cat([h_lig.unsqueeze(1).expand(-1, n_pocket, -1), h_pocket_expanded, cross_dist], dim=-1)
        e_pocket_ro = self.readout_pocket_mlp(pocket_msg_in)  # (N, M, H)
        gate_pocket = self.readout_pocket_gate(e_pocket_ro)  # (N, M, 1)
        f_pocket = (gate_pocket * cross_diff).mean(dim=1)  # (N, 3)

        f = f_bond + f_pocket  # (N, 3) per-atom "force"

        centroid = ligand_pos.mean(dim=0, keepdim=True)
        r = ligand_pos - centroid

        v_trans = f.mean(dim=0)
        v_rot = torch.cross(r, f, dim=-1).mean(dim=0)

        return v_trans, v_rot


def predict_batch(
    model: RigidBodyVectorField,
    ligand_x_list: list[Tensor],
    ligand_edge_index_list: list[Tensor],
    ligand_edge_attr_list: list[Tensor],
    ligand_pos_list: list[Tensor],
    pocket_x_list: list[Tensor],
    pocket_pos_list: list[Tensor],
    t_list: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Run ``model`` over a Python list of complexes and stack the outputs.

    See module docstring for why batching is a plain loop rather than a
    fused batched-graph operation at this milestone's scale.

    Returns:
        ``(v_trans, v_rot)``, each ``(B, 3)``.
    """
    v_trans_out, v_rot_out = [], []
    for args in zip(
        ligand_x_list, ligand_edge_index_list, ligand_edge_attr_list, ligand_pos_list, pocket_x_list, pocket_pos_list, t_list
    ):
        v_trans, v_rot = model(*args)
        v_trans_out.append(v_trans)
        v_rot_out.append(v_rot)
    return torch.stack(v_trans_out, dim=0), torch.stack(v_rot_out, dim=0)
