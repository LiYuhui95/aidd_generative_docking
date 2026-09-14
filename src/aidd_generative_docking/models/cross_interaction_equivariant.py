"""SE(3)-aware rigid-body vector-field network with dynamic cross-interaction
message passing (Milestone 3).

This is an **additive upgrade**, not a replacement, of
``models.rigid_equivariant.RigidBodyVectorField``: that model and its tests
are left untouched, and this module implements a second, separate model so
the two can be trained and compared under the same protocol
(``training/smoke_train_rigid.py``, reused unchanged via its
``model_factory`` hook).

What changes vs. ``RigidBodyVectorField``
------------------------------------------
The Milestone 2 model conditioned each ligand atom on the pocket via a
*dense* all-pairs sum/mean over every pocket residue, using the raw scalar
distance as the only geometric feature. This model instead:

1. **Builds dynamic protein-ligand cross edges**: only ligand-atom/pocket-
   residue pairs within ``cross_edge_cutoff_angstrom`` of the *current*
   pose ``xt`` are connected. Since ``xt`` changes every training step and
   every sampling step, this edge set is rebuilt from scratch on every
   forward call -- as the ligand moves, which residues it "talks to"
   changes too. These cross edges are entirely separate from the ligand's
   own fixed covalent bond graph (``ligand_edge_index``/``edge_attr``),
   which is never touched here and still drives the ligand-ligand message
   passing exactly as in ``RigidBodyVectorField``.
2. **Encodes cross-interaction distances with Gaussian RBFs**
   (``models.rbf.GaussianRBF``) instead of a single raw scalar.
3. Aggregates cross messages with a **sum**, not the dense model's mean:
   with a cutoff, a ligand atom's neighbor count varies (and can be zero,
   e.g. a random starting pose far from the pocket), so a mean would
   divide by zero for isolated atoms; sum is always well-defined and still
   E(3)-equivariant.

The rigid-body reduction at the end -- per-atom force -> mean force
(translation) and mean(r x force) (rotation) -- is unchanged from
``RigidBodyVectorField``; see that module's docstring for why this
pooling is E(3)-equivariant.
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


class CrossInteractionRigidBodyVectorField(nn.Module):
    """Rigid-body vector field with dynamic, RBF-encoded protein-ligand cross edges.

    Args:
        ligand_feature_dim: dimension of ligand atom node features.
        pocket_feature_dim: dimension of pocket residue node features.
        bond_feature_dim: dimension of ligand bond edge features.
        hidden_dim: shared hidden width for all embeddings/MLPs.
        num_layers: number of scalar-only message-passing rounds before the
            final equivariant read-out.
        cross_edge_cutoff_angstrom: distance cutoff (Angstrom) for dynamic
            protein-ligand cross edges. 8 A is a common protein-ligand
            interaction-distance cutoff in the docking/scoring literature
            (e.g. contact/interaction fingerprints), tighter than the 10 A
            pocket-*selection* radius already baked into the preprocessed
            pocket graph (``featurize.featurize_protein_pocket``) -- pocket
            membership stays generous, but which of those residues actively
            exchange messages with the ligand is gated more tightly to
            "currently close." Not tuned for performance.
        num_rbf_centers: number of Gaussian RBF centers spanning
            ``[0, cross_edge_cutoff_angstrom]``. 16 is a modest, standard
            default (see ``models.rbf.GaussianRBF``); not tuned.
    """

    def __init__(
        self,
        ligand_feature_dim: int,
        pocket_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cross_edge_cutoff_angstrom: float = 8.0,
        num_rbf_centers: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cross_edge_cutoff_angstrom = cross_edge_cutoff_angstrom

        self.ligand_embed = nn.Linear(ligand_feature_dim, hidden_dim)
        self.pocket_embed = nn.Linear(pocket_feature_dim, hidden_dim)
        self.time_embed = _mlp(1, hidden_dim, hidden_dim)
        self.rbf = GaussianRBF(num_rbf_centers, cross_edge_cutoff_angstrom)

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

    def build_cross_edges(self, ligand_pos: Tensor, pocket_pos: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build the dynamic ligand<->pocket cross-edge set for the current pose.

        Args:
            ligand_pos: ``(N, 3)`` current ligand atom coordinates.
            pocket_pos: ``(M, 3)`` fixed pocket residue coordinates.

        Returns:
            ``(cross_i, cross_j, cross_diff, cross_dist)``: ligand and
            pocket indices of each edge (each ``(E_cross,)``), the relative
            vector ``ligand_pos[cross_i] - pocket_pos[cross_j]``
            (``(E_cross, 3)``), and its norm (``(E_cross,)``). ``E_cross``
            can be 0 (e.g. a pose far from the pocket).
        """
        dist_matrix = torch.cdist(ligand_pos, pocket_pos)  # (N, M)
        mask = dist_matrix <= self.cross_edge_cutoff_angstrom
        cross_i, cross_j = mask.nonzero(as_tuple=False).t()
        cross_diff = ligand_pos[cross_i] - pocket_pos[cross_j]
        cross_dist = dist_matrix[cross_i, cross_j]
        return cross_i, cross_j, cross_diff, cross_dist

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
        """Predict rigid-body velocity for one complex. Shapes as in
        ``RigidBodyVectorField.forward``; see that docstring for details
        shared with this model.
        """
        n_atoms = ligand_pos.shape[0]

        h_lig = self.ligand_embed(ligand_x) + self.time_embed(t.reshape(1, 1)).squeeze(0)
        h_pocket = self.pocket_embed(pocket_x)  # static context, never updated

        src, dst = ligand_edge_index[0], ligand_edge_index[1]
        bond_diff = ligand_pos[src] - ligand_pos[dst]
        bond_dist = bond_diff.norm(dim=-1, keepdim=True)

        cross_i, cross_j, cross_diff, cross_dist = self.build_cross_edges(ligand_pos, pocket_pos)
        cross_rbf = self.rbf(cross_dist)  # (E_cross, num_rbf_centers)

        for layer in range(self.num_layers):
            bond_msg_in = torch.cat([h_lig[src], h_lig[dst], ligand_edge_attr, bond_dist], dim=-1)
            e_bond = self.bond_scalar_mlps[layer](bond_msg_in)
            agg_bond = torch.zeros(n_atoms, self.hidden_dim, device=h_lig.device, dtype=h_lig.dtype)
            agg_bond.index_add_(0, src, e_bond)

            cross_msg_in = torch.cat([h_lig[cross_i], h_pocket[cross_j], cross_rbf], dim=-1)
            e_cross = self.cross_scalar_mlps[layer](cross_msg_in)  # (E_cross, H)
            agg_cross = torch.zeros(n_atoms, self.hidden_dim, device=h_lig.device, dtype=h_lig.dtype)
            agg_cross.index_add_(0, cross_i, e_cross)

            h_lig = h_lig + self.node_update_mlps[layer](torch.cat([h_lig, agg_bond, agg_cross], dim=-1))

        bond_msg_in = torch.cat([h_lig[src], h_lig[dst], ligand_edge_attr, bond_dist], dim=-1)
        e_bond_ro = self.readout_bond_mlp(bond_msg_in)
        gate_bond = self.readout_bond_gate(e_bond_ro)
        f_bond = torch.zeros(n_atoms, 3, device=h_lig.device, dtype=h_lig.dtype)
        f_bond.index_add_(0, src, gate_bond * bond_diff)

        cross_msg_in = torch.cat([h_lig[cross_i], h_pocket[cross_j], cross_rbf], dim=-1)
        e_cross_ro = self.readout_cross_mlp(cross_msg_in)
        gate_cross = self.readout_cross_gate(e_cross_ro)  # (E_cross, 1)
        f_cross = torch.zeros(n_atoms, 3, device=h_lig.device, dtype=h_lig.dtype)
        f_cross.index_add_(0, cross_i, gate_cross * cross_diff)

        f = f_bond + f_cross

        centroid = ligand_pos.mean(dim=0, keepdim=True)
        r = ligand_pos - centroid

        v_trans = f.mean(dim=0)
        v_rot = torch.cross(r, f, dim=-1).mean(dim=0)

        return v_trans, v_rot


def predict_batch(
    model: CrossInteractionRigidBodyVectorField,
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
