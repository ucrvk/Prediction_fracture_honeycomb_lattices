"""Direction-aware residual GCN implemented directly from the node operator.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SHELLNET = Path(__file__).resolve().parents[1] / "shellnet"
if str(SHELLNET) not in sys.path:
    sys.path.append(str(SHELLNET))

from shells import SHELL_R, SHELL_TAPCOUNT, SHELL_TAPS  # noqa: E402


# Each real node receives four sublattice-specific features and one shared
# cell-survival feature, exactly as in the original data pipeline.
CH_T, CH_B, CH_SHARED = [0, 2, 4, 7], [1, 3, 5, 8], [6]
DEFAULT_SCHEDULE = (1, 1, 2, 3, 4, 5, 6, 2, 6)
OPS = ("directional", "iso", "gcn")


def node_id(s: int, j: int, i: int, height: int, width: int) -> int:
    """Flatten ``(sublattice, row, column)`` with all T nodes before B nodes."""

    return s * height * width + j * width + i


def build_direction_edges(
    height: int, width: int, m: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build directed ``source -> target`` edges for distance order ``m``.

    The edge type is the direction index used by ``SHELL_TAPS``.  Its ordering
    is defined geometrically and is shared by the T and B sublattices.
    """

    if m not in SHELL_R:
        raise ValueError(f"unsupported distance order m={m}; choose {sorted(SHELL_R)}")

    sources, targets, edge_types = [], [], []
    for s in (0, 1):
        for j in range(height):
            parity = (j - 1) % 2
            for i in range(width):
                target = node_id(s, j, i, height, width)
                for t, (s2, dj, di) in enumerate(SHELL_TAPS[(s, parity, m)]):
                    j2, i2 = j + dj, i + di
                    if 0 <= j2 < height and 0 <= i2 < width:
                        sources.append(node_id(s2, j2, i2, height, width))
                        targets.append(target)
                        edge_types.append(t)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    return edge_index, edge_type


def rot180_node_permutation(height: int, width: int) -> torch.Tensor:
    """Old node id -> node id after 180-degree rotation and T/B exchange."""

    perm = torch.empty(2 * height * width, dtype=torch.long)
    for s in (0, 1):
        for j in range(height):
            for i in range(width):
                old = node_id(s, j, i, height, width)
                new = node_id(1 - s, height - 1 - j, width - 1 - i, height, width)
                perm[old] = new
    return perm


def rot180_edge_permutation(height: int, width: int, m: int = 1) -> torch.Tensor:
    """Old edge id -> edge id under the data augmentation symmetry."""

    edge_index, edge_type = build_direction_edges(height, width, m)
    lookup = {
        (int(src), int(dst)): k
        for k, (src, dst) in enumerate(edge_index.t().tolist())
    }
    node_perm = rot180_node_permutation(height, width)
    edge_perm = torch.empty(edge_index.shape[1], dtype=torch.long)
    for k, (src, dst) in enumerate(edge_index.t().tolist()):
        mapped = (int(node_perm[src]), int(node_perm[dst]))
        if mapped not in lookup:
            raise RuntimeError(f"rotated edge is absent from the template: {mapped}")
        kk = lookup[mapped]
        if int(edge_type[kk]) != int(edge_type[k]):
            raise RuntimeError("direction labels are not preserved by 180-degree rotation")
        edge_perm[k] = kk
    return edge_perm


def augment(
    xb: torch.Tensor,
    yb: torch.Tensor,
    mb: torch.Tensor,
    edge_mask_m1: torch.Tensor,
    edge_perm_m1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same 180-degree augmentation to fields and first-distance edges."""

    xb = torch.flip(xb, dims=(-2, -1))
    xb = torch.cat(
        [
            xb[:, [1, 0]],
            xb[:, [3, 2]],
            xb[:, [5, 4]],
            xb[:, 6:7],
            xb[:, [8, 7]],
        ],
        dim=1,
    )
    yb = torch.flip(yb, dims=(-2, -1)).flip(1)
    mb = torch.flip(mb, dims=(-2, -1)).flip(1)
    rotated_edges = torch.zeros_like(edge_mask_m1)
    rotated_edges[:, edge_perm_m1] = edge_mask_m1
    return xb, yb, mb, rotated_edges


class DirectionalGraphConv(nn.Module):
    """Implement ``W0 h_i + sum_t W_t h_{i_t} + b`` by typed edges."""

    def __init__(self, cin: int, cout: int, height: int, width: int, m: int,
                 op: str = "directional"):
        super().__init__()
        if op not in OPS:
            raise ValueError(f"op must be one of {OPS}")
        self.cin, self.cout, self.m, self.op = cin, cout, m, op
        self.num_relations = SHELL_TAPCOUNT[m]
        edge_index, edge_type = build_direction_edges(height, width, m)
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_type", edge_type, persistent=False)

        # weight[t] is W_t and root.weight is W_0.  "iso" ties every direction
        # to one matrix; "gcn" additionally folds the self term into the
        # symmetrically normalised adjacency, as in Kipf and Welling (2017).
        n_weights = self.num_relations if op == "directional" else 1
        self.weight = nn.Parameter(torch.empty(n_weights, cout, cin))
        self.root = None if op == "gcn" else nn.Linear(cin, cout, bias=False)
        self.bias = nn.Parameter(torch.zeros(cout))
        self.reset_parameters()

        relation_edges = []
        for t in range(self.num_relations):
            relation_edges.append(torch.nonzero(edge_type == t, as_tuple=False).flatten())
        for t, ids in enumerate(relation_edges):
            self.register_buffer(f"relation_edge_ids_{t}", ids, persistent=False)

    def reset_parameters(self) -> None:
        if self.op == "gcn":
            # Glorot uniform, the initialisation used by Kipf and Welling.
            nn.init.xavier_uniform_(self.weight[0])
        else:
            fan_in = (self.num_relations + 1) * self.cin
            std = math.sqrt(2.0 / fan_in)
            nn.init.normal_(self.weight, std=std)
            nn.init.normal_(self.root.weight, std=std)
        nn.init.zeros_(self.bias)

    def _relation_edge_ids(self, t: int) -> torch.Tensor:
        return getattr(self, f"relation_edge_ids_{t}")

    def forward(
        self,
        h: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply typed message passing to ``h`` of shape ``(B,V,Cin)``.

        ``edge_mask`` is optional and has shape ``(B,E)``.  It is used for
        first-distance edges so removed struts do not transmit messages even if
        both endpoint nodes remain in the specimen.
        """

        if self.op == "gcn":
            return self._forward_gcn(h, node_mask, edge_mask)

        batch, vertices, _ = h.shape
        aggregate = h.new_zeros(batch, vertices, self.cout)
        source_all, target_all = self.edge_index

        for t in range(self.num_relations):
            ids = self._relation_edge_ids(t)
            source = source_all.index_select(0, ids)
            target = target_all.index_select(0, ids)
            source_features = h.index_select(1, source)
            # "directional" uses one matrix per lattice direction; "iso" ties
            # them, which makes the sum over neighbours permutation invariant.
            message = F.linear(source_features, self.weight[0 if self.op == "iso" else t])
            if edge_mask is not None:
                active = edge_mask.index_select(1, ids).unsqueeze(-1)
                message = message * active.to(message.dtype)
            aggregate = aggregate.index_add(1, target, message)

        out = self.root(h) + aggregate + self.bias
        return out * node_mask.unsqueeze(-1).to(out.dtype)

    def _forward_gcn(
        self,
        h: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """One Kipf-Welling layer: ``D^-1/2 (A + I) D^-1/2 H W + b``.

        The adjacency is the first-distance template restricted to the edges
        that actually exist in the specimen, so the degree is computed per
        sample rather than from the defect-free lattice.
        """

        batch, vertices, _ = h.shape
        source, target = self.edge_index
        alive = node_mask.to(h.dtype)

        active = h.new_ones(batch, source.shape[0])
        if edge_mask is not None:
            active = active * edge_mask.to(h.dtype)
        active = active * alive.index_select(1, source) * alive.index_select(1, target)

        # self-loop contributes 1 at every retained node
        degree = alive.clone().index_add(1, target, active).clamp_min(1.0)
        inv_sqrt = degree.pow(-0.5) * alive

        weight = active * inv_sqrt.index_select(1, source) * inv_sqrt.index_select(1, target)
        message = h.index_select(1, source) * weight.unsqueeze(-1)
        aggregate = h.new_zeros(batch, vertices, self.cin).index_add(1, target, message)
        aggregate = aggregate + h * (inv_sqrt * inv_sqrt).unsqueeze(-1)

        out = F.linear(aggregate, self.weight[0]) + self.bias
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class DirectionalResidualGCN(nn.Module):
    """Nine-block direction-aware residual GCN with optional dropout."""

    def __init__(
        self,
        height: int,
        width: int,
        hidden: int = 64,
        schedule: Iterable[int] = DEFAULT_SCHEDULE,
        drop: float = 0.0,
        defect_mode: str = "strict",
        norm_mode: str = "node_layer",
        drop_mode: str = "element",
        fusion_mode: str = "none",
        feature_mode: str = "original",
        op: str = "directional",
    ):
        super().__init__()
        if op not in OPS:
            raise ValueError(f"op must be one of {OPS}")
        if defect_mode not in ("strict", "parent"):
            raise ValueError("defect_mode must be 'strict' or 'parent'")
        if norm_mode not in ("node_layer", "group"):
            raise ValueError("norm_mode must be 'node_layer' or 'group'")
        if drop_mode not in ("element", "channel"):
            raise ValueError("drop_mode must be 'element' or 'channel'")
        if fusion_mode not in ("none", "global_max"):
            raise ValueError("fusion_mode must be 'none' or 'global_max'")
        if feature_mode not in ("original", "loading4"):
            raise ValueError("feature_mode must be 'original' or 'loading4'")
        self.op = op
        self.height, self.width, self.hidden = height, width, hidden
        self.schedule = tuple(int(m) for m in schedule)
        if op == "gcn" and any(m != 1 for m in self.schedule):
            raise ValueError(
                "op='gcn' is the Kipf-Welling layer and propagates only over "
                "struts; use --schedule 1,1,1,1,1,1,1,1,1"
            )
        self.feature_mode = feature_mode
        self.input_projection = nn.Linear(5 if feature_mode == "original" else 4, hidden)

        # Signed vertical loading for the prescribed unit-tension problem.
        # The invalid padded positions in these rows are removed by node_mask.
        loading_code = torch.zeros(2, height, width)
        loading_code[0, 0, :] = -1.0
        loading_code[1, 1, :] = -1.0
        loading_code[0, height - 2, :] = 1.0
        loading_code[1, height - 1, :] = 1.0
        self.register_buffer("loading_code", loading_code.reshape(-1), persistent=False)
        self.convs = nn.ModuleList(
            DirectionalGraphConv(hidden, hidden, height, width, m, op=op)
            for m in self.schedule
        )
        if norm_mode == "node_layer":
            self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in self.schedule)
        else:
            if (2 * hidden) % 8:
                raise ValueError("2*hidden must be divisible by 8 for GroupNorm")
            self.norms = nn.ModuleList(nn.GroupNorm(8, 2 * hidden) for _ in self.schedule)
        if drop <= 0:
            self.dropout = nn.Identity()
        elif drop_mode == "channel":
            self.dropout = nn.Dropout2d(drop)
        else:
            self.dropout = nn.Dropout(drop)
        self.drop = float(drop)
        self.drop_mode = drop_mode
        self.defect_mode = defect_mode
        self.norm_mode = norm_mode
        self.fusion_mode = fusion_mode
        if fusion_mode == "global_max":
            self.readout = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.readout = nn.Linear(hidden, 1)
        self.receptive = 1.0 + sum(SHELL_R[m] for m in self.schedule)

    def tensor_to_nodes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 9:
            raise ValueError(f"expected input (B,9,H,W), received {tuple(x.shape)}")
        if tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError(
                f"model grid {(self.height, self.width)} != input grid {tuple(x.shape[-2:])}"
            )
        batch = x.shape[0]
        if self.feature_mode == "original":
            t = x[:, CH_T + CH_SHARED].permute(0, 2, 3, 1).reshape(batch, -1, 5)
            b = x[:, CH_B + CH_SHARED].permute(0, 2, 3, 1).reshape(batch, -1, 5)
        else:
            # [axial SED, bending SED, normalized coordination, signed loading].
            # Existence is represented by node_mask, while the retained graph
            # itself represents the defect instead of an owning-cell flag.
            spatial = self.height * self.width
            t = x[:, [2, 4, 7]].permute(0, 2, 3, 1).reshape(batch, spatial, 3)
            b = x[:, [3, 5, 8]].permute(0, 2, 3, 1).reshape(batch, spatial, 3)
            t_load = self.loading_code[:spatial].view(1, spatial, 1).expand(batch, -1, -1)
            b_load = self.loading_code[spatial:].view(1, spatial, 1).expand(batch, -1, -1)
            t = torch.cat([t, t_load.to(dtype=t.dtype)], dim=-1)
            b = torch.cat([b, b_load.to(dtype=b.dtype)], dim=-1)
        nodes = torch.cat([t, b], dim=1)
        node_mask = torch.cat(
            [x[:, 0].reshape(batch, -1), x[:, 1].reshape(batch, -1)], dim=1
        ).bool()
        return nodes, node_mask

    def nodes_to_tensor(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        spatial = self.height * self.width
        return torch.stack(
            [
                z[:, :spatial].reshape(batch, self.height, self.width),
                z[:, spatial:].reshape(batch, self.height, self.width),
            ],
            dim=1,
        )

    def nodes_to_hidden_tensor(self, h: torch.Tensor) -> torch.Tensor:
        """Convert ``(B,2HW,C)`` to the original ``(B,2C,H,W)`` layout."""

        batch, _, channels = h.shape
        spatial = self.height * self.width
        t = h[:, :spatial].reshape(batch, self.height, self.width, channels)
        b = h[:, spatial:].reshape(batch, self.height, self.width, channels)
        return torch.cat([t.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)], dim=1)

    def hidden_tensor_to_nodes(self, h: torch.Tensor) -> torch.Tensor:
        """Convert ``(B,2C,H,W)`` back to ``(B,2HW,C)``."""

        batch = h.shape[0]
        t = h[:, : self.hidden].permute(0, 2, 3, 1).reshape(batch, -1, self.hidden)
        b = h[:, self.hidden :].permute(0, 2, 3, 1).reshape(batch, -1, self.hidden)
        return torch.cat([t, b], dim=1)

    def forward(
        self, x: torch.Tensor, edge_mask_m1: torch.Tensor | None = None
    ) -> torch.Tensor:
        nodes, node_mask = self.tensor_to_nodes(x)
        graph_mask = node_mask if self.defect_mode == "strict" else torch.ones_like(node_mask)
        h = F.relu(self.input_projection(nodes))
        if self.defect_mode == "strict":
            h = h * node_mask.unsqueeze(-1).to(h.dtype)

        for m, conv, norm in zip(self.schedule, self.convs, self.norms):
            edge_mask = edge_mask_m1 if (self.defect_mode == "strict" and m == 1) else None
            update = conv(h, graph_mask, edge_mask=edge_mask)
            if self.norm_mode == "group":
                update = self.hidden_tensor_to_nodes(norm(self.nodes_to_hidden_tensor(update)))
            else:
                update = norm(update)
            activated = F.relu(update)
            if self.drop_mode == "channel" and self.drop > 0:
                activated = self.hidden_tensor_to_nodes(
                    self.dropout(self.nodes_to_hidden_tensor(activated))
                )
            else:
                activated = self.dropout(activated)
            h = h + activated
            if self.defect_mode == "strict":
                h = h * node_mask.unsqueeze(-1).to(h.dtype)

        if self.fusion_mode == "global_max":
            # One invariant specimen-level descriptor is broadcast to every
            # node and fused with its local feature.  Missing/padded nodes are
            # excluded from the maximum rather than contributing zeros.
            pool_input = h.masked_fill(~node_mask.unsqueeze(-1), -torch.inf)
            global_feature = pool_input.amax(dim=1)
            global_feature = torch.where(
                torch.isfinite(global_feature), global_feature, torch.zeros_like(global_feature)
            )
            global_feature = global_feature.unsqueeze(1).expand(-1, h.shape[1], -1)
            readout_input = torch.cat([h, global_feature], dim=-1)
        else:
            readout_input = h

        logits = self.readout(readout_input).squeeze(-1)
        logits = logits * node_mask.to(logits.dtype)
        return self.nodes_to_tensor(logits)

    @property
    def edge_counts(self) -> Dict[int, int]:
        return {
            m: int(conv.edge_index.shape[1])
            for m, conv in zip(self.schedule, self.convs)
        }
