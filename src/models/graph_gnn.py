"""River-topology message passing over the 51 gauge nodes.

Modes, selected by `configs/model.yaml: graph.mode`:

``none``
    No message passing. The fused embedding passes through unchanged (and no
    GNN parameters are created at all). Use this as the graph-free control.
``flow``
    Only the 35 directed flow edges. Hydrologically the defensible subset: water
    actually travels along them.
``both`` (the original behaviour)
    Flow edges plus the 204 distance-weighted spatial k-NN edges.

Why the mode switch exists
--------------------------
A sibling project on this same panel
(`Srilanka-Flood-Data-Set-Creation`, see its `docs/RESULTS.md` §2) ran the
graph as the only moving part against an otherwise identical encoder, loss,
panel, seeds and calibrator, and measured message passing as a **cost**:

  * `P3 − P0_x5` (graph vs graph-free, 5 seeds each) = **−0.0828 PR-AUC**,
    95% CI [−0.1003, −0.0668], p = 0.000.
  * `P1 − P0` (adding the 204 spatial k-NN edges) = **−0.0872 PR-AUC**,
    95% CI [−0.1074, −0.0680], p = 0.000 — the spatial edges specifically are
    where most of the damage is.
  * `P2 − P1` (adding flow edges on top of spatial) = −0.0035, CI spans zero —
    the flow edges are at worst neutral.

So the graph is not a free addition to be left permanently on, and the spatial
edges are the part with measured harm. `flow` is therefore a better default than
`both`, and `none` has to be runnable for the ablation to mean anything.

The residual gate
-----------------
The block keeps a residual, but the residual is now **gated by a learned scalar
initialised at zero**:

    x_out = LayerNorm(x_in + tanh(alpha) * gnn(x_in)),   alpha init 0

At initialisation `tanh(0) = 0`, so the GNN contributes exactly nothing and the
model is bit-identical to the graph-free one. Message passing then has to earn
its way in by moving `alpha` off zero. The previous un-gated residual
(`LayerNorm(gnn_out + x_in)`) let training *discount* the graph but never
switch it off: two randomly-initialised GATv2 layers inject full-magnitude noise
into the fused embedding from step one, and given the measured −0.0828 above,
starting from "graph off" is the right prior. `alpha` is also a single readable
number — if it stays near zero, the graph genuinely isn't earning its place, and
that is reportable rather than merely suspected.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GraphGNN(nn.Module):
    """2-layer GATv2 block with a zero-initialised gated residual.

    Parameters
    ----------
    in_channels / hidden_channels / out_channels : widths (128 throughout)
    heads      : attention heads per layer (4)
    num_layers : 1 or 2 GATv2 layers (2)
    dropout    : dropout after layer 1 (0.2)
    mode       : 'none' | 'flow' | 'both'
    """

    def __init__(self, in_channels=128, hidden_channels=128, out_channels=128,
                 heads=4, num_layers=2, dropout=0.2, mode='flow'):
        super().__init__()
        if mode not in ('none', 'flow', 'both'):
            raise ValueError(f"graph.mode must be none|flow|both, got {mode!r}")
        self.mode = mode
        self.num_layers = num_layers
        if mode == 'none':
            return

        if hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must divide by heads ({heads})")

        # concat=True on layer 1 keeps the width at hidden_channels
        self.conv1 = GATv2Conv(in_channels, hidden_channels // heads,
                               heads=heads, concat=True, edge_dim=1)
        self.relu  = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.drop1 = nn.Dropout(p=dropout)
        # concat=False on the final layer prevents a width blow-up
        self.conv2 = (GATv2Conv(hidden_channels, out_channels, heads=heads,
                                concat=False, edge_dim=1)
                      if num_layers >= 2 else None)
        self.norm2 = nn.LayerNorm(out_channels)

        # Zero-initialised residual gate — see the module docstring.
        self.alpha = nn.Parameter(torch.zeros(1))

    @property
    def gate(self) -> float:
        """Current residual gate value, tanh(alpha). 0.0 means the graph is off."""
        if self.mode == 'none':
            return 0.0
        return float(torch.tanh(self.alpha).detach())

    def forward(self, x, edge_index_flow, edge_index_spatial, edge_weight_spatial):
        if self.mode == 'none':
            return x

        # Flow edges carry weight 1.0; spatial edges carry exp(-distance_km/40).
        w_flow = torch.ones(edge_index_flow.size(1), device=x.device, dtype=x.dtype)
        if self.mode == 'flow':
            edge_index  = edge_index_flow
            edge_weight = w_flow
        else:
            edge_index  = torch.cat([edge_index_flow, edge_index_spatial], dim=1)
            edge_weight = torch.cat([w_flow, edge_weight_spatial.to(x.dtype)], dim=0)
        edge_attr = edge_weight.unsqueeze(1)

        x_in = x
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.drop1(self.norm1(self.relu(h)))
        if self.conv2 is not None:
            h = self.conv2(h, edge_index, edge_attr=edge_attr)

        # Gated residual: tanh(alpha) starts at exactly 0, so this block is the
        # identity at initialisation and the graph is opt-in via training.
        return self.norm2(x_in + torch.tanh(self.alpha) * h)
