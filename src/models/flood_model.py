"""Terrain-Aware Multimodal Flood GNN — top-level model.

Wires five modalities into a single node representation:

  1. PLR Tokenising Transformer + Cross-feature Attention  (temporal history)
  2. FiLM Terrain Modulation                               (static conditioning)
  3. SAR CNN with learned missing-embedding                 (satellite imagery)
  4. Multimodal Fusion MLP
  5. GATv2 Graph (flow + spatial message passing)           (spatial context)

Then decoupled classification (4 heads) and regression (2 heads) output the
predictions.

Forward signature
-----------------
temporal_features  : [N, L, F]    all N nodes, L lookback days, F features
terrain_features   : [N, 9]       static node features
sar_chips          : [N, 2, H, W] SAR chip per node (zeros where absent)
has_sar            : [N] bool     presence mask
edge_index_flow    : [2, E_flow]
edge_index_spatial : [2, E_sp]
edge_weight_spatial: [E_sp]

Returns
-------
dict
  "logits" : [N, 4]  — raw classification logits (NO sigmoid applied)
  "reg"    : [N, 2]  — raw regression outputs
"""
import torch
import torch.nn as nn

from .temporal_encoder import TemporalEncoder
from .film_terrain     import FiLMTerrain
from .sar_cnn          import SARCNN
from .fusion           import FusionBlock
from .graph_gnn        import GraphGNN
from .heads            import OutputHeads


class FloodModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.temporal_encoder = TemporalEncoder()
        self.film_terrain     = FiLMTerrain(input_dim=9)
        self.sar_cnn          = SARCNN()
        self.fusion           = FusionBlock()
        self.gnn              = GraphGNN()
        self.heads            = OutputHeads()

    def forward(
        self,
        temporal_features,     # [N, L, F]
        terrain_features,      # [N, 9]
        sar_chips,             # [N, 2, H, W]
        has_sar,               # [N] bool
        edge_index_flow,       # [2, E_flow]
        edge_index_spatial,    # [2, E_sp]
        edge_weight_spatial,   # [E_sp]
    ) -> dict:
        # 1. Temporal + cross-feature encoding
        h = self.temporal_encoder(temporal_features)          # [N, 128]

        # 2. FiLM terrain conditioning
        h = self.film_terrain(terrain_features, h)            # [N, 128]

        # 3. SAR embedding (learned missing-embedding when has_sar=False)
        sar_emb = self.sar_cnn(sar_chips, has_sar)           # [N, 64]

        # 4. Multimodal fusion
        h = self.fusion(h, sar_emb)                           # [N, 128]

        # 5. Graph message passing (flow + spatial)
        h = self.gnn(h, edge_index_flow,
                     edge_index_spatial, edge_weight_spatial)  # [N, 128]

        # 6. Output heads
        return self.heads(h)                                   # {"logits":[N,4], "reg":[N,2]}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
