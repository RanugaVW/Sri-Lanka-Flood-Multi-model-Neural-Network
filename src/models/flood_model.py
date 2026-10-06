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
terrain_features   : [N, 10]      static node features (see graph_builder.py)
basin_idx          : [N] long     basin identity (embedded inside FiLMTerrain)
sar_chips          : [N, 2, H, W] SAR chip per node (zeros where absent)
has_sar            : [N] bool     presence mask
sar_age_days       : [N] float    age in days of each node's nearest chip
                                  (optional; consumed by the gated fusion)
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

from data.graph_builder import TERRAIN_DIM


class FloodModel(nn.Module):
    """Top-level model. `config` is `configs/model.yaml`, parsed, not ignored.

    Previously every submodule was built from its own constructor defaults and
    `config` was accepted but never read, so `configs/model.yaml` was decorative:
    turning `sar_cnn.enabled` off, or changing `graph.num_layers`, changed
    nothing at all. Every key below is now actually threaded through, so an
    ablation config is a real knob.
    """

    def __init__(self, config=None, num_basins=16):
        super().__init__()
        cfg = config or {}
        te_c  = cfg.get('temporal_encoder', {}) or {}
        fl_c  = cfg.get('film_terrain', {}) or {}
        sar_c = cfg.get('sar_cnn', {}) or {}
        fu_c  = cfg.get('fusion', {}) or {}
        gr_c  = cfg.get('graph', {}) or {}
        hd_c  = cfg.get('heads', {}) or {}

        d_model = int(te_c.get('hidden_dim', 128))

        # ── 1. Temporal ──────────────────────────────────────────────────────
        self.temporal_encoder = TemporalEncoder(
            input_dim=int(te_c.get('input_dim', 33)),
            hidden_dim=d_model,
            lookback=int(te_c.get('window_days', 14)),
            n_layers=int(te_c.get('num_layers', 3)),
            n_heads=int(te_c.get('num_heads', 4)),
            d_emb=int(te_c.get('d_emb', 8)),
            n_freq=int(te_c.get('n_freq', 8)),
            dropout=float(te_c.get('dropout', 0.2)),
            feat_layers=int(te_c.get('feature_layers', 1)),
        )

        # ── 2. FiLM terrain ──────────────────────────────────────────────────
        self.film_terrain = FiLMTerrain(
            input_dim=TERRAIN_DIM,
            num_basins=int(fl_c.get('num_basins', num_basins)),
            hidden_dim=int(fl_c.get('hidden_dim', 64)),
            output_dim=d_model,
        )

        # ── 3. SAR ───────────────────────────────────────────────────────────
        # `sar_cnn.enabled: false` now genuinely removes the branch rather than
        # building an 11.2 M-parameter ResNet-18 and leaving it in the graph.
        self.sar_enabled = bool(sar_c.get('enabled', True))
        sar_dim = int(sar_c.get('embedding_dim', 64))
        self.sar_cnn = (
            SARCNN(input_channels=int(sar_c.get('input_channels', 2)),
                   embedding_dim=sar_dim)
            if self.sar_enabled else None
        )
        if self.sar_enabled and bool(sar_c.get('freeze_backbone', False)):
            # Frozen pretrained features: the sibling project's N6_gated rung
            # froze the encoder because a ResNet-18 cannot learn "what water
            # looks like" from a target with a 1.9% positive rate.
            for prm in self.sar_cnn.cnn.parameters():
                prm.requires_grad = False

        # ── 4. Fusion ────────────────────────────────────────────────────────
        fusion_mode = fu_c.get('mode', 'gated' if self.sar_enabled else 'none')
        if not self.sar_enabled:
            fusion_mode = 'none'
        self.fusion = FusionBlock(
            mode=fusion_mode,
            temporal_dim=d_model,
            sar_dim=sar_dim,
            hidden_dim=int(fu_c.get('hidden_dim', 192)),
            output_dim=int(fu_c.get('output_dim', 128)),
            dropout=float(fu_c.get('dropout', 0.2)),
            gate_bias=float(fu_c.get('gate_bias', -3.0)),
            max_age_days=float(fu_c.get('sar_max_age_days', 12.0)),
        )
        fused_dim = int(fu_c.get('output_dim', 128))

        # ── 5. Graph ─────────────────────────────────────────────────────────
        self.gnn = GraphGNN(
            in_channels=fused_dim,
            hidden_channels=fused_dim,
            out_channels=fused_dim,
            heads=int(gr_c.get('heads', 4)),
            num_layers=int(gr_c.get('num_layers', 2)),
            dropout=float(gr_c.get('dropout', 0.2)),
            mode=gr_c.get('mode', 'flow'),
        )

        # ── 6. Heads ─────────────────────────────────────────────────────────
        self.heads = OutputHeads(
            in_features=fused_dim,
            hidden_features=int(hd_c.get('hidden_dim', 64)),
        )

    def forward(
        self,
        temporal_features,     # [N, L, F]
        terrain_features,      # [N, TERRAIN_DIM]
        basin_idx,              # [N] long
        sar_chips,             # [N, 2, H, W]
        has_sar,               # [N] bool
        edge_index_flow,       # [2, E_flow]
        edge_index_spatial,    # [2, E_sp]
        edge_weight_spatial,   # [E_sp]
        sar_age_days=None,     # [N] float, optional
    ) -> dict:
        # 1. Temporal + cross-feature encoding
        h = self.temporal_encoder(temporal_features)          # [N, 128]

        # 2. FiLM terrain conditioning
        h = self.film_terrain(terrain_features, basin_idx, h)  # [N, 128]

        # 3. SAR embedding (learned missing-embedding when has_sar=False).
        #    Skipped entirely when the branch is disabled — no chip tensor is
        #    touched, which is also why the dataset emits a 1x1 placeholder then.
        sar_emb = self.sar_cnn(sar_chips, has_sar) if self.sar_enabled else None

        # 4. Multimodal fusion (gated on chip presence + frame age)
        h = self.fusion(h, sar_emb, has_sar, sar_age_days)     # [N, 128]

        # 5. Graph message passing (flow + spatial)
        h = self.gnn(h, edge_index_flow,
                     edge_index_spatial, edge_weight_spatial)  # [N, 128]

        # 6. Output heads
        return self.heads(h)                                   # {"logits":[N,4], "reg":[N,2]}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
