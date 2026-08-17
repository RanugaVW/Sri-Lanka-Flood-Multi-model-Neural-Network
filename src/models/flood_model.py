import torch
import torch.nn as nn
from .temporal_encoder import TemporalEncoder
from .film_terrain import FiLMTerrain
from .sar_cnn import SARCNN
from .fusion import FusionBlock
from .graph_gnn import GraphGNN
from .heads import OutputHeads

class FloodModel(nn.Module):
    """
    Top-level model wiring all components together.
    """
    def __init__(self, config=None):
        super().__init__()
        # In a real setup, config parsing would happen here. We use defaults.
        self.temporal_encoder = TemporalEncoder()
        self.film_terrain = FiLMTerrain(input_dim=9)  # 9 static terrain features per spec
        self.sar_cnn = SARCNN()
        self.fusion = FusionBlock()
        self.gnn = GraphGNN()
        self.heads = OutputHeads()
        
    def forward(self, 
                temporal_features, 
                terrain_features, 
                sar_chips, 
                has_sar, 
                edge_index_flow, 
                edge_index_spatial, 
                edge_weight_spatial):
        
        # 1. Temporal Encoding (Modality 1)
        temporal_out = self.temporal_encoder(temporal_features)
        
        # 2. Terrain Modulation (Modality 2)
        modulated_temporal = self.film_terrain(terrain_features, temporal_out)
        
        # 3. SAR Encoding (Modality 3)
        sar_embedding = self.sar_cnn(sar_chips, has_sar)
        
        # 4. Multimodal Fusion
        fused = self.fusion(modulated_temporal, sar_embedding)
        
        # 5. Graph Processing (Modality 4)
        gnn_out = self.gnn(fused, edge_index_flow, edge_index_spatial, edge_weight_spatial)
        
        # 6. Output Heads
        predictions = self.heads(gnn_out)
        
        return predictions
