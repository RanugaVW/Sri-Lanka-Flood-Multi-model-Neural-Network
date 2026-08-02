import torch
import torch.nn as nn
from models.temporal_encoder import TemporalEncoder
from models.heads import OutputHeads

class NodeLSTMGRU(nn.Module):
    """
    Per-node LSTM/GRU baseline (No Graph).
    Uses the same temporal encoder and output heads as the main model, 
    but skips the GNN and spatial fusion entirely.
    """
    def __init__(self, config=None):
        super().__init__()
        self.temporal_encoder = TemporalEncoder()
        
        # Directly map the 128-dim temporal output to the output heads
        self.heads = OutputHeads(in_features=128)
        
    def forward(self, temporal_features):
        """
        temporal_features: [batch_nodes, window_days, num_features]
        """
        # Encode temporal history
        temporal_out = self.temporal_encoder(temporal_features) # [batch_nodes, 128]
        
        # Predict directly without spatial graph sharing
        predictions = self.heads(temporal_out) # [batch_nodes, 6]
        
        return predictions
