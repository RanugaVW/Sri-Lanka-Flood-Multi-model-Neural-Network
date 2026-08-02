import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GraphGNN(nn.Module):
    """
    Modality 4: Graph GNN (Single Mode).
    As per ARCHITECTURE_SPEC.md 2.6:
    - Input: [51 nodes, 128] fused vectors
    - 2 layers of GATv2, multi-head (4 heads)
    - Output dimension remains 128.
    """
    def __init__(self, in_channels=128, hidden_channels=128, out_channels=128, heads=4):
        super().__init__()
        # To keep output dim at hidden_channels, set out_channels for first conv = hidden_channels // heads
        self.conv1 = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, concat=True, edge_dim=1)
        # For the final layer, concat=False prevents dimension blow up
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=heads, concat=False, edge_dim=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index_flow, edge_index_spatial, edge_weight_spatial):
        # We combine flow and spatial edges into a single adjacency matrix for the single-mode stack
        # Flow edges get a default weight of 1.0
        edge_weight_flow = torch.ones(edge_index_flow.size(1), device=x.device)
        
        combined_edge_index = torch.cat([edge_index_flow, edge_index_spatial], dim=1)
        combined_edge_weight = torch.cat([edge_weight_flow, edge_weight_spatial], dim=0).unsqueeze(1)
        
        x = self.conv1(x, combined_edge_index, edge_attr=combined_edge_weight)
        x = self.relu(x)
        x = self.conv2(x, combined_edge_index, edge_attr=combined_edge_weight)
        
        return x
