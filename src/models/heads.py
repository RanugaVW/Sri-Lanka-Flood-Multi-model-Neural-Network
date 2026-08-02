import torch
import torch.nn as nn

class OutputHeads(nn.Module):
    """
    Output Heads.
    As per ARCHITECTURE_SPEC.md 2.7:
    - Per-node MLP: Linear(128, 64) -> ReLU -> Linear(64, 6)
    - 6 outputs: P(flood t+1), P(flood t+2), P(flood t+3), onset, discharge_t1, zscore_3d_max
    - Sigmoid on first 4, linear on last 2.
    """
    def __init__(self, in_features=128, hidden_features=64, num_outputs=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_outputs)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, node_embeddings):
        # node_embeddings: [num_nodes, 128]
        logits = self.mlp(node_embeddings) # [num_nodes, 6]
        
        # Apply sigmoid to the first 4 (probabilities / binary onset)
        probs = self.sigmoid(logits[:, :4])
        
        # Leave last 2 as linear (regression)
        regression = logits[:, 4:]
        
        return torch.cat([probs, regression], dim=1)
