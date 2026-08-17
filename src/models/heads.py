import torch
import torch.nn as nn

class OutputHeads(nn.Module):
    """
    Output Heads.
    - Decoupled MLPs for Classification (4 outputs) and Regression (2 outputs).
    - Prevents negative transfer between probabilities and continuous levels.
    """
    def __init__(self, in_features=128, hidden_features=64, dropout=0.1):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 4)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 2)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_embeddings):
        # node_embeddings: [num_nodes, 128]
        cls_logits = self.cls_head(node_embeddings)  # [num_nodes, 4]
        reg_preds  = self.reg_head(node_embeddings)  # [num_nodes, 2]

        # Apply sigmoid to classification outputs (probabilities / binary onset)
        probs = self.sigmoid(cls_logits)

        # Concatenate for backwards compatibility with MultiTaskLoss indexing
        return torch.cat([probs, reg_preds], dim=1)
