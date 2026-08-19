import torch
import torch.nn as nn


class OutputHeads(nn.Module):
    """Decoupled classification and regression output heads.

    Returns raw LOGITS for the four classification heads and raw continuous
    values for the two regression heads.  Sigmoid is deliberately NOT applied
    here for two reasons:

      1. Numerical stability: BCEWithLogitsLoss (used in the loss module) fuses
         sigmoid + BCE into a single numerically stable operation.
      2. Calibration: temperature scaling is applied to logits after training,
         before sigmoid is ever called.  Applying sigmoid inside the model would
         prevent the calibrator from working correctly.

    Callers that need probabilities (evaluation, inference) should do:
        probs = torch.sigmoid(out["logits"] / temperature)
    """

    def __init__(self, in_features: int = 128, hidden_features: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 4),   # 4 classification logits
        )
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 2),   # 2 regression outputs (raw)
        )

    def forward(self, node_embeddings: torch.Tensor) -> dict:
        """
        node_embeddings : [N, in_features]
        Returns
        -------
        dict with keys
          "logits" : [N, 4]  — raw classification logits (NO sigmoid)
          "reg"    : [N, 2]  — raw regression outputs
        """
        return {
            "logits": self.cls_head(node_embeddings),
            "reg":    self.reg_head(node_embeddings),
        }
