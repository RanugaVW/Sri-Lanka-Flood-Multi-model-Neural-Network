import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    """
    Fusion block for the multimodal inputs.
    As per ARCHITECTURE_SPEC.md 2.5:
    - Input: [128 (FiLM temporal) + 64 (SAR)] = 192 dim
      (Note: we use 192 instead of 320 because FiLM modulation replaces
       a separate terrain embedding branch to avoid double counting static info).
    - MLP: Linear(192, 192) -> ReLU -> Dropout(0.2) -> Linear(192, 128) -> LayerNorm
    """
    def __init__(self, concat_dim=192, hidden_dim=192, output_dim=128, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, modulated_temporal, sar_embedding):
        """
        modulated_temporal: [N, 128]
        sar_embedding:      [N, 64]
        """
        fused = torch.cat([modulated_temporal, sar_embedding], dim=1)  # [N, 192]
        return self.norm(self.mlp(fused))
