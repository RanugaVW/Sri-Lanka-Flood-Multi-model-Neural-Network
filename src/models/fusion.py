import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    """
    Fusion block for the multimodal inputs.
    As per ARCHITECTURE_SPEC.md 2.5:
    - Input: [128 (FiLM temporal) + 64 (SAR)] = 192 dim
      (Note: we use 192 instead of 320 because FiLM modulation replaces
       a separate terrain embedding branch to avoid double counting static info).
    - MLP: Linear(192, 192) -> Linear(192, 128) + ReLU
    """
    def __init__(self, concat_dim=192, hidden_dim=192, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, modulated_temporal, sar_embedding):
        """
        modulated_temporal: [batch, 128]
        sar_embedding: [batch, 64]
        """
        fused = torch.cat([modulated_temporal, sar_embedding], dim=1) # [batch, 192]
        return self.mlp(fused)
