import torch
import torch.nn as nn

class FiLMTerrain(nn.Module):
    """
    FiLM Terrain Modulator for Modality 2.
    As per ARCHITECTURE_SPEC.md 2.3:
    - Input: [batch, 9] static terrain features.
    - MLP: Linear(9, 64) -> ReLU() -> Dropout -> Linear(64, 256) -> split to gamma, beta.
    - Applies FiLM modulation to the temporal encoder output.
    - LayerNorm + residual added for training stability.
    """
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=128, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim * 2)  # 256 → gamma + beta
        )
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim
        
    def forward(self, terrain_features, temporal_out):
        # terrain_features: [batch, 9]
        # temporal_out: [batch, 128]
        
        mlp_out = self.mlp(terrain_features)  # [batch, 256]
        
        # Split into gamma and beta
        gamma = mlp_out[:, :self.output_dim]
        beta  = mlp_out[:, self.output_dim:]
        
        # FiLM modulation with residual + LayerNorm for stability
        modulated = gamma * temporal_out + beta
        return self.norm(modulated + temporal_out)   # residual keeps temporal signal
