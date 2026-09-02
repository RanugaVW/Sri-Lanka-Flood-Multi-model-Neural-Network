import torch
import torch.nn as nn

class FiLMTerrain(nn.Module):
    """
    FiLM Terrain Modulator for Modality 2.
    - Input: [batch, input_dim] continuous/one-hot static terrain features
      (elevation, flow-topology-derived drainage proxies, zone/position
      one-hot — see graph_builder.py) concatenated with a learned basin
      embedding (16 river systems, looked up by `basin_idx`).
    - MLP: Linear(input_dim + basin_emb_dim, 64) -> ReLU -> Dropout ->
      Linear(64, 256) -> split to gamma, beta.
    - Applies FiLM modulation to the temporal encoder output.
    - LayerNorm + residual added for training stability.
    """
    def __init__(self, input_dim=10, num_basins=16, basin_emb_dim=8,
                 hidden_dim=64, output_dim=128, dropout=0.2):
        super().__init__()
        self.basin_emb = nn.Embedding(num_basins, basin_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + basin_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim * 2)  # 256 → gamma + beta
        )
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, terrain_features, basin_idx, temporal_out):
        # terrain_features: [batch, input_dim]
        # basin_idx: [batch] long
        # temporal_out: [batch, 128]

        basin = self.basin_emb(basin_idx)                      # [batch, basin_emb_dim]
        mlp_out = self.mlp(torch.cat([terrain_features, basin], dim=-1))  # [batch, 256]

        # Split into gamma and beta
        gamma = mlp_out[:, :self.output_dim]
        beta  = mlp_out[:, self.output_dim:]

        # FiLM modulation with residual + LayerNorm for stability
        modulated = gamma * temporal_out + beta
        return self.norm(modulated + temporal_out)   # residual keeps temporal signal
