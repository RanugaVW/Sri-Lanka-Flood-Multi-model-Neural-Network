import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SARCNN(nn.Module):
    """
    Modality 3: Enhanced SAR CNN.
    - Pre-processing: Average Pooling for despeckling noise.
    - SAR-specific input normalization: VV and VH channels are normalized
      using SAR backscatter statistics (not ImageNet statistics) before
      being mapped to the 3-channel ResNet input space.
    - Architecture: Pre-trained ResNet-18 (ImageNet weights).
    - 1x1 Conv maps 2-channel SAR to 3-channel input for ResNet.
    - Final BatchNorm1d stabilizes the embedding for downstream fusion.
    """
    def __init__(self, input_channels=2, embedding_dim=64,
                 # Typical Sentinel-1 SAR statistics (dB scale, per channel)
                 # VV channel: mean ≈ -10 dB, std ≈ 5 dB
                 # VH channel: mean ≈ -17 dB, std ≈ 5 dB
                 sar_mean=(-10.0, -17.0),
                 sar_std=(5.0, 5.0)):
        super().__init__()

        # Register SAR normalization stats as buffers (move with .to(device))
        self.register_buffer('sar_mean',
                             torch.tensor(sar_mean, dtype=torch.float32).view(1, 2, 1, 1))
        self.register_buffer('sar_std',
                             torch.tensor(sar_std,  dtype=torch.float32).view(1, 2, 1, 1))

        # Despeckle layer: 3×3 Average Pooling to reduce SAR speckle noise
        self.despeckle = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Map 2-channel normalized SAR → 3-channel to utilize ImageNet weights
        self.channel_map = nn.Conv2d(input_channels, 3, kernel_size=1)

        # Pre-trained ResNet-18
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace final FC: ResNet output (512) → embedding_dim
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),   # stabilize embedding for fusion
        )

        # Learned missing embedding (used when SAR chip is absent)
        self.missing_embedding = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, sar_chips, has_sar):
        """
        sar_chips : [N, 2, H, W]   raw SAR backscatter values (dB)
        has_sar   : [N] bool       True where SAR data is available
        """
        N = has_sar.size(0)
        output = torch.zeros(N, self.missing_embedding.size(0),
                             device=has_sar.device)

        valid_indices = torch.where(has_sar)[0]
        if len(valid_indices) > 0:
            chips = sar_chips[valid_indices]                     # [V, 2, H, W]

            # 1. Normalize SAR values using backscatter statistics
            chips = (chips - self.sar_mean) / (self.sar_std + 1e-6)

            # 2. Despeckle
            chips = self.despeckle(chips)                        # [V, 2, H, W]

            # 3. Map to 3-channel space for ResNet
            chips = self.channel_map(chips)                      # [V, 3, H, W]

            # 4. ResNet forward + BatchNorm embedding
            output[valid_indices] = self.cnn(chips)              # [V, embedding_dim]

        missing_indices = torch.where(~has_sar)[0]
        if len(missing_indices) > 0:
            output[missing_indices] = self.missing_embedding.expand(
                len(missing_indices), -1)

        return output
