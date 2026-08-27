import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SARCNN(nn.Module):
    """
    Modality 3: Enhanced SAR CNN.

    Pre-processing pipeline (per chip):
      1. InstanceNorm2d  — zero-mean / unit-std each channel independently,
                           per-chip.  No hardcoded dB assumptions; works
                           regardless of whether chips are in dB, linear,
                           or already pre-normalised.
      2. AvgPool2d       — 3×3 despeckle pass.
      3. 1×1 Conv        — 2 SAR channels → 3 channels for ResNet.
      4. ResNet-18       — pretrained ImageNet backbone (fine-tuned).
      5. LayerNorm       — stabilise the 64-dim SAR embedding for fusion.
                           (LayerNorm works for any number of SAR chips per
                           snapshot; BatchNorm1d collapses to zero when V=1.)

    Missing chips: a learned `missing_embedding` parameter replaces the CNN
    output for nodes with no SAR coverage.
    """
    def __init__(self, input_channels=2, embedding_dim=64):
        super().__init__()

        # Per-chip, per-channel normalisation — no dB-scale assumptions
        self.instance_norm = nn.InstanceNorm2d(input_channels, affine=True)

        # Despeckle: 3×3 Average Pooling to reduce SAR speckle noise
        self.despeckle = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Map 2-channel SAR → 3-channel to utilise ImageNet weights
        self.channel_map = nn.Conv2d(input_channels, 3, kernel_size=1)

        # Pre-trained ResNet-18 backbone
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace final FC: ResNet output (512) → embedding_dim
        # LayerNorm instead of BatchNorm1d — works for any batch size V ≥ 1
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),   # V-agnostic, no collapse at V=1
        )

        # Learned missing embedding (used when SAR chip is absent)
        self.missing_embedding = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, sar_chips, has_sar):
        """
        sar_chips : [N, 2, H, W]   SAR chip values (any scale)
        has_sar   : [N] bool       True where SAR data is available
        """
        N = has_sar.size(0)
        output = torch.zeros(N, self.missing_embedding.size(0),
                             device=has_sar.device)

        valid_indices = torch.where(has_sar)[0]
        if len(valid_indices) > 0:
            chips = sar_chips[valid_indices]          # [V, 2, H, W]

            # 1. Per-chip, per-channel normalisation (InstanceNorm2d)
            chips = self.instance_norm(chips)         # [V, 2, H, W]

            # 2. Despeckle
            chips = self.despeckle(chips)             # [V, 2, H, W]

            # 3. Map to 3-channel space for ResNet
            chips = self.channel_map(chips)           # [V, 3, H, W]

            # 4. ResNet forward + LayerNorm embedding
            output[valid_indices] = self.cnn(chips)  # [V, embedding_dim]

        missing_indices = torch.where(~has_sar)[0]
        if len(missing_indices) > 0:
            output[missing_indices] = self.missing_embedding.expand(
                len(missing_indices), -1)

        return output
