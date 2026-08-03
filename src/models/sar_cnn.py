import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SARCNN(nn.Module):
    """
    Modality 3: Enhanced SAR CNN.
    - Pre-processing: Average Pooling for despeckling noise.
    - Architecture: Pre-trained ResNet-18 (ImageNet weights).
    - 1x1 Conv maps 2-channel SAR to 3-channel input for ResNet.
    """
    def __init__(self, input_channels=2, embedding_dim=64):
        super().__init__()
        # Despeckle layer: 3x3 Average Pooling to reduce SAR speckle noise
        self.despeckle = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # Map 2-channel SAR to 3-channel to utilize ImageNet weights
        self.channel_map = nn.Conv2d(input_channels, 3, kernel_size=1)
        
        # Pre-trained ResNet18
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Change fc output to embedding_dim
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embedding_dim)
        
        # Learned missing embedding
        self.missing_embedding = nn.Parameter(torch.randn(embedding_dim))
        
    def forward(self, sar_chips, has_sar):
        """
        sar_chips: [batch, 2, 512, 512]
        has_sar: [batch] boolean mask
        """
        batch_size = has_sar.size(0)
        output = torch.zeros(batch_size, self.missing_embedding.size(0), device=has_sar.device)
        
        # Where has_sar is True, run CNN
        valid_indices = torch.where(has_sar)[0]
        if len(valid_indices) > 0:
            valid_chips = sar_chips[valid_indices]
            
            # Apply enhancements
            clean_chips = self.despeckle(valid_chips)
            rgb_chips = self.channel_map(clean_chips)
            
            output[valid_indices] = self.cnn(rgb_chips)
            
        # Where has_sar is False, use missing_embedding
        missing_indices = torch.where(~has_sar)[0]
        if len(missing_indices) > 0:
            # Broadcast the missing embedding to the right shape
            output[missing_indices] = self.missing_embedding.expand(len(missing_indices), -1)
            
        return output
