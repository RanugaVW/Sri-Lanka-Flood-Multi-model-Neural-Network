import torch
import torch.nn as nn
from torchvision.models import resnet18

class SARCNN(nn.Module):
    """
    Modality 3: SAR CNN.
    As per ARCHITECTURE_SPEC.md 2.4:
    - Input: [batch, 2, 512, 512] (VV+VH)
    - ResNet-18-style, trained from scratch. Final embedding [batch, 64].
    - Missing chip strategy: presence mask with learned embedding.
    """
    def __init__(self, input_channels=2, embedding_dim=64):
        super().__init__()
        # ResNet18 from scratch
        self.cnn = resnet18(weights=None)
        # Adapt for 2 input channels instead of 3
        self.cnn.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
            output[valid_indices] = self.cnn(valid_chips)
            
        # Where has_sar is False, use missing_embedding
        missing_indices = torch.where(~has_sar)[0]
        if len(missing_indices) > 0:
            # Broadcast the missing embedding to the right shape
            output[missing_indices] = self.missing_embedding.expand(len(missing_indices), -1)
            
        return output
