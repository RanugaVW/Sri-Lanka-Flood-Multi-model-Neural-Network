import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithConfidence(nn.Module):
    """
    Focal BCE loss optionally weighted by label confidence.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, confidence_weights=None):
        # inputs are expected to be probabilities (after sigmoid)
        # Avoid log(0)
        eps = 1e-7
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if confidence_weights is not None:
            focal_loss = focal_loss * confidence_weights
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for the flood model.
    As per ARCHITECTURE_SPEC.md 2.7 & 3:
    - focal BCE (label_confidence-weighted) for:
      P(flood t+1), P(flood t+2), P(flood t+3), onset
    - Huber regression terms for:
      discharge_t1, zscore_3d_max
    """
    def __init__(self, focal_gamma=2.0, regression_weight=0.3):
        super().__init__()
        self.classification_loss = FocalLossWithConfidence(gamma=focal_gamma, reduction='mean')
        self.regression_loss = nn.HuberLoss(reduction='mean')
        self.regression_weight = regression_weight

    def forward(self, predictions, targets, confidence_weights=None):
        """
        predictions: [batch, 6] (4 probs, 2 regression)
        targets: [batch, 6]
        """
        # Classification indices: 0 to 3
        class_preds = predictions[:, :4]
        class_targets = targets[:, :4]
        
        loss_cls = self.classification_loss(class_preds, class_targets, confidence_weights)
        
        # Regression indices: 4 to 5
        reg_preds = predictions[:, 4:]
        reg_targets = targets[:, 4:]
        
        loss_reg = self.regression_loss(reg_preds, reg_targets)
        
        total_loss = loss_cls + self.regression_weight * loss_reg
        return total_loss
