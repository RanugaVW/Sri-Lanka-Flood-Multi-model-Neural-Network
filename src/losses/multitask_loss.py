import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

LossComponents = namedtuple('LossComponents', ['total', 'cls', 'reg'])


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

    Returns
    -------
    LossComponents(total, cls, reg) — a named tuple so callers can log
    cls/reg breakdowns without a second forward pass.
    """
    def __init__(self, focal_gamma=2.0, focal_alpha=0.25, regression_weight=0.3):
        super().__init__()
        self.classification_loss = FocalLossWithConfidence(
            alpha=focal_alpha, gamma=focal_gamma, reduction='mean'
        )
        self.regression_loss = nn.HuberLoss(reduction='mean')
        self.regression_weight = regression_weight

    def forward(self, predictions, targets, confidence_weights=None):
        """
        predictions : [N, 6]  (4 probs already sigmoid-ed, 2 regression)
        targets     : [N, 6]
        Returns     : LossComponents(total, cls, reg)
        """
        # Classification indices: 0–3
        loss_cls = self.classification_loss(
            predictions[:, :4], targets[:, :4], confidence_weights
        )
        # Regression indices: 4–5
        loss_reg = self.regression_loss(predictions[:, 4:], targets[:, 4:])

        total = loss_cls + self.regression_weight * loss_reg
        return LossComponents(total=total, cls=loss_cls, reg=loss_reg)
