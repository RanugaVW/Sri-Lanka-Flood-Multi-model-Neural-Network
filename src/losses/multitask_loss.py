"""Multi-task loss for the flood early-warning model.

Design decisions
----------------
* Default loss is plain **BCE** (BCEWithLogitsLoss), not focal.
  The friend's ablation ladder found plain BCE outperforms focal on this
  dataset (N3 BCE: PR-AUC 0.8269, N4 focal: 0.7592 — focal was WORSE).

* When focal is enabled, alpha=0.75 up-weights the minority flood class
  (~1.9% positive rate).  The previous default of alpha=0.25 was backwards —
  it was down-weighting the flood class.

* Loss operates on raw LOGITS (not probabilities) via BCEWithLogitsLoss,
  which fuses sigmoid + BCE into a single numerically stable operation.

* `mask`  : [N] float — 1 for valid node-days, 0 for invalid.
* `conf`  : [N] float — label_confidence; down-weights ambiguous days
             (±15% of the discharge threshold).  Only applied when
             loss='focal_conf'.

* Head weights: primary target (flood_t+1) gets weight 1.0; auxiliary
  targets get 0.3 / 0.3 / 0.5 following the friend's defaults.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Optional, Dict

LossComponents = namedtuple('LossComponents', ['total', 'cls', 'reg', 'per_head'])

# Fixed head weights: flood_t+1 is the primary target
HEAD_WEIGHTS = torch.tensor([1.0, 0.3, 0.3, 0.5])


def _focal_bce_logits(logits: torch.Tensor, targets: torch.Tensor,
                      alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Element-wise focal loss on logits (no reduction)."""
    bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p     = torch.sigmoid(logits)
    p_t   = p * targets + (1 - p) * (1 - targets)
    a_t   = alpha * targets + (1 - alpha) * (1 - targets)
    return a_t * (1 - p_t).pow(gamma) * bce


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss over four classification + two regression heads.

    Parameters
    ----------
    loss            : 'bce' | 'focal' | 'focal_conf'
                      'bce'        — plain BCE (recommended default)
                      'focal'      — focal BCE, alpha up-weights flood class
                      'focal_conf' — focal + label_confidence weighting
    focal_alpha     : alpha for focal variants (0.75 = up-weight minority)
    focal_gamma     : focusing exponent (2.0 standard)
    regression_weight : weight for the Huber regression term
    """

    def __init__(
        self,
        loss:               str   = 'bce',
        focal_alpha:        float = 0.75,
        focal_gamma:        float = 2.0,
        regression_weight:  float = 0.2,
    ):
        super().__init__()
        self.loss_type         = loss
        self.focal_alpha       = focal_alpha
        self.focal_gamma       = focal_gamma
        self.regression_weight = regression_weight

    def forward(
        self,
        out:     Dict[str, torch.Tensor],   # from model: {"logits":[N,4], "reg":[N,2]}
        targets: torch.Tensor,               # [N, 6]  (4 cls + 2 reg)
        mask:    Optional[torch.Tensor] = None,   # [N] valid node-day mask
        conf:    Optional[torch.Tensor] = None,   # [N] label confidence
    ) -> LossComponents:
        """
        Parameters
        ----------
        out     : model output dict
        targets : [N, 6]
        mask    : [N] float  — 0 for invalid node-days (padded / outside split)
        conf    : [N] float  — label_confidence for focal_conf mode

        Returns
        -------
        LossComponents(total, cls, reg, per_head)
        """
        logits = out['logits']  # [N, 4]
        reg    = out['reg']     # [N, 2]

        y_cls = targets[:, :4].float()
        y_reg = targets[:, 4:].float()

        # Build per-sample weight: mask × (conf if focal_conf)
        w = mask.float() if mask is not None else torch.ones(logits.size(0), device=logits.device)
        if self.loss_type == 'focal_conf' and conf is not None:
            w = w * conf.float()

        w = w.unsqueeze(-1)  # [N, 1]  → broadcast over heads

        # ── Classification loss ───────────────────────────────────────────────
        if self.loss_type == 'bce':
            per = F.binary_cross_entropy_with_logits(logits, y_cls, reduction='none')
        else:
            per = _focal_bce_logits(logits, y_cls, self.focal_alpha, self.focal_gamma)

        hw = HEAD_WEIGHTS.to(logits.device)          # [4]
        denom      = w.sum().clamp_min(1.0)
        per_head   = (per * w).sum(dim=0) / denom    # [4] — per-head avg loss
        loss_cls   = (per_head * hw).sum()

        # ── Regression loss (Huber) ───────────────────────────────────────────
        reg_per  = F.huber_loss(reg, y_reg, reduction='none', delta=1.0)  # [N, 2]
        reg_denom = (w * reg_per.size(-1)).sum().clamp_min(1.0)
        loss_reg = (reg_per * w).sum() / reg_denom

        total = loss_cls + self.regression_weight * loss_reg

        return LossComponents(
            total=total,
            cls=loss_cls.detach(),
            reg=loss_reg.detach(),
            per_head=per_head.detach(),
        )
