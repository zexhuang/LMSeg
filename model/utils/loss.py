from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class BCELogitsSmoothingLoss(nn.Module):
    def __init__(self, 
                 eps: float = 0.1, 
                 reduction: str = 'mean'):
        super(BCELogitsSmoothingLoss, self).__init__()
        assert 0 <= eps < 1, "eps must be between 0 and 1."
        self.eps = eps
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        if self.eps > 0:
            target = target * (1 - self.eps) + (1 - target) * self.eps

        loss = self.bce_with_logits(input, target)
        return loss
    
    
class CrossEntropyWithLabelWeight(nn.Module):
    def __init__(self, 
                 ignore_index: Optional[int] = None, 
                 label_smoothing: float = 0.0, 
                 reduction: str = 'mean',
                 label_weights: Optional[torch.Tensor] = None):
        super(CrossEntropyWithLabelWeight, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.label_weights = label_weights

    def forward(self, input, target):        
        loss = F.cross_entropy(input, target, 
                               weight=self.label_weights.to(input.device), 
                               ignore_index=self.ignore_index, 
                               label_smoothing=self.label_smoothing, 
                               reduction=self.reduction)
        return loss