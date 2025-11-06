# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)
        loss = -log_preds.sum(dim=1)
        nll = F.nll_loss(log_preds, target, reduction='none')
        loss = (1 - self.eps) * nll + self.eps / n_classes * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class HybridFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, eps=0.1, lambda_focal=0.7):
        super(HybridFocalLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.ls = LabelSmoothingCrossEntropy(eps=eps)
        self.lambda_focal = lambda_focal

    def forward(self, inputs, targets):
        loss_focal = self.focal(inputs, targets)
        loss_ls = self.ls(inputs, targets)
        return self.lambda_focal * loss_focal + (1 - self.lambda_focal) * loss_ls
