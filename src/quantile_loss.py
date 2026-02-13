import torch
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super().__init__()
        self.q = quantile

    def forward(self, y_pred, y_true):
        e = y_true - y_pred
        return torch.max(self.q * e, (self.q - 1) * e).mean()


class QuantilesLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            losses.append(QuantileLoss(q)(preds[:, i], target))
        return sum(losses) / len(self.quantiles)    

class MultiHorizonQuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        preds:  [batch, 3, 3] (3 horizons, 3 quantiles)
        target: [batch, 3]    (Les 3 vrais temps futurs)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            # Erreur pour le quantile i sur les 3 horizons
            errors = target - preds[:, :, i] 
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
            
        return sum(losses) / len(self.quantiles)