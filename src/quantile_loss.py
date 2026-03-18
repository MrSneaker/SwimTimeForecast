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
    
class MultiHorizonQuantileLossV2(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        Adaptation pour TemporalFusionTransformer
        
        preds:  [batch, horizon, 1, num_quantiles]  <-- Sortie du modèle TFT
        target: [batch, horizon]                    <-- Les vrais temps futurs
        """
        # On vérifie si la dimension "target_dimension" (le 1) est présente
        # Si oui, on la supprime pour pouvoir faire la soustraction avec target
        if preds.dim() == 4:
            # On passe de [B, H, 1, Q] à [B, H, Q]
            preds = preds.squeeze(2)

        losses = []
        for i, q in enumerate(self.quantiles):
            # Extraction du i-ème quantile pour tous les horizons
            # pred_q shape: [batch, horizon]
            pred_q = preds[:, :, i] 
            
            # Calcul de l'erreur (différence entre réalité et prédiction du quantile)
            errors = target - pred_q
            
            # Formule de la Pinball Loss (Quantile Loss)
            # loss = max((q-1)*error, q*error)
            loss = torch.max((q - 1) * errors, q * errors)
            
            # On stocke la moyenne de cet échantillon
            losses.append(loss.mean())
            
        # Retourne la moyenne des pertes sur tous les quantiles
        return sum(losses) / len(self.quantiles)