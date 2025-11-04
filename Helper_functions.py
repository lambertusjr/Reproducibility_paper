import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_curve, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score


try:
    from torch.amp import autocast as _autocast_new  # torch>=2.0 preferred API

    def _autocast_disabled(device_type: str):
        return _autocast_new(device_type=device_type, enabled=False)
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as _autocast_old  # fallback for older torch

    def _autocast_disabled(device_type: str):
        return _autocast_old(enabled=False)

def calculate_metrics(y_true, y_pred, y_pred_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_illicit = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0) # illicit is class 1
    recall = recall_score(y_true, y_pred, average='weighted')
    recall_illicit = recall_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
    
    roc_auc = roc_auc_score(y_true, y_pred_prob[:,1])  # assuming class 1 is the positive class
    roc_auc_illicit = roc_auc_score(y_true, y_pred_prob[:,1])
    
    PR_curve = precision_recall_curve(y_true, y_pred_prob[:,1])
    PRFS = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'precision_illicit': precision_illicit,
        'recall': recall,
        'recall_illicit': recall_illicit,
        'f1': f1,
        'f1_illicit': f1_illicit,
        'roc_auc': roc_auc,
        'roc_auc_illicit': roc_auc_illicit,
        'PR_curve': PR_curve,
        'PRFS': PRFS,
        'kappa': kappa,
    }
    
    return metrics

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            alpha_val = float(alpha)
            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError("alpha float must lie in [0, 1]")
            self.alpha = torch.tensor([alpha_val, 1.0 - alpha_val], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("alpha must be None, float, sequence, or torch.Tensor")

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.ndim != 1:
                raise ValueError("alpha tensor must be 1-dimensional")
            if torch.any(self.alpha < 0):
                raise ValueError("alpha tensor must be non-negative")
            if self.alpha.sum() == 0:
                raise ValueError("alpha tensor must have positive sum")
            self.alpha = self.alpha / self.alpha.sum()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logits = inputs.float()
        targets = targets.long()
        device_type = 'cuda' if logits.is_cuda else 'cpu'
        with _autocast_disabled(device_type):
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prob of the true class

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply alpha correctly depending on type
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]  # per-class weights
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss