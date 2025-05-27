import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

cross_entropy_val = nn.CrossEntropyLoss

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta,device, num_classes=6):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.device=device

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()
    

class GCODLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, u_B, y_B, y_soft_B, a_train):
        """
        logits: Tensor of shape (B, C) from f_theta(Z_B)
        u_B: Tensor of shape (B,) (trainable logits)
        y_B: Tensor of shape (B, C) (one-hot)
        y_soft_B: Tensor of shape (B, C) (soft labels)
        a_train: float (training accuracy between 0 and 1)
        """

        # L1: modified cross-entropy with soft supervision
        ce_loss = self.ce(logits, y_B.argmax(dim=1))  # standard CE
        weight_matrix = y_B * u_B.unsqueeze(1) # shape: (B, C)
        soft_ce = torch.sum(weight_matrix * y_soft_B, dim=1).mean()
        L1 = ce_loss + a_train * soft_ce

        # L2: soft label regression
        weighted_soft = y_soft_B + y_B * u_B.unsqueeze(1)  # shape: (B, C)
        L2 = F.mse_loss(weighted_soft, y_B)

        # L3: KL divergence between distributions from logits and u_B
        p = F.softmax(logits, dim=1)               # shape (B, C)
        log_p = torch.log(p + 1e-8)                # numerical stability
    
        # q from u_B: we first make sure u_B is positive
        u_B_clamped = torch.clamp(u_B, min=1e-6)   # avoid log(0)
        log_u_B = torch.log(u_B_clamped)           # shape (B,)
        q_scalar = torch.sigmoid(-log_u_B)         # shape (B,) in (0,1)
    
        # expand to match class dim
        q = q_scalar.unsqueeze(1).expand_as(p)     # shape (B, C)
    
        L3 = (1 - a_train) * F.kl_div(log_p, q, reduction='batchmean', log_target=False)

        return L1, L2, L3