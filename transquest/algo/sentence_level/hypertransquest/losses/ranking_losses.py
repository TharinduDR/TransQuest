import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=64):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn):
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m
        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss

class ListMLELoss(nn.Module):
    def __init__(self):
        super(ListMLELoss, self).__init__()
    def forward(self, scores, labels):
        sorted_labels, indices = torch.sort(labels, descending=True, dim=0)
        sorted_scores = scores[indices]
        loss = 0
        for i in range(sorted_scores.size(0)):
            denom = torch.logsumexp(sorted_scores[i:], dim=0)
            loss += (denom - sorted_scores[i])
        return loss / sorted_scores.size(0)

class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return (ce_loss + self.epsilon * (1 - pt)).mean()

class HyperbolicAngleLoss(nn.Module):
    def __init__(self, curvature=1.0):
        super().__init__()
        self.c = curvature
    def forward(self, u, v):
        cos_sim = F.cosine_similarity(u, v, dim=-1)
        return 1 - cos_sim.mean()

class EntailmentConeLoss(nn.Module):
    def __init__(self, K=0.1):
        super().__init__()
        self.K = K
    def forward(self, u, v):
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        aperture_input = (self.K * (1 - u_norm**2) / u_norm).clamp(-1+1e-6, 1-1e-6)
        aperture = torch.asin(aperture_input)
        cos_sim = F.cosine_similarity(u, v, dim=-1).unsqueeze(1).clamp(-1+1e-6, 1-1e-6)
        angle = torch.acos(cos_sim)
        return torch.clamp(angle - aperture, min=0.0).mean()

class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        return 1 - cost