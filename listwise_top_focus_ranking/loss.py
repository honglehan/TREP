import math
import torch
from torch import nn

class TopFocusListMLE(nn.Module):

    def __init__(self, train_targets, alpha=2.0, tau=0.5, temperature=1.0):
        super().__init__()
        if not all((math.isfinite(x) for x in (alpha, tau, temperature))):
            raise ValueError('Parameters must be finite')
        if alpha < 0 or not 0 <= tau < 1 or temperature <= 0:
            raise ValueError('Invalid loss parameters')
        ref = torch.as_tensor(train_targets).detach().float().reshape(-1).sort().values
        if not ref.numel() or not torch.isfinite(ref).all():
            raise ValueError('Training reference must be nonempty and finite')
        self.register_buffer('reference', ref)
        self.alpha, self.tau, self.temperature = (alpha, tau, temperature)

    def ranking_loss(self, pred, target, inactive_score=None):
        score = pred.reshape(-1)
        target = target.reshape(-1).to(score.device)
        if score.shape != target.shape or not score.numel():
            raise ValueError('Scores and targets must have equal nonempty shapes')
        if not torch.isfinite(score).all() or not torch.isfinite(target).all():
            raise ValueError('Scores and targets must be finite')
        order = target.sort(descending=True).indices
        z = score[order] / self.temperature
        remaining = torch.logcumsumexp(z.flip(0), dim=0).flip(0)
        if inactive_score is not None:
            inactive = inactive_score.reshape(-1).to(device=score.device, dtype=score.dtype)
            if not torch.isfinite(inactive).all():
                raise ValueError('Inactive scores must be finite')
            if inactive.numel():
                tail = torch.logsumexp(inactive / self.temperature, dim=0)
                remaining = torch.logaddexp(remaining, tail)
        ell = remaining - z
        if self.alpha == 0:
            ranking = ell.sum()
        else:
            q = torch.searchsorted(self.reference, target[order].contiguous(), right=True).to(z.dtype) / self.reference.numel()
            r = ((q - self.tau) / (1 - self.tau)).clamp_min(0)
            a = 1 + self.alpha * r
            ranking = score.numel() * (a * ell).sum() / a.sum()
        return ranking

    def forward(self, pred, target, inactive_score=None):
        return self.ranking_loss(pred, target, inactive_score)
