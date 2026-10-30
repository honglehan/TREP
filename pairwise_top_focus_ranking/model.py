from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopFocusedLearningSpace(nn.Module):

    def __init__(self, input_dim: int=1280, hidden_dim: int=256, bottleneck_dim: int=64, dropout: float=0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.projector = nn.Sequential(nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, 2))
        self.register_buffer('a_plus', torch.tensor([1.0, 0.0], dtype=torch.float32))
        self.register_buffer('a_minus', torch.tensor([0.0, 1.0], dtype=torch.float32))

    @staticmethod
    def _canonicalize_embedding(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() not in (2, 3):
            raise ValueError(f'Expected embedding with shape [B, D], [B, L, D], or [B, 1, L, D], but got {tuple(x.shape)}.')
        return x

    def pool_embedding(self, x: torch.Tensor, input_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = self._canonicalize_embedding(x)
        if x.dim() == 2:
            return x
        if input_mask is None:
            input_mask = x.abs().sum(dim=-1).ne(0)
        else:
            if input_mask.dim() == 3 and input_mask.size(-1) == 1:
                input_mask = input_mask.squeeze(-1)
            input_mask = input_mask.bool()
        if input_mask.shape != x.shape[:2]:
            raise ValueError(f'input_mask must have shape [B, L]. Got mask={tuple(input_mask.shape)}, x={tuple(x.shape)}.')
        mask = input_mask.unsqueeze(-1).to(dtype=x.dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denominator

    def encode_learning_space(self, x: torch.Tensor, input_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        pooled = self.pool_embedding(x, input_mask=input_mask)
        parameter_dtype = next(self.projector.parameters()).dtype
        pooled = pooled.to(dtype=parameter_dtype)
        return torch.tanh(self.projector(pooled))

    def anchor_scores(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive_projection = z @ self.a_plus
        inactive_projection = z @ self.a_minus
        return (positive_projection, inactive_projection)

    def score_from_space(self, z: torch.Tensor) -> torch.Tensor:
        positive_projection, inactive_projection = self.anchor_scores(z)
        return positive_projection - inactive_projection

    def forward(self, x: torch.Tensor, input_mask: Optional[torch.Tensor]=None, return_z: bool=False):
        z = self.encode_learning_space(x, input_mask=input_mask)
        score = self.score_from_space(z).view(-1)
        if return_z:
            return (score, z)
        return score

class TopFocusedRankingLoss(nn.Module):

    def __init__(self, tau: float=0.5, temperature: float=0.2, alpha: float=2.0, inactive_weight: float=0.1, inactive_margin: float=0.0, eps: float=1e-08) -> None:
        super().__init__()
        if not 0.0 <= tau < 1.0:
            raise ValueError('tau must satisfy 0 <= tau < 1.')
        if temperature <= 0:
            raise ValueError('temperature must be positive.')
        if alpha < 0:
            raise ValueError('alpha must be non-negative.')
        if inactive_weight < 0:
            raise ValueError('inactive_weight must be non-negative.')
        self.tau = float(tau)
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.inactive_weight = float(inactive_weight)
        self.inactive_margin = float(inactive_margin)
        self.eps = float(eps)
        self.register_buffer('sorted_reference_targets', torch.empty(0, dtype=torch.float32))

    @torch.no_grad()
    def set_reference_targets(self, target: torch.Tensor) -> None:
        target = torch.as_tensor(target, dtype=torch.float32, device=self.sorted_reference_targets.device).detach().view(-1)
        target = target[torch.isfinite(target)]
        if target.numel() == 0:
            raise ValueError('Reference targets are empty or non-finite.')
        sorted_target = torch.sort(target).values
        self.sorted_reference_targets = sorted_target.to(device=self.sorted_reference_targets.device)

    def _check_reference_targets(self) -> None:
        if self.sorted_reference_targets.numel() == 0:
            raise RuntimeError('TopFocusedRankingLoss has no reference targets. Call criterion.set_reference_targets(all_train_targets) before training.')

    def active_percentile(self, target: torch.Tensor) -> torch.Tensor:
        self._check_reference_targets()
        target = target.float().view(-1)
        reference = self.sorted_reference_targets.to(target.device)
        rank = torch.searchsorted(reference, target, right=True)
        return rank.to(dtype=target.dtype) / float(reference.numel())

    def active_relevance(self, target: torch.Tensor) -> torch.Tensor:
        q = self.active_percentile(target)
        return torch.clamp((q - self.tau) / (1.0 - self.tau), min=0.0, max=1.0)

    def _weighted_pairwise_loss(self, score: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        score = score.view(-1)
        relevance = relevance.view(-1)
        relevance_diff = relevance.unsqueeze(1) - relevance.unsqueeze(0)
        pair_mask = relevance_diff > 0
        if not pair_mask.any():
            zero = score.sum() * 0.0
            return zero
        score_diff = score.unsqueeze(1) - score.unsqueeze(0)
        pair_weight = relevance_diff * (1.0 + self.alpha * relevance.unsqueeze(1))
        pair_loss = F.softplus(-score_diff / self.temperature)
        selected_weight = pair_weight[pair_mask]
        selected_loss = pair_loss[pair_mask]
        weighted_loss = (selected_weight * selected_loss).sum() / selected_weight.sum().clamp_min(self.eps)
        return weighted_loss

    def forward(self, active_score: torch.Tensor, active_target: torch.Tensor, inactive_score: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor | int]:
        active_score = active_score.view(-1)
        active_target = active_target.float().view(-1)
        if active_score.numel() != active_target.numel():
            raise ValueError('active_score and active_target must have the same length.')
        active_relevance = self.active_relevance(active_target)
        if inactive_score is None:
            inactive_score = active_score.new_empty(0)
        else:
            inactive_score = inactive_score.view(-1)
        inactive_relevance = active_score.new_zeros(inactive_score.numel())
        all_score = torch.cat([active_score, inactive_score], dim=0)
        all_relevance = torch.cat([active_relevance, inactive_relevance], dim=0)
        rank_loss = self._weighted_pairwise_loss(score=all_score, relevance=all_relevance)
        if inactive_score.numel() > 0:
            inactive_loss = F.relu(inactive_score - self.inactive_margin).pow(2).mean()
        else:
            inactive_loss = active_score.sum() * 0.0
        total_loss = rank_loss + self.inactive_weight * inactive_loss
        return {'loss': total_loss, 'rank_loss': rank_loss, 'inactive_loss': inactive_loss}
