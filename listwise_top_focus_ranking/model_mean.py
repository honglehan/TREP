from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

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
