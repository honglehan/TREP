import math
import sys
import torch
from torch import nn
try:
    from filepaths import SEQ_MODELS
except ImportError:
    SEQ_MODELS = None
if SEQ_MODELS:
    sys.path.append(SEQ_MODELS)
HEAD_TYPE = 'activity_2d'
DEFAULT_HEAD_CONFIG = {'hidden_dim': 256, 'bottleneck_dim': 64, 'dropout': 0.1, 'tanh_scale': 2.0}

class ActivitySpaceHead(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, bottleneck_dim=64, dropout=0.1, tanh_scale=2.0):
        super().__init__()
        if min(input_dim, hidden_dim, bottleneck_dim) < 1:
            raise ValueError('Head dimensions must be positive')
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('Dropout must be finite and in [0,1)')
        if not math.isfinite(tanh_scale) or tanh_scale <= 0:
            raise ValueError('tanh_scale must be finite and positive')
        self.projector = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, 2))
        self.register_buffer('a_plus', torch.tensor([1.0, 0.0]))
        self.register_buffer('a_minus', torch.tensor([0.0, 1.0]))
        self.register_buffer('tanh_scale', torch.tensor(float(tanh_scale)))

    def coordinates(self, pooled):
        return torch.tanh(self.projector(pooled) / self.tanh_scale)

    def score(self, z):
        return (z * (self.a_plus - self.a_minus)).sum(dim=-1, keepdim=True)

    def forward(self, pooled):
        return self.score(self.coordinates(pooled))

class ESMAttention1d(nn.Module):

    def __init__(self, max_length, d_embedding, **head_config):
        super().__init__()
        from sequence_models.structure import Attention1d
        self.attention1d = Attention1d(in_dim=d_embedding)
        self.head = ActivitySpaceHead(d_embedding, **head_config)

    def forward_space(self, x, input_mask):
        pooled = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        return self.head.coordinates(pooled)

    def forward(self, x, input_mask):
        return self.head.score(self.forward_space(x, input_mask))

class ESMAttention1dMean(nn.Module):

    def __init__(self, d_embedding, **head_config):
        super().__init__()
        self.head = ActivitySpaceHead(d_embedding, **head_config)

    def forward_space(self, x):
        return self.head.coordinates(x)

    def forward(self, x):
        return self.head.score(self.forward_space(x))

def build_model(d_embedding, max_length=None, mean=False, head_config=None):
    config = dict(DEFAULT_HEAD_CONFIG)
    if head_config is not None:
        config.update(head_config)
    if mean:
        return ESMAttention1dMean(d_embedding, **config)
    return ESMAttention1d(max_length, d_embedding, **config)
