from pathlib import Path
import torch

def _extract_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ('embedding', 'embeddings', 'features', 'feature', 'x', 'data', 'representations', 'representation', 'per_tok', 'mean_representations', 'tensor'):
            if key in value:
                return _extract_tensor(value[key])
        if len(value) == 1:
            return _extract_tensor(next(iter(value.values())))
    raise ValueError('Inactive file must contain a tensor or an unambiguous embedding dictionary')

def load_inactive_embeddings(path, mean, dimension):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Inactive embedding file not found: {path}. Supply --synthetic-embeddings /actual/path/to/inactive_embeddings.pt')
    tensor = _extract_tensor(torch.load(path, map_location='cpu', weights_only=True)).detach()
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim not in (2, 3) or not len(tensor) or tensor.shape[-1] != dimension:
        raise ValueError(f'Invalid inactive embedding shape {tuple(tensor.shape)}; expected dimension {dimension}')
    if mean and tensor.ndim == 3:
        mask = tensor.abs().sum(dim=-1).ne(0).unsqueeze(-1)
        tensor = (tensor * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    if not mean and tensor.ndim != 3:
        raise ValueError('Attention1d requires token inactive embeddings [N,L,D]; pooled [N,D] embeddings cannot reconstruct token inputs. Supply matching token embeddings or train all data with --mean.')
    if not tensor.is_floating_point():
        tensor = tensor.float()
    print(f'inactive embeddings (train only): {path}, shape={tuple(tensor.shape)}', flush=True)
    return tensor

class InactiveBatchSampler:

    def __init__(self, embeddings, batch_size=5, seed=11):
        if batch_size < 1 or batch_size > len(embeddings):
            raise ValueError('Inactive pool must contain at least negative_batch_size samples')
        self.embeddings, self.batch_size = (embeddings, batch_size)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self):
        indices = torch.randperm(len(self.embeddings), generator=self.generator)[:self.batch_size]
        batch = self.embeddings[indices]
        if not torch.isfinite(batch).all():
            raise ValueError('Inactive batch contains nonfinite embeddings')
        return batch
