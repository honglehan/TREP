from __future__ import annotations
import inspect
import math
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
EMBEDDING_KEYS = ('embedding', 'embeddings', 'features', 'feature', 'x', 'data', 'representations', 'representation', 'per_tok', 'mean_representations', 'tensor')
TARGET_CANDIDATES = ('target', 'activity', 'fitness', 'label', 'DMS_score', 'dms_score', 'score', 'y')
SPLIT_CANDIDATES = ('split', 'subset', 'partition', 'fold')
SET_CANDIDATES = ('set', 'dataset')
VALIDATION_CANDIDATES = ('validation', 'is_validation', 'valid', 'is_valid', 'val')

def _torch_load(path: Path) -> Any:
    kwargs = {'map_location': 'cpu'}
    try:
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = False
    except Exception:
        pass
    return torch.load(path, **kwargs)

def _extract_tensor(obj: Any, source: str) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for key in EMBEDDING_KEYS:
            if key in obj:
                try:
                    return _extract_tensor(obj[key], f'{source}[{key!r}]')
                except (TypeError, ValueError):
                    pass
        candidates = []
        for key, value in obj.items():
            try:
                candidates.append(_extract_tensor(value, f'{source}[{key!r}]'))
            except (TypeError, ValueError):
                pass
        if not candidates:
            raise ValueError(f'No tensor found in {source}.')
        candidates.sort(key=lambda t: (t.dim() >= 2, t.numel()), reverse=True)
        return candidates[0]
    if isinstance(obj, (list, tuple)):
        tensors = []
        for i, value in enumerate(obj):
            try:
                tensors.append(_extract_tensor(value, f'{source}[{i}]'))
            except (TypeError, ValueError):
                pass
        if not tensors:
            raise ValueError(f'No tensor found in {source}.')
        if len(tensors) == len(obj) and len(tensors) > 1:
            shape = tensors[0].shape
            if all((t.shape == shape for t in tensors)):
                try:
                    return torch.stack(tensors, dim=0)
                except RuntimeError:
                    pass
        tensors.sort(key=lambda t: (t.dim() >= 2, t.numel()), reverse=True)
        return tensors[0]
    raise TypeError(f'Unsupported object type in {source}: {type(obj).__name__}')

def load_embedding(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Embedding file not found: {path}')
    tensor = _extract_tensor(_torch_load(path), str(path)).detach().cpu()
    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)
    if tensor.dim() not in (2, 3):
        raise ValueError(f'{path}: expected [N,D], [N,L,D], or [N,1,L,D], got {tuple(tensor.shape)} after cleanup.')
    if not tensor.is_floating_point():
        tensor = tensor.float()
    print(f'[embedding] {path}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}')
    return tensor

def _pick_column(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str], required: bool) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f'Column {explicit!r} not found. Available: {list(df.columns)}')
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise KeyError(f'Could not find one of {list(candidates)} in {list(df.columns)}')
    return None

def _text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def _matches(series: pd.Series, values: Sequence[str]) -> pd.Series:
    accepted = {str(v).strip().lower() for v in values}
    return _text(series).isin(accepted)

def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').fillna(0).ne(0)
    return _text(series).isin({'1', 'true', 't', 'yes', 'y', 'val', 'valid', 'validation'})

def split_frame(df: pd.DataFrame, target_col: Optional[str]=None, split_col: Optional[str]=None, train_value: str='train', val_value: str='val', test_value: str='test') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    target_col = _pick_column(df, target_col, TARGET_CANDIDATES, True)
    selected_split = split_col
    if selected_split is None:
        selected_split = _pick_column(df, None, SPLIT_CANDIDATES, False)
    if selected_split is not None:
        selected_split = _pick_column(df, selected_split, (), True)
        values = df[selected_split]
        train_mask = _matches(values, (train_value, 'train', 'training'))
        val_mask = _matches(values, (val_value, 'val', 'valid', 'validation', 'dev'))
        test_mask = _matches(values, (test_value, 'test', 'testing'))
    else:
        set_col = _pick_column(df, None, SET_CANDIDATES, False)
        validation_col = _pick_column(df, None, VALIDATION_CANDIDATES, False)
        if set_col is None:
            raise KeyError('No split column and no set column found.')
        values = df[set_col]
        base_train = _matches(values, (train_value, 'train', 'training'))
        test_mask = _matches(values, (test_value, 'test', 'testing'))
        if validation_col is None:
            val_mask = _matches(values, (val_value, 'val', 'valid', 'validation', 'dev'))
            train_mask = base_train
        else:
            is_val = _truthy(df[validation_col])
            val_mask = base_train & is_val
            train_mask = base_train & ~is_val
    parts = [df.loc[train_mask].copy(), df.loc[val_mask].copy(), df.loc[test_mask].copy()]
    for name, part in zip(('train', 'val', 'test'), parts):
        part[target_col] = pd.to_numeric(part[target_col], errors='coerce')
        if part[target_col].isna().any():
            raise ValueError(f'{name} contains non-numeric/NaN targets.')
    return (parts[0], parts[1], parts[2], target_col)

def load_targets(expected_counts, split_csv, target_col=None, split_col=None):
    split_csv = Path(split_csv)
    train, val, test, col = split_frame(pd.read_csv(split_csv), target_col=target_col, split_col=split_col)
    counts = (len(train), len(val), len(test))
    if counts != expected_counts:
        raise ValueError(f'CSV counts {counts} do not match embedding counts {expected_counts}.')
    print(f'[labels] target_col={col!r}, counts={counts}')
    return (train[col].to_numpy(np.float32), val[col].to_numpy(np.float32), test[col].to_numpy(np.float32), {'mode': 'split', 'file': str(split_csv), 'target_col': str(col)})

class EmbeddingTargetDataset(Dataset):

    def __init__(self, embedding: torch.Tensor, target: np.ndarray):
        if embedding.size(0) != len(target):
            raise ValueError('Embedding and target counts differ.')
        self.embedding = embedding
        self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return self.embedding.size(0)

    def __getitem__(self, index: int):
        return (self.embedding[index], self.target[index])

class EmbeddingOnlyDataset(Dataset):

    def __init__(self, embedding: torch.Tensor):
        self.embedding = embedding

    def __len__(self) -> int:
        return self.embedding.size(0)

    def __getitem__(self, index: int):
        return self.embedding[index]

class StratifiedActiveBatchSampler(Sampler[List[int]]):

    def __init__(self, targets: Sequence[float] | np.ndarray | torch.Tensor, top_count: int=32, middle_count: int=32, bottom_count: int=32, top_fraction: float=0.1, middle_fraction: float=0.3, num_batches: Optional[int]=None, seed: int=42) -> None:
        target_array = torch.as_tensor(targets, dtype=torch.float32).view(-1).cpu().numpy()
        if target_array.size < 3:
            raise ValueError('Stratified sampling requires at least 3 samples.')
        if not np.isfinite(target_array).all():
            raise ValueError('Training targets contain NaN or infinite values.')
        counts = {'top_count': int(top_count), 'middle_count': int(middle_count), 'bottom_count': int(bottom_count)}
        if any((value <= 0 for value in counts.values())):
            raise ValueError(f'Every stratum count must be positive: {counts}')
        if not 0.0 < top_fraction < 1.0:
            raise ValueError('top_fraction must satisfy 0 < value < 1.')
        if not 0.0 < middle_fraction < 1.0:
            raise ValueError('middle_fraction must satisfy 0 < value < 1.')
        if top_fraction + middle_fraction >= 1.0:
            raise ValueError('top_fraction + middle_fraction must be smaller than 1.')
        self.top_count = counts['top_count']
        self.middle_count = counts['middle_count']
        self.bottom_count = counts['bottom_count']
        self.batch_size = self.top_count + self.middle_count + self.bottom_count
        self.top_fraction = float(top_fraction)
        self.middle_fraction = float(middle_fraction)
        self.seed = int(seed)
        self.epoch = 0
        ascending = np.argsort(target_array, kind='stable')
        n = len(ascending)
        n_top = max(1, int(math.ceil(n * self.top_fraction)))
        n_middle = max(1, int(math.ceil(n * self.middle_fraction)))
        if n_top + n_middle >= n:
            raise ValueError('The dataset is too small for the requested stratum fractions.')
        top_start = n - n_top
        middle_start = top_start - n_middle
        self.bottom_indices = ascending[:middle_start].astype(np.int64)
        self.middle_indices = ascending[middle_start:top_start].astype(np.int64)
        self.top_indices = ascending[top_start:].astype(np.int64)
        if min(len(self.top_indices), len(self.middle_indices), len(self.bottom_indices)) == 0:
            raise ValueError('At least one target stratum is empty.')
        if num_batches is None or int(num_batches) <= 0:
            self.num_batches = int(math.ceil(n / self.batch_size))
        else:
            self.num_batches = int(num_batches)
        self.target_array = target_array
        self.thresholds = {'top_min_target': float(np.min(target_array[self.top_indices])), 'middle_min_target': float(np.min(target_array[self.middle_indices])), 'middle_max_target': float(np.max(target_array[self.middle_indices])), 'bottom_max_target': float(np.max(target_array[self.bottom_indices]))}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw_cyclic(indices: np.ndarray, count: int, rng: np.random.Generator, state: dict) -> List[int]:
        output: List[int] = []
        while len(output) < count:
            permutation = state.get('permutation')
            cursor = int(state.get('cursor', 0))
            if permutation is None or cursor >= len(permutation):
                permutation = rng.permutation(indices)
                cursor = 0
            take = min(count - len(output), len(permutation) - cursor)
            output.extend((int(value) for value in permutation[cursor:cursor + take]))
            cursor += take
            state['permutation'] = permutation
            state['cursor'] = cursor
        return output

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        states = {'top': {}, 'middle': {}, 'bottom': {}}
        for _ in range(self.num_batches):
            batch = []
            batch.extend(self._draw_cyclic(self.top_indices, self.top_count, rng, states['top']))
            batch.extend(self._draw_cyclic(self.middle_indices, self.middle_count, rng, states['middle']))
            batch.extend(self._draw_cyclic(self.bottom_indices, self.bottom_count, rng, states['bottom']))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches

    def summary(self) -> dict:
        return {'dataset_size': int(len(self.target_array)), 'batch_size': int(self.batch_size), 'num_batches_per_epoch': int(self.num_batches), 'draws_per_batch': {'top_10_percent': int(self.top_count), 'next_30_percent': int(self.middle_count), 'bottom_60_percent': int(self.bottom_count)}, 'stratum_sizes': {'top_10_percent': int(len(self.top_indices)), 'next_30_percent': int(len(self.middle_indices)), 'bottom_60_percent': int(len(self.bottom_indices))}, 'thresholds': self.thresholds, 'seed': int(self.seed)}

def make_stratified_active_loader(dataset: Dataset, targets: Sequence[float] | np.ndarray | torch.Tensor, top_count: int, middle_count: int, bottom_count: int, top_fraction: float, middle_fraction: float, num_batches: Optional[int], seed: int, num_workers: int, pin_memory: bool) -> DataLoader:
    batch_sampler = StratifiedActiveBatchSampler(targets=targets, top_count=top_count, middle_count=middle_count, bottom_count=bottom_count, top_fraction=top_fraction, middle_fraction=middle_fraction, num_batches=num_batches, seed=seed)
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)

def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, drop_last: bool=False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
