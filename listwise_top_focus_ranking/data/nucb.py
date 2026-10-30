from __future__ import annotations
import gc
import inspect
import math
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
EMBEDDING_KEYS = ('embedding', 'embeddings', 'features', 'feature', 'x', 'data', 'representations', 'representation', 'per_tok', 'mean_representations', 'tensor')
TARGET_CANDIDATES = ('target', 'activity', 'fitness', 'label', 'DMS_score', 'dms_score', 'score', 'y')
SPLIT_CANDIDATES = ('split', 'subset', 'partition', 'fold')
SET_CANDIDATES = ('set', 'dataset')
VALIDATION_CANDIDATES = ('validation', 'is_validation', 'valid', 'is_valid', 'val')

def _torch_load(path: Path, *, mmap: bool=False, large_file_fallback_limit_gib: float=8.0) -> Any:
    path = Path(path)
    kwargs = {'map_location': 'cpu'}
    signature = None
    try:
        signature = inspect.signature(torch.load)
        if 'weights_only' in signature.parameters:
            kwargs['weights_only'] = False
        if mmap and 'mmap' in signature.parameters:
            kwargs['mmap'] = True
    except Exception:
        signature = None
    try:
        return torch.load(path, **kwargs)
    except Exception as exc:
        if not mmap:
            raise
        size_gib = path.stat().st_size / 1024 ** 3
        mmap_available = signature is not None and 'mmap' in signature.parameters
        if size_gib > float(large_file_fallback_limit_gib):
            reason = 'torch.load(..., mmap=True) failed' if mmap_available else 'this PyTorch version does not expose torch.load(..., mmap=True)'
            raise RuntimeError(f'Cannot safely load large embedding file {path} ({size_gib:.2f} GiB): {reason}. Refusing eager fallback because it can exhaust RAM. Use a PyTorch version with mmap support or create a pooled [N,D] cache first.') from exc
        fallback = dict(kwargs)
        fallback.pop('mmap', None)
        return torch.load(path, **fallback)

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

def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {'source_path': str(path.resolve()), 'source_size': int(stat.st_size), 'source_mtime_ns': int(stat.st_mtime_ns)}

def _cache_is_valid(cache_obj: Any, source_path: Path) -> bool:
    if not isinstance(cache_obj, dict) or 'embedding' not in cache_obj:
        return False
    sig = _source_signature(source_path)
    return all((cache_obj.get(key) == value for key, value in sig.items()))

@torch.no_grad()
def _mean_pool_token_embedding(tensor: torch.Tensor, *, chunk_size: int=64) -> torch.Tensor:
    if tensor.dim() != 3:
        raise ValueError(f'Expected [N,L,D], got {tuple(tensor.shape)}')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive.')
    n, _, d = tensor.shape
    pooled = torch.empty((n, d), dtype=torch.float32, device='cpu')
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        chunk = tensor[start:end]
        if not chunk.is_floating_point():
            chunk = chunk.float()
        valid = torch.linalg.vector_norm(chunk, dim=-1).ne(0)
        denominator = valid.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
        chunk_sum = chunk.sum(dim=1, dtype=torch.float32)
        pooled[start:end].copy_(chunk_sum / denominator.unsqueeze(1))
        if start == 0 or end == n or start // int(chunk_size) % 100 == 0:
            print(f'[pool] {end:,}/{n:,} sequences', flush=True)
    return pooled

def load_embedding(path: str | Path, *, pool_tokens: bool=False, pool_chunk_size: int=64, cache_path: Optional[str | Path]=None, use_mmap: bool=True) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Embedding file not found: {path}')
    cache = Path(cache_path) if cache_path is not None else None
    if pool_tokens and cache is not None and cache.exists():
        cache_obj = _torch_load(cache, mmap=use_mmap)
        if _cache_is_valid(cache_obj, path):
            tensor = _extract_tensor(cache_obj, str(cache)).detach()
            if tensor.dim() != 2:
                raise ValueError(f'Pooled cache {cache} should contain [N,D], got {tuple(tensor.shape)}')
            print(f'[embedding cache] {cache}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}', flush=True)
            return tensor
        print(f'[embedding cache] stale cache ignored: {cache}', flush=True)
        del cache_obj
        gc.collect()
    raw_obj = _torch_load(path, mmap=use_mmap)
    tensor = _extract_tensor(raw_obj, str(path)).detach()
    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)
    if tensor.dim() not in (2, 3):
        raise ValueError(f'{path}: expected [N,D], [N,L,D], or [N,1,L,D], got {tuple(tensor.shape)} after cleanup.')
    if not tensor.is_floating_point():
        tensor = tensor.float()
    print(f'[embedding raw] {path}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, mmap={use_mmap}', flush=True)
    if not pool_tokens or tensor.dim() == 2:
        return tensor
    print(f'[embedding] mean-pooling token embeddings in chunks of {pool_chunk_size}...', flush=True)
    pooled = _mean_pool_token_embedding(tensor, chunk_size=pool_chunk_size)
    del tensor
    del raw_obj
    gc.collect()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {**_source_signature(path), 'pooling': 'zero_padding_aware_mean', 'embedding': pooled}
        torch.save(payload, cache)
        print(f'[embedding cache] saved pooled embedding: {cache}', flush=True)
        del pooled
        del payload
        gc.collect()
        cache_obj = _torch_load(cache, mmap=use_mmap)
        pooled = _extract_tensor(cache_obj, str(cache)).detach()
    print(f'[embedding pooled] {path}: shape={tuple(pooled.shape)}, dtype={pooled.dtype}', flush=True)
    return pooled

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

def auto_find_split_csv(data_root: str | Path, expected_counts: Tuple[int, int, int], target_col: Optional[str]=None, split_col: Optional[str]=None) -> Path:
    data_root = Path(data_root)
    matches = []
    for path in sorted(data_root.rglob('*.csv')):
        try:
            df = pd.read_csv(path)
            train, val, test, detected_target = split_frame(df, target_col=target_col, split_col=split_col)
            if (len(train), len(val), len(test)) == expected_counts:
                score = 0
                name = path.name.lower()
                if 'base_two_to_many' in name:
                    score += 100
                if 'two_to_many' in name:
                    score += 30
                if 'split' in name:
                    score += 10
                if str(detected_target).lower() == 'target':
                    score += 5
                matches.append((score, path))
        except Exception:
            continue
    if not matches:
        raise FileNotFoundError(f'No CSV below {data_root} has train/val/test counts {expected_counts}. Pass --split-csv explicitly.')
    matches.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    if len(matches) > 1:
        print('[labels] matching CSV files:')
        for score, path in matches:
            print(f'  score={score:3d}: {path}')
    print(f'[labels] selected: {matches[0][1]}')
    return matches[0][1]

def load_targets(expected_counts: Tuple[int, int, int], data_root: str | Path, split_csv: Optional[str | Path]=None, train_csv: Optional[str | Path]=None, val_csv: Optional[str | Path]=None, test_csv: Optional[str | Path]=None, target_col: Optional[str]=None, split_col: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n_train, n_val, n_test = expected_counts
    separate = (train_csv, val_csv, test_csv)
    if any((x is not None for x in separate)):
        if not all((x is not None for x in separate)):
            raise ValueError('Supply all of --train-csv, --val-csv, --test-csv.')
        outputs = []
        cols = []
        for name, path, count in zip(('train', 'val', 'test'), separate, expected_counts):
            df = pd.read_csv(path)
            col = _pick_column(df, target_col, TARGET_CANDIDATES, True)
            values = pd.to_numeric(df[col], errors='coerce').to_numpy(np.float32)
            if np.isnan(values).any() or len(values) != count:
                raise ValueError(f'{name} CSV target/count problem: csv={len(values)}, embedding={count}, file={path}')
            outputs.append(values)
            cols.append(col)
        return (outputs[0], outputs[1], outputs[2], {'mode': 'separate', 'files': [str(x) for x in separate], 'columns': cols})
    if split_csv is None:
        split_csv = auto_find_split_csv(data_root, expected_counts, target_col=target_col, split_col=split_col)
    split_csv = Path(split_csv)
    df = pd.read_csv(split_csv)
    train, val, test, col = split_frame(df, target_col=target_col, split_col=split_col)
    counts = (len(train), len(val), len(test))
    if counts != expected_counts:
        raise ValueError(f'CSV counts {counts} do not match embedding counts {expected_counts}.')
    print(f'[labels] target_col={col!r}, counts={counts}')
    return (train[col].to_numpy(np.float32), val[col].to_numpy(np.float32), test[col].to_numpy(np.float32), {'mode': 'split', 'file': str(split_csv), 'target_col': str(col)})

class CategoricalUniqueBatchSampler(Sampler[List[int]]):

    def __init__(self, targets: Sequence[float] | np.ndarray | torch.Tensor, batch_size: int=64, pair_sampling_alpha: float=0.5, top_class_min_count: int=2, min_class_count: int=1, num_batches: Optional[int]=None, seed: int=42) -> None:
        target_array = torch.as_tensor(targets, dtype=torch.float32).view(-1).cpu().numpy()
        if target_array.size < 2:
            raise ValueError('Categorical unique sampling requires at least 2 samples.')
        if not np.isfinite(target_array).all():
            raise ValueError('Training targets contain NaN or infinite values.')
        if int(batch_size) <= 0:
            raise ValueError('batch_size must be positive.')
        if float(pair_sampling_alpha) < 0:
            raise ValueError('pair_sampling_alpha must be non-negative.')
        if int(min_class_count) < 0 or int(top_class_min_count) < 0:
            raise ValueError('minimum class counts must be non-negative.')
        self.target_array = target_array
        self.batch_size = int(batch_size)
        self.pair_sampling_alpha = float(pair_sampling_alpha)
        self.top_class_min_count = int(top_class_min_count)
        self.min_class_count = int(min_class_count)
        self.seed = int(seed)
        self.epoch = 0
        self.distinct_classes = sorted((float(x) for x in np.unique(target_array)))
        if len(self.distinct_classes) < 2:
            raise ValueError('Categorical unique sampling requires at least 2 target classes.')
        if self.batch_size < len(self.distinct_classes) * self.min_class_count:
            raise ValueError('batch_size is too small for the requested minimum samples per class.')
        self.class_to_indices = {cat: np.flatnonzero(target_array == cat).astype(np.int64) for cat in self.distinct_classes}
        self.possible_cat_pairs = [(self.distinct_classes[i], self.distinct_classes[j]) for i in range(len(self.distinct_classes)) for j in range(i + 1, len(self.distinct_classes))]
        pair_weights = np.asarray([(len(self.class_to_indices[a]) * len(self.class_to_indices[b])) ** self.pair_sampling_alpha for a, b in self.possible_cat_pairs], dtype=np.float64)
        self.category_pair_probabilities = pair_weights / pair_weights.sum()
        endpoint = {cat: 0.0 for cat in self.distinct_classes}
        for (a, b), probability in zip(self.possible_cat_pairs, self.category_pair_probabilities):
            endpoint[a] += float(probability) / 2.0
            endpoint[b] += float(probability) / 2.0
        self.class_endpoint_probabilities = endpoint
        self.class_batch_counts = self._allocate_batch_counts()
        if num_batches is None or int(num_batches) <= 0:
            self.num_batches = int(math.ceil(len(target_array) / self.batch_size))
        else:
            self.num_batches = int(num_batches)

    def _allocate_batch_counts(self) -> dict:
        classes = self.distinct_classes
        raw = {cat: self.class_endpoint_probabilities[cat] * self.batch_size for cat in classes}
        required = {cat: self.min_class_count for cat in classes}
        required[classes[-1]] = max(required[classes[-1]], self.top_class_min_count)
        counts = {cat: max(required[cat], int(math.floor(raw[cat]))) for cat in classes}
        for cat in classes:
            counts[cat] = min(counts[cat], len(self.class_to_indices[cat]))
            if counts[cat] < required[cat]:
                raise ValueError(f'Class {cat:g} has only {len(self.class_to_indices[cat])} samples, smaller than required per-batch minimum {required[cat]}.')
        while sum(counts.values()) > self.batch_size:
            candidates = [cat for cat in classes if counts[cat] > required[cat]]
            if not candidates:
                raise ValueError('Cannot satisfy categorical batch-count constraints.')
            cat = max(candidates, key=lambda c: counts[c] - raw[c])
            counts[cat] -= 1
        while sum(counts.values()) < self.batch_size:
            candidates = [cat for cat in classes if counts[cat] < len(self.class_to_indices[cat])]
            if not candidates:
                raise ValueError('Not enough unique samples to fill the requested batch.')
            cat = max(candidates, key=lambda c: (raw[c] - counts[c], self.class_endpoint_probabilities[c]))
            counts[cat] += 1
        return counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            batch: List[int] = []
            for cat in self.distinct_classes:
                count = int(self.class_batch_counts[cat])
                if count <= 0:
                    continue
                indices = self.class_to_indices[cat]
                chosen = rng.choice(indices, size=count, replace=False)
                batch.extend((int(x) for x in chosen))
            if len(batch) != self.batch_size:
                raise RuntimeError(f'Internal sampler error: produced {len(batch)} != {self.batch_size}.')
            if len(set(batch)) != len(batch):
                raise RuntimeError('Categorical unique sampler produced duplicate indices.')
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches

    def summary(self) -> dict:
        return {'strategy': 'tempered_category_unique_batch_sampling', 'description': 'pair probabilities are converted to class endpoint probabilities; each mini-batch directly draws unique sequences per class; ranking pairs are constructed only by the loss', 'dataset_size': int(len(self.target_array)), 'batch_size': int(self.batch_size), 'pair_sampling_alpha': float(self.pair_sampling_alpha), 'top_class_min_count': int(self.top_class_min_count), 'min_class_count': int(self.min_class_count), 'num_batches_per_epoch': int(self.num_batches), 'classes': [float(x) for x in self.distinct_classes], 'class_sizes': {str(cat): int(len(self.class_to_indices[cat])) for cat in self.distinct_classes}, 'category_pairs': [{'pair': [float(a), float(b)], 'probability': float(prob)} for (a, b), prob in zip(self.possible_cat_pairs, self.category_pair_probabilities)], 'class_endpoint_probabilities': {str(cat): float(self.class_endpoint_probabilities[cat]) for cat in self.distinct_classes}, 'unique_samples_per_batch': {str(cat): int(self.class_batch_counts[cat]) for cat in self.distinct_classes}, 'total_unique_samples_per_batch': int(sum(self.class_batch_counts.values())), 'seed': int(self.seed)}
