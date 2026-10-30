from __future__ import annotations
import math
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
EMBEDDING_KEYS = ('embedding', 'embeddings', 'features', 'feature', 'x', 'data', 'representations', 'representation', 'per_tok', 'mean_representations', 'tensor')
TARGET_CANDIDATES = ('target', 'activity', 'fitness', 'label', 'DMS_score', 'dms_score', 'score', 'y')
SPLIT_CANDIDATES = ('split', 'subset', 'partition', 'fold')
SET_CANDIDATES = ('set', 'dataset')
VALIDATION_CANDIDATES = ('validation', 'is_validation', 'valid', 'is_valid', 'val')

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

class StratifiedActiveBatchSampler(Sampler[List[int]]):

    def __init__(self, targets: Sequence[float] | np.ndarray | torch.Tensor, top_count: int=24, middle_count: int=16, bottom_count: int=24, top_fraction: float=0.1, middle_fraction: float=0.3, num_batches: Optional[int]=None, seed: int=42) -> None:
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
