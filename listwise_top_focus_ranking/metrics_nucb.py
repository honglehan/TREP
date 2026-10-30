from __future__ import annotations
import numpy as np
from sklearn.metrics import ndcg_score

def _sampled_pairwise_accuracy(target, score, max_pairs=250000, seed=42):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    n = len(target)
    if n < 2:
        return float('nan')
    total_pairs = n * (n - 1) // 2
    max_pairs = max(1, int(max_pairs))
    if total_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        i_parts, j_parts = ([], [])
        remaining = max_pairs
        while remaining > 0:
            draw = max(remaining * 2, 1024)
            ii = rng.integers(0, n, size=draw, dtype=np.int64)
            jj = rng.integers(0, n, size=draw, dtype=np.int64)
            keep = ii != jj
            ii, jj = (ii[keep], jj[keep])
            take = min(remaining, len(ii))
            i_parts.append(ii[:take])
            j_parts.append(jj[:take])
            remaining -= take
        i, j = (np.concatenate(i_parts), np.concatenate(j_parts))
    target_diff = target[i] - target[j]
    score_diff = score[i] - score[j]
    non_tie = target_diff != 0
    if not np.any(non_tie):
        return float('nan')
    return float(np.mean(target_diff[non_tie] * score_diff[non_tie] > 0))

def _descending_order(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.argsort(-values, kind='stable')

def _topk_retrieval(target, score, k):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    if len(target) < int(k) or int(k) <= 0:
        return float('nan')
    k = int(k)
    true_top = set(_descending_order(target)[:k].tolist())
    pred_top = set(_descending_order(score)[:k].tolist())
    return float(len(true_top.intersection(pred_top)) / k)

def _enrichment_factor_at_k(target, score, k):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    n = len(target)
    k = int(k)
    if k <= 0 or n < k:
        return float('nan')
    topk = _topk_retrieval(target, score, k)
    if not np.isfinite(topk):
        return float('nan')
    random_hit_rate = k / float(n)
    return float(topk / random_hit_rate)

def _top_class_precision_at_k(target, score, k=100):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    if len(target) == 0 or int(k) <= 0:
        return float('nan')
    k = min(int(k), len(target))
    top_class = np.max(target)
    pred_idx = _descending_order(score)[:k]
    return float(np.mean(target[pred_idx] == top_class))

def _top_class_recall_at_n(target, score):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    if len(target) == 0:
        return float('nan')
    top_class = np.max(target)
    truth = target == top_class
    n_top = int(truth.sum())
    if n_top == 0:
        return float('nan')
    pred_idx = _descending_order(score)[:n_top]
    return float(np.sum(truth[pred_idx]) / n_top)

def _graded_ndcg_at_k(target, score, k=100):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    if len(target) == 0 or int(k) <= 0:
        return float('nan')
    k = min(int(k), len(target))
    pred_order = _descending_order(score)[:k]
    ideal_order = _descending_order(target)[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    gains_pred = np.power(2.0, target[pred_order]) - 1.0
    gains_ideal = np.power(2.0, target[ideal_order]) - 1.0
    dcg = float(np.sum(gains_pred * discounts))
    idcg = float(np.sum(gains_ideal * discounts))
    return dcg / idcg if idcg > 0 else float('nan')

def _binary_ndcg_at_k(target, score, k=100):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    n = len(target)
    k = int(k)
    if k <= 0 or n < k:
        return float('nan')
    true_top = set(_descending_order(target)[:k].tolist())
    pred_order = _descending_order(score)[:k]
    relevance = np.asarray([1.0 if idx in true_top else 0.0 for idx in pred_order], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(relevance * discounts))
    idcg = float(np.sum(discounts))
    return float(dcg / idcg) if idcg > 0 else float('nan')

def flip2_ndcg(target, score):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    if len(target) < 2:
        return float('nan')
    positive_target = target - target.min()
    return float(ndcg_score(positive_target.reshape(1, -1), score.reshape(1, -1)))

def _class_pair_accuracies(target, score):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = (target[finite], score[finite])
    classes = np.unique(target)
    results = {}
    values = []
    for hi_pos in range(1, len(classes)):
        hi = classes[hi_pos]
        hi_score = score[target == hi]
        if len(hi_score) == 0:
            continue
        for lo_pos in range(hi_pos):
            lo = classes[lo_pos]
            lo_score = score[target == lo]
            if len(lo_score) == 0:
                continue
            low_sorted = np.sort(lo_score)
            correct = np.searchsorted(low_sorted, hi_score, side='left').sum()
            acc = float(correct / (len(hi_score) * len(lo_score)))
            key = f'class_{(int(hi) if hi.is_integer() else hi)}_vs_{(int(lo) if lo.is_integer() else lo)}_acc'
            results[key] = acc
            values.append(acc)
    results['macro_class_pair_acc'] = float(np.mean(values)) if values else float('nan')
    if len(classes) >= 2:
        top = classes[-1]
        second = classes[-2]
        key = f'class_{(int(top) if top.is_integer() else top)}_vs_{(int(second) if second.is_integer() else second)}_acc'
        results['top_vs_second_acc'] = results.get(key, float('nan'))
    else:
        results['top_vs_second_acc'] = float('nan')
    return results
