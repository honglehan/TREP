import numpy as np

def _prepare_ranking_arrays(labels, preds):
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    if labels.shape[0] != preds.shape[0]:
        raise ValueError(f'labels and preds must have the same length, got {labels.shape[0]} and {preds.shape[0]}')
    if labels.shape[0] == 0:
        raise ValueError('labels and preds must not be empty')
    return (labels, preds)

def topk_retrieval(labels, preds, k=10):
    labels, preds = _prepare_ranking_arrays(labels, preds)
    k = min(int(k), len(labels))
    if k <= 0:
        raise ValueError('k must be >= 1')
    true_top = set(np.argsort(labels)[-k:])
    pred_top = set(np.argsort(preds)[-k:])
    hits = len(true_top.intersection(pred_top))
    return hits / k

def enrichment_factor_at_k(labels, preds, k=10):
    labels, preds = _prepare_ranking_arrays(labels, preds)
    n = len(labels)
    k = min(int(k), n)
    if k <= 0:
        raise ValueError('k must be >= 1')
    topk = topk_retrieval(labels, preds, k=k)
    random_hit_rate = k / n
    return topk / random_hit_rate

def ndcg_at_k(labels, preds, k=10):
    labels, preds = _prepare_ranking_arrays(labels, preds)
    n = len(labels)
    k = min(int(k), n)
    if k <= 0:
        raise ValueError('k must be >= 1')
    true_top_idx = set(np.argsort(labels)[-k:])
    pred_rank_idx = np.argsort(preds)[::-1][:k]
    relevance = np.array([1.0 if idx in true_top_idx else 0.0 for idx in pred_rank_idx], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance * discounts))
    idcg = float(np.sum(discounts))
    return dcg / idcg if idcg > 0 else 0.0

def ranking_metrics_at_k(labels, preds, k=10):
    return (topk_retrieval(labels, preds, k=k), enrichment_factor_at_k(labels, preds, k=k), ndcg_at_k(labels, preds, k=k))
