import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
import metrics_nucb as nucb
from metrics_ired import ranking_metrics_at_k


def retrieval(y, scores, k, dataset):
    y, scores = np.asarray(y).reshape(-1), np.asarray(scores).reshape(-1)
    if y.shape != scores.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(scores).all():
        raise ValueError('Evaluation requires finite aligned labels and scores')
    valid = len(y) > 1 and np.std(y) > 0 and np.std(scores) > 0
    result = dict(spearman=float(spearmanr(y, scores).statistic) if valid else float('nan'),
                  pearson=float(pearsonr(y, scores).statistic) if valid else float('nan'))
    for cutoff in sorted({10, 100, k}):
        if dataset == 'IRED':
            top, ef, ndcg = ranking_metrics_at_k(y, scores, cutoff)
        else:
            top = nucb._topk_retrieval(y, scores, cutoff)
            ef = nucb._enrichment_factor_at_k(y, scores, cutoff)
            ndcg = nucb._binary_ndcg_at_k(y, scores, cutoff)
        result.update({f'top{cutoff}': top, f'ef{cutoff}': ef, f'ndcg{cutoff}': ndcg})
    result.update(report_k=k, topk=result[f'top{k}'], efk=result[f'ef{k}'], ndcgk=result[f'ndcg{k}'])
    if dataset == 'NucB':
        result.update(graded_ndcg10=nucb._graded_ndcg_at_k(y, scores, 10),
                      graded_ndcg100=nucb._graded_ndcg_at_k(y, scores, 100),
                      ndcg=nucb.flip2_ndcg(y, scores),
                      top_class_precision100=nucb._top_class_precision_at_k(y, scores, 100),
                      top_class_recall_at_n=nucb._top_class_recall_at_n(y, scores),
                      pair_acc=nucb._sampled_pairwise_accuracy(y, scores))
        result.update(nucb._class_pair_accuracies(y, scores))
    return result


def forward(model, x, device, attention):
    x = x.float().to(device)
    if attention:
        return model(x, x.abs().sum(-1).ne(0)).reshape(-1)
    return model(x).reshape(-1)


@torch.no_grad()
def evaluate(model, x, y, device, criterion, args, k):
    model.eval()
    scores = torch.cat([forward(model, x[start:start + args.eval_batch_size], device,
                       args.dataset == 'IRED' and not args.mean).cpu()
                       for start in range(0, len(x), args.eval_batch_size)])
    losses = [float(criterion(scores[start:start + args.loss_batch_size].to(device),
                   torch.as_tensor(y[start:start + args.loss_batch_size], device=device)))
              for start in range(0, len(y), args.loss_batch_size)]
    return dict(loss=float(np.mean(losses)), **retrieval(y, scores.numpy(), k, args.dataset))
