import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import DEFAULTS, configure
from data import amylase, nucb, trpb
from data.ired import load_inactive_embeddings, InactiveBatchSampler
from loss import TopFocusListMLE
from metrics import evaluate, forward
from model_mean import TopFocusedLearningSpace


def parser():
    p = argparse.ArgumentParser(description='Shared Top-Focus ListMLE training')
    p.add_argument('--dataset', choices=DEFAULTS, required=True)
    p.add_argument('--project-root', default='/home/proif/nilufer/lehan/FLIP')
    p.add_argument('--embedding-model', default='esm2-650M')
    for name in ('embedding-dir', 'train-emb', 'val-emb', 'test-emb', 'negative-emb',
                 'train-labels', 'val-labels', 'test-labels', 'split-csv', 'target-col',
                 'split-col', 'output-dir', 'pooled-cache-dir'):
        p.add_argument('--' + name)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=300)
    for name in ('batch-size', 'seed', 'loss-batch-size'):
        p.add_argument('--' + name, type=int)
    for name in ('learning-rate', 'weight-decay', 'dropout', 'grad-clip'):
        p.add_argument('--' + name, type=float)
    p.add_argument('--negative-batch-size', type=int, default=5)
    p.add_argument('--eval-batch-size', type=int, default=2048)
    p.add_argument('--pool-chunk-size', type=int, default=64)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--bottleneck-dim', type=int, default=64)
    p.add_argument('--alpha', type=float, default=2.)
    p.add_argument('--tau', type=float, default=.5)
    p.add_argument('--temperature', type=float, default=.2)
    p.add_argument('--pair-sampling-alpha', type=float, default=.5)
    p.add_argument('--top-class-min-count', type=int, default=2)
    p.add_argument('--min-class-count', type=int, default=1)
    p.add_argument('--top-active-fraction', type=float, default=.1)
    p.add_argument('--middle-active-fraction', type=float, default=.3)
    p.add_argument('--mean', action='store_true', help='Use IRED pre-pooled embeddings')
    p.add_argument('--no-mmap', action='store_true')
    return p


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=True) + '\n')


def load_data(args, out):
    module = {'Amylase': amylase, 'NucB': nucb, 'TrpB': trpb}.get(args.dataset)
    cache = Path(args.pooled_cache_dir) if args.pooled_cache_dir else out / 'pooled_embeddings'
    if args.dataset == 'IRED':
        xs, ys = [], []
        for part in ('train', 'val', 'test'):
            x = torch.load(getattr(args, part + '_emb'), map_location='cpu', weights_only=True)
            y = torch.load(getattr(args, part + '_labels'), map_location='cpu', weights_only=True)
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise ValueError('IRED embedding and label files must contain tensors')
            if x.ndim != (2 if args.mean else 3):
                raise ValueError('IRED needs [N,L,D] token or --mean [N,D] embeddings')
            xs.append(x)
            ys.append(y.reshape(-1).float().numpy())
        negative = load_inactive_embeddings(args.negative_emb, args.mean, xs[0].shape[-1])
        info = dict(mode='tensor_labels', files=[args.train_labels, args.val_labels, args.test_labels])
    else:
        cache.mkdir(parents=True, exist_ok=True)
        def load(path, name):
            loader = nucb if args.dataset == 'NucB' else trpb
            return loader.load_embedding(path, pool_tokens=True, pool_chunk_size=args.pool_chunk_size,
                        cache_path=cache / f'{name}_mean.pt', use_mmap=not args.no_mmap)
        xs = [load(getattr(args, part + '_emb'), part) for part in ('train', 'val', 'test')]
        negative = load(args.negative_emb, 'negative')
        kwargs = dict(expected_counts=tuple(map(len, xs)), split_csv=args.split_csv,
                      target_col=args.target_col, split_col=args.split_col)
        if args.dataset != 'TrpB':
            kwargs['data_root'] = str(Path(args.project_root) / 'data' / args.dataset)
        *ys, info = module.load_targets(**kwargs)
    for part, x, y in zip(('train', 'val', 'test'), xs, ys):
        if len(x) != len(y) or not len(y) or not np.isfinite(y).all():
            raise ValueError(f'Invalid {part} labels/count')
        if args.dataset == 'NucB' and not np.isin(y, [0, 1, 2, 3]).all():
            raise ValueError('NucB labels must be 0,1,2,3')
    if len({x.shape[-1] for x in [*xs, negative]}) != 1:
        raise ValueError('Embedding dimensions differ')
    if len(negative) < args.negative_batch_size:
        raise ValueError('Inactive pool is smaller than negative batch size')
    save_json(out / 'label_info.json', info)
    return xs, ys, negative


def make_train_loader(args, x, y):
    dataset = TensorDataset(x, torch.as_tensor(y, dtype=torch.float32))
    if args.dataset == 'IRED':
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if args.dataset == 'NucB':
        sampler = nucb.CategoricalUniqueBatchSampler(y, batch_size=args.batch_size,
                    pair_sampling_alpha=args.pair_sampling_alpha,
                    top_class_min_count=args.top_class_min_count,
                    min_class_count=args.min_class_count, seed=args.seed)
    else:
        module = amylase if args.dataset == 'Amylase' else trpb
        sampler = module.StratifiedActiveBatchSampler(y, top_count=32, middle_count=32,
                    bottom_count=32, top_fraction=args.top_active_fraction,
                    middle_fraction=args.middle_active_fraction, seed=args.seed)
    return DataLoader(dataset, batch_sampler=sampler)


def main():
    args = configure(parser().parse_args())
    for name in ('epochs', 'batch_size', 'loss_batch_size', 'negative_batch_size',
                 'eval_batch_size', 'pool_chunk_size'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.dataset != 'IRED' and args.batch_size != 96:
        raise ValueError('Amylase/NucB/TrpB use measured batch size 96')
    if args.mean and args.dataset != 'IRED':
        raise ValueError('--mean selects IRED pre-pooled inputs only')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning-rate must be finite and positive')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('weight-decay must be finite and nonnegative')
    if not np.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ValueError('grad-clip must be finite and nonnegative')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = Path(args.output_dir or f'runs/{args.dataset}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out.resolve())
    device = torch.device(args.device)
    xs, ys, negative = load_data(args, out)
    val_k = 10
    test_k = int((ys[2] == 3).sum()) if args.dataset == 'NucB' else 100
    if val_k < 1 or test_k < 1 or len(ys[1]) < val_k:
        raise ValueError('Invalid validation/test retrieval K')
    args.val_k, args.test_k = val_k, test_k
    args.selection_metric = f'val_ef{val_k}'
    save_json(out / 'config.json', vars(args))
    train_loader = make_train_loader(args, xs[0], ys[0])
    if hasattr(train_loader.batch_sampler, 'summary'):
        save_json(out / 'active_sampler.json', train_loader.batch_sampler.summary())
    if args.dataset == 'IRED':
        from model_ired import build_model
        model = build_model(xs[0].shape[-1], mean=args.mean,
                    head_config=dict(hidden_dim=args.hidden_dim, bottleneck_dim=args.bottleneck_dim,
                                     dropout=args.dropout, tanh_scale=2.)).to(device)
        inactive_sampler = InactiveBatchSampler(negative, args.negative_batch_size, args.seed + 1)
    else:
        model = TopFocusedLearningSpace(xs[0].shape[-1], args.hidden_dim,
                                       args.bottleneck_dim, args.dropout).to(device)
        negative_loader = DataLoader(TensorDataset(negative), batch_size=args.negative_batch_size,
                                     shuffle=True, drop_last=True)
    criterion = TopFocusListMLE(ys[0], args.alpha, args.tau, args.temperature).to(device)
    optimizer_type = torch.optim.Adam if args.dataset == 'IRED' else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best, best_epoch, history = -float('inf'), 0, []
    attention = args.dataset == 'IRED' and not args.mean
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch - 1)
        model.train()
        if args.dataset != 'IRED':
            negative_iterator = iter(negative_loader)
        totals, measured = [], []
        for x, y in train_loader:
            if args.dataset == 'IRED':
                nx = inactive_sampler.sample()
            else:
                try:
                    nx = next(negative_iterator)[0]
                except StopIteration:
                    negative_iterator = iter(negative_loader)
                    nx = next(negative_iterator)[0]
            optimizer.zero_grad(set_to_none=True)
            scores = forward(model, x, device, attention)
            inactive_scores = forward(model, nx, device, attention)
            y = y.to(device)
            loss = criterion(scores, y, inactive_scores)
            with torch.no_grad():
                measured.append(float(criterion(scores.detach(), y)))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals.append(float(loss.detach()))
        row = dict(epoch=epoch, train_batch_loss=float(np.mean(totals)),
                   train_batch_measured_loss=float(np.mean(measured)),
                   train_batch_inactive_increment=float(np.mean(totals) - np.mean(measured)))
        for split, x, y in zip(('train', 'val'), xs[:2], ys[:2]):
            metrics = evaluate(model, x, y, device, criterion, args, val_k)
            row.update({f'{split}_{key}': value for key, value in metrics.items()})
        history.append(row)
        with (out / 'history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        state = dict(epoch=epoch, model_state_dict=model.state_dict(),
                     optimizer_state_dict=optimizer.state_dict(), config=vars(args),
                     input_dim=xs[0].shape[-1], validation=row)
        value = row[args.selection_metric]
        if np.isfinite(value) and value >= best:
            best, best_epoch = value, epoch
            torch.save(state, out / 'best_model.pt')
        if epoch == args.epochs:
            torch.save(state, out / 'last_model.pt')
        print(f'epoch={epoch:03d} train_loss={row["train_batch_loss"]:.6f} '
              f'{args.selection_metric}={value:.6f} best_epoch={best_epoch}', flush=True)
    if best_epoch == 0:
        raise ValueError('Validation selection metric was undefined in every epoch')
    checkpoint = torch.load(out / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    results = dict(best_epoch=best_epoch, selection_metric=args.selection_metric,
                   best_validation_value=best, tie_break='latest epoch',
                   test_used_for_selection=False)
    for split, x, y in zip(('train', 'val', 'test'), xs, ys):
        results[split] = evaluate(model, x, y, device, criterion, args,
                                 test_k if split == 'test' else val_k)
    save_json(out / 'metrics.json', results)
    print(f'Finished: best epoch={best_epoch}, output={out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
