from pathlib import Path

DEFAULTS = {
    'Amylase': dict(split='base_two_to_many', batch_size=96, seed=42,
                    learning_rate=1e-3, weight_decay=1e-4, dropout=.1, grad_clip=1.),
    'IRED': dict(split='two_to_many', batch_size=256, seed=10,
                 learning_rate=1e-3, weight_decay=0., dropout=.1, grad_clip=0.),
    'NucB': dict(split='nucb_two_to_many', batch_size=96, seed=42,
                 learning_rate=3e-4, weight_decay=5e-4, dropout=.2, grad_clip=1.),
    'TrpB': dict(split='trpb_two_to_many', batch_size=96, seed=42,
                 learning_rate=1e-3, weight_decay=1e-4, dropout=.1, grad_clip=1.),
}


def configure(args):
    defaults = DEFAULTS[args.dataset]
    for key, value in defaults.items():
        if key != 'split' and getattr(args, key) is None:
            setattr(args, key, value)
    root = Path(args.project_root).expanduser()
    base = root / 'embeddings' / args.dataset / args.embedding_model
    directory = Path(args.embedding_dir).expanduser() if args.embedding_dir else base / defaults['split']
    suffix = 'mean' if args.mean and args.dataset == 'IRED' else 'aa'
    for part in ('train', 'val', 'test'):
        if getattr(args, part + '_emb') is None:
            setattr(args, part + '_emb', str(directory / f'{part}_{suffix}.pt'))
        if getattr(args, part + '_labels') is None:
            setattr(args, part + '_labels', str(directory / f'{part}_labels.pt'))
    if args.negative_emb is None:
        args.negative_emb = str(directory.parent / 'new_neg' / 'new_neg.pt')
    if args.split_csv is None:
        args.split_csv = str(root / 'data' / args.dataset / (defaults['split'] + '.csv'))
    if args.loss_batch_size is None:
        args.loss_batch_size = args.batch_size if args.dataset == 'IRED' else 96
    return args
