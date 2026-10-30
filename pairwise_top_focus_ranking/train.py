import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from data import load_embedding, load_targets, StratifiedActiveBatchSampler
from model import TopFocusedLearningSpace, TopFocusedRankingLoss
from nucb_data import CategoricalUniqueBatchSampler


@torch.no_grad()
def predict(model, x, device, batch=2048):
    model.eval()
    return torch.cat([model(v.to(device)) .cpu() for v in x.split(batch)]).numpy()


def metrics(y, scores, k, nucb=False):
    if k < 1 or k > len(y):
        raise ValueError(f'Invalid K={k} for N={len(y)}')
    order = np.argsort(-scores, kind='stable')[:k]
    positive = np.zeros(len(y), dtype=bool)
    positive[np.argsort(-y, kind='stable')[:k]] = True
    hit = positive[order].mean()
    discount = 1/np.log2(np.arange(2, k+2))
    ideal = discount[:min(k, int(positive.sum()))].sum()
    extra = {}
    if nucb:
        class3 = y == 3
        extra = dict(class3_precision=float(class3[order].mean()),
                     class3_recall=float(class3[np.argsort(-scores, kind='stable')[:int(class3.sum())]].mean()) if class3.any() else 0.)
    return dict(**extra,spearman=float(spearmanr(y,scores).statistic) if np.ptp(y)>0 and np.ptp(scores)>0 else None,
                pearson=float(pearsonr(y,scores).statistic) if np.ptp(y)>0 and np.ptp(scores)>0 else None,
                topk=float(hit), ef=float(hit/positive.mean()) if positive.any() else 0.,
                ndcg=float(np.dot(positive[order],discount)/ideal) if ideal else 0., k=k)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['Amylase','IRED','NucB','TrpB'],required=True)
    for name in ['train-emb','val-emb','test-emb','negative-emb','split-csv','output-dir']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--target-col');p.add_argument('--split-col')
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--negative-batch-size',type=int,default=32)
    p.add_argument('--test-k',type=int)
    p.add_argument('--tau',type=float,default=.5);p.add_argument('--alpha',type=float,default=2.)
    p.add_argument('--temperature',type=float,default=.2);p.add_argument('--inactive-weight',type=float,default=.1)
    p.add_argument('--inactive-margin',type=float,default=0.);p.add_argument('--learning-rate',type=float)
    p.add_argument('--weight-decay',type=float)
    p.add_argument('--pair-sampling-alpha',type=float,default=.5)
    a=p.parse_args();nucb=a.dataset=='NucB'
    if a.epochs<1 or a.negative_batch_size<1:raise ValueError('Epoch and batch counts must be positive')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Use an empty output directory')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device(a.device)
    def load(path):
        x=load_embedding(path).float()
        if x.ndim==3:
            mask=x.abs().sum(-1).ne(0).unsqueeze(-1)
            x=(x*mask).sum(1)/mask.sum(1).clamp_min(1)
        if not len(x) or not torch.isfinite(x).all():raise ValueError('Invalid embeddings')
        return x
    tx,vx,sx,nx=[load(v) for v in [a.train_emb,a.val_emb,a.test_emb,a.negative_emb]]
    ty,vy,sy,_=load_targets((len(tx),len(vx),len(sx)),a.split_csv,a.target_col,a.split_col)
    if len({v.shape[1] for v in [tx,vx,sx,nx]})!=1:raise ValueError('Embedding dimensions differ')
    for y in [ty,vy,sy]:
        if not np.isfinite(y).all():raise ValueError('Invalid targets')
        if nucb and not np.isin(y,[0,1,2,3]).all():raise ValueError('NucB requires classes 0,1,2,3')
    sk=a.test_k or (int((sy==3).sum()) if nucb else 100)
    vk=10
    if not 1<=vk<=len(vy) or not 1<=sk<=len(sy):raise ValueError('Invalid evaluation K or no validation class 3')
    cfg=dict(input_dim=tx.shape[1],dropout=.2 if nucb else .1)
    model=TopFocusedLearningSpace(**cfg).to(device)
    loss=TopFocusedRankingLoss(tau=a.tau,alpha=a.alpha,temperature=a.temperature,inactive_weight=a.inactive_weight,inactive_margin=a.inactive_margin).to(device)
    loss.set_reference_targets(ty)
    if a.learning_rate is None:a.learning_rate=.0003 if nucb else .001
    if a.weight_decay is None:a.weight_decay=.0005 if nucb else .0001
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    sampler=(CategoricalUniqueBatchSampler(ty,batch_size=96,pair_sampling_alpha=a.pair_sampling_alpha,seed=a.seed)
             if nucb else StratifiedActiveBatchSampler(ty,seed=a.seed))
    history=[];best=-float('inf')
    for epoch in range(1,a.epochs+1):
        sampler.set_epoch(epoch-1)
        batches=iter(sampler)
        model.train();values=[]
        for ids in batches:
            ni=torch.randperm(len(nx))[:min(len(nx),a.negative_batch_size)]
            opt.zero_grad(set_to_none=True)
            terms=loss(model(tx[ids].to(device)),torch.as_tensor(ty[ids],device=device),model(nx[ni].to(device)))
            if not torch.isfinite(terms['loss']):raise FloatingPointError('Non-finite loss')
            terms['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            values.append(float(terms['loss'].detach()))
        val=metrics(vy,predict(model,vx,device),vk,nucb)
        state=dict(epoch=epoch,model_state_dict=model.state_dict(),model_config=cfg,arguments=vars(a),validation=val)
        if val['ef']>=best:
            best=val['ef'];torch.save(state,out/'best_model.pt')
        torch.save(state,out/'last_model.pt')
        history.append(dict(epoch=epoch,train_loss=float(np.mean(values)),validation=val))
        print(epoch,history[-1],flush=True)
    state=torch.load(out/'best_model.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    result=dict(dataset=a.dataset,best_epoch=state['epoch'],validation=state['validation'],test=metrics(sy,predict(model,sx,device),sk,nucb),selection='max validation EF@10; ties use latest epoch',test_used_for_selection=False)
    (out/'arguments.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
