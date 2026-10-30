from pathlib import Path
import sys
import numpy as np
import torch
project = Path(__file__).resolve().parent
sys.path.insert(0, str(project))
from loss import TopFocusListMLE
from data import amylase, trpb, nucb
torch.set_num_threads(1)
for alpha in (0.0, 2.0):
    for m in (0, 5):
        y = torch.tensor([0.0, 2.0, 1.0, 2.0], dtype=torch.float64)
        s = torch.tensor([0.1, 0.2, -0.3, 0.4], dtype=torch.float64, requires_grad=True)
        u = torch.linspace(-0.5, 0.5, m, dtype=torch.float64, requires_grad=True)
        ref = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0])
        c = TopFocusListMLE(ref, alpha, 0.5, 0.2)
        actual = c(s, y, u)
        order = y.sort(descending=True).indices
        weights = []
        terms = []
        for pos, idx in enumerate(order):
            q = (ref <= y[idx]).double().mean()
            w = 1 + alpha * max(float((q - 0.5) / 0.5), 0)
            weights.append(w)
            candidates = torch.cat([s[order[pos:]], u]) / 0.2
            terms.append(w * (torch.logsumexp(candidates, 0) - s[idx] / 0.2))
        expected = len(s) * sum(terms) / sum(weights)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual, (s, u), retain_graph=True, allow_unused=True)
        ge = torch.autograd.grad(expected, (s, u), allow_unused=True)
        torch.testing.assert_close(ga[0], ge[0])
        if m:
            torch.testing.assert_close(ga[1], ge[1])
            assert (ga[1] > 0).all()
            torch.testing.assert_close(actual, c(s, y, u.flip(0)))
print('PASS independent ListMLE values, gradients, inactive ordering')
for mod in (amylase, trpb):
    sampler = mod.StratifiedActiveBatchSampler(np.arange(400), top_count=32, middle_count=32, bottom_count=32)
    b = next(iter(sampler))
    assert len(b) == 96
    assert sum((i >= 360 for i in b)) == 32 and sum((240 <= i < 360 for i in b)) == 32
sampler = nucb.CategoricalUniqueBatchSampler(np.repeat(np.arange(4), 100), batch_size=96)
b = next(iter(sampler))
assert len(b) == len(set(b)) == 96
assert all((sum((i // 100 == c for i in b)) >= (2 if c == 3 else 1) for c in range(4)))
print('PASS stratified and NucB unique sampling constraints')
