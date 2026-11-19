#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""位移分类版：预测「相对上一颗音的横向位移」，离散成格子做分类，不再回归绝对值。"""
import os, json, time, argparse, math
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class DeltaNet(nn.Module):
    def __init__(self, d=96, layers=2, heads=4, nb=81, nf=14):
        super().__init__()
        self.nb = nb
        self.inp = nn.Linear(nf, d)
        self.pos = nn.Parameter(torch.zeros(1, 128, d))
        nn.init.normal_(self.pos, std=.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=.1, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, nb)

    def forward(self, f, mask=None):
        h = self.inp(f) + self.pos[:, :f.shape[1]]
        h = self.enc(h, mask=mask)
        return self.out(h)


def to_bin(delta, nb):
    idx = np.round((delta + 1.0) / 2.0 * (nb - 1)).astype(np.int64)
    return np.clip(idx, 0, nb - 1)


def bin_to_delta(idx, nb):
    return idx / (nb - 1) * 2.0 - 1.0


def ar_eval(net, F, Y, M, nb, sample=False, seed=0, guard=False, prior=None, alpha=0.0, scale=1.0):
    """自回归：整窗只有第一颗给真值，其余全靠自己推出来的位移累加。
    guard=True 加边界物理约束：贴近边缘不准再往外走（手跑到轨道头必须掉头）。
    prior: 81 维真谱位置分布（概率），alpha 控制往真谱常用区域收的力度。
    scale: 位移幅度缩放，<1 收得更紧。"""
    net.eval()
    rng = np.random.default_rng(seed)
    preds = []
    edges = bin_to_delta(np.arange(nb), nb)
    lp = None
    if prior is not None and alpha > 0:
        pr = np.asarray(prior, dtype=np.float64)
        pr = pr / max(pr.sum(), 1e-12)
        lp = torch.from_numpy(np.log(np.clip(pr, 1e-8, None))).float()
    with torch.no_grad():
        for i in range(0, len(F), 128):
            f = torch.from_numpy(F[i:i + 128].astype(np.float32).copy())
            b, n = f.shape[0], 128
            x = torch.zeros(b, n)
            for k in range(n):
                logits = net(f)[:, k]
                if lp is not None:
                    base = torch.clamp((x[:, k - 1] if k >= 1 else torch.zeros(b))[:, None] + edges[None, :], -1, 1)
                    bq = torch.clamp(torch.round((base + 1) / 2 * (nb - 1)), 0, nb - 1).long()
                    logits = logits + alpha * lp[bq]
                if guard and k >= 1:
                    cur = x[:, k - 1].numpy()
                    nxt = cur[:, None] + edges[None, :] * scale
                    bad = (nxt > 1.03) | (nxt < -1.03)
                    allbad = bad.all(axis=1, keepdims=True)
                    bad = bad & (~allbad)
                    logits = logits.masked_fill(torch.from_numpy(bad), -30.0)
                if sample:
                    p = torch.softmax(logits.reshape(-1, nb), -1).numpy()
                    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
                    idx = torch.from_numpy(np.array([rng.choice(nb, p=p[j]) for j in range(p.shape[0])]))
                else:
                    idx = logits.argmax(-1)
                d = bin_to_delta(idx.numpy(), nb) * scale
                x[:, k] = torch.clamp((x[:, k - 1] if k >= 1 else 0.0) + torch.from_numpy(d).float(), -1.0, 1.0)
                if k >= 1:
                    f[:, k, 8] = x[:, k - 1]
                    f[:, k, 9] = x[:, k - 2] if k >= 2 else 0.0
                    f[:, k, 10] = f[:, k, 8] - f[:, k, 9]
            preds.append(x.numpy())
    P = np.concatenate(preds)
    m = M.astype(bool)
    return float(np.mean(np.abs(P[m] - Y[m]) * 675)), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max-train', type=int, default=40000)
    ap.add_argument('--bins', type=int, default=81)
    a = ap.parse_args()

    torch.set_num_threads(2)
    d = np.load(a.data, allow_pickle=True)
    F = d['F'].astype(np.float32)
    Y = d['Y'].astype(np.float32)
    M = d['M'].astype(np.float32)
    fid = d['fid'].astype(str)

    D = np.zeros_like(Y)
    D[:, 0] = Y[:, 0]
    D[:, 1:] = Y[:, 1:] - Y[:, :-1]
    NB = a.bins
    DIDX = to_bin(D, NB)

    split = json.load(open(a.split))
    val = set(split['val'])
    test = set(split['test'])
    tr = ~np.isin(fid, list(val | test))
    va = np.isin(fid, list(val))
    te = np.isin(fid, list(test))
    tr_idx = np.where(tr)[0]
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(tr_idx, size=min(a.max_train, len(tr_idx)), replace=False)
    print('windows', len(F), 'train', len(tr_idx), 'val', int(va.sum()), 'test', int(te.sum()), 'bins', NB, flush=True)

    net = DeltaNet(nb=NB, nf=F.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    tmask = torch.triu(torch.full((128, 128), float('-inf')), diagonal=1)
    ce = nn.CrossEntropyLoss(reduction='none')

    best = 1e9
    for ep in range(a.epochs):
        t0 = time.time()
        net.train()
        dl = DataLoader(
            TensorDataset(torch.from_numpy(F[tr_idx]), torch.from_numpy(DIDX[tr_idx]), torch.from_numpy(M[tr_idx])),
            batch_size=a.bs, shuffle=True, drop_last=True)
        tot = n = 0
        for f, yb, m in dl:
            opt.zero_grad()
            logits = net(f, mask=tmask)
            loss = (ce(logits.reshape(-1, NB), yb.reshape(-1)).reshape(yb.shape) * m).sum() / m.sum()
            loss.backward()
            opt.step()
            tot += float(loss) * len(f)
            n += len(f)
        vmae, _ = ar_eval(net, F[va], Y[va], M[va], NB)
        tmae, _ = ar_eval(net, F[te], Y[te], M[te], NB)
        tag = ''
        torch.save({'model': net.state_dict(), 'bins': NB, 'nf': F.shape[-1]}, a.out + '_last.pt')
        if vmae < best:
            best = vmae
            torch.save({'model': net.state_dict(), 'bins': NB, 'nf': F.shape[-1]}, a.out)
            tag = ' *saved'
        print('epoch %d loss %.4f val_AR_MAE %.1fpx test_AR_MAE %.1fpx %ds%s' % (
            ep + 1, tot / n, vmae, tmae, time.time() - t0, tag), flush=True)

    base = float(np.mean(np.abs(Y[M.astype(bool)]) * 675))
    print('baseline(全猜0) %.1fpx' % base, flush=True)


if __name__ == '__main__':
    main()
