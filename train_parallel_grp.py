#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整押并行版：一次出整窗位置，同拍组内用间距约束强制错开。
输入去掉 prev/prev2/delta（并行拿不到）；损失 = 位置分类 CE + 同拍间距正则。"""
import os, json, time, argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]


class ParNet(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, nb=81, nf=14):
        super().__init__()
        self.nb = nb
        self.inp = nn.Linear(nf, d)
        self.pos = nn.Parameter(torch.zeros(1, 128, d))
        nn.init.normal_(self.pos, std=.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=.1, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, nb)

    def forward(self, f):
        h = self.inp(f) + self.pos[:, :f.shape[1]]
        return self.out(self.enc(h))


def to_bin(y, nb):
    return np.clip(np.round((y + 1.0) / 2.0 * (nb - 1)).astype(np.int64), 0, nb - 1)


def expected_x(logits, nb):
    """softmax 期望格子位置，归一化到 -1..1，连续可导。"""
    prob = torch.softmax(logits, -1)
    grid = torch.arange(nb, dtype=logits.dtype, device=logits.device)
    return (prob * grid).sum(-1) / (nb - 1) * 2.0 - 1.0


def group_gap_loss(px, g, m, gap=0.10):
    """同拍组内相邻位置间距小于 gap 的部分罚回来。"""
    B, L = px.shape
    gb = g.unsqueeze(2)
    same = (gb == gb.transpose(1, 2))
    mm = (m.unsqueeze(2) * m.unsqueeze(1)) > 0
    eye = torch.eye(L, dtype=torch.bool, device=px.device)
    same = same & mm & (~eye)
    diff = torch.abs(px.unsqueeze(2) - px.unsqueeze(1))
    pen = torch.relu(gap - diff)
    return (pen * same).sum() / same.sum().clamp(min=1)


def parallel_eval(net, F, Y, M, nb, bs=96):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(F), bs):
            logits = net(torch.from_numpy(F[i:i + bs].astype(np.float32)))
            idx = logits.argmax(-1).numpy()
            x = idx / (nb - 1) * 2.0 - 1.0
            out.append(x)
    P = np.concatenate(out)
    m = M.astype(bool)
    return float(np.mean(np.abs(P[m] - Y[m]) * 675)), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max-train', type=int, default=40000)
    ap.add_argument('--bins', type=int, default=81)
    ap.add_argument('--gap', type=float, default=0.10)
    ap.add_argument('--wgap', type=float, default=0.5)
    ap.add_argument('--resume', default='')
    a = ap.parse_args()

    torch.set_num_threads(2)
    d = np.load(a.data, allow_pickle=True)
    Ffull = d['F']
    F = np.ascontiguousarray(Ffull[:, :, KEEP], dtype=np.float32)
    del Ffull
    Y = d['Y'].astype(np.float32)
    M = d['M'].astype(np.float32)
    G = d['G'].astype(np.int64)
    fid = d['fid'].astype(str)
    NB = a.bins
    YB = to_bin(Y, NB)

    split = json.load(open(a.split))
    val, test = set(split['val']), set(split['test'])
    tr_idx = np.where(~np.isin(fid, list(val | test)))[0]
    va, te = np.isin(fid, list(val)), np.isin(fid, list(test))
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(tr_idx, size=min(a.max_train, len(tr_idx)), replace=False)
    print('windows', len(F), 'feat', F.shape[-1], 'train', len(tr_idx),
          'val', int(va.sum()), 'test', int(te.sum()), 'bins', NB, flush=True)

    net = ParNet(nf=F.shape[-1], nb=NB)
    start_ep = 1
    rpath = a.out[:-3] + '_last.pt'
    if a.resume and os.path.exists(rpath):
        ck = torch.load(rpath, map_location='cpu')
        net.load_state_dict(ck['model'])
        start_ep = int(ck.get('epoch', 0)) + 1
        print('resumed from %s epoch %d' % (rpath, start_ep), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss(reduction='none')
    best = 1e9

    for ep in range(start_ep - 1, a.epochs):
        t0 = time.time()
        net.train()
        dl = DataLoader(
            TensorDataset(torch.from_numpy(F[tr_idx]), torch.from_numpy(YB[tr_idx]),
                           torch.from_numpy(M[tr_idx]), torch.from_numpy(G[tr_idx])),
            batch_size=a.bs, shuffle=True, drop_last=True)
        tot = tot_g = n = 0
        for f, yb, m, g in dl:
            opt.zero_grad()
            logits = net(f)
            ce_loss = (ce(logits.reshape(-1, NB), yb.reshape(-1)).reshape(yb.shape) * m).sum() / m.sum()
            px = expected_x(logits, NB)
            gap_loss = group_gap_loss(px, g, m, gap=a.gap)
            loss = ce_loss + a.wgap * gap_loss
            loss.backward()
            opt.step()
            tot += float(ce_loss) * len(f)
            tot_g += float(gap_loss) * len(f)
            n += len(f)
        vmae, _ = parallel_eval(net, F[va], Y[va], M[va], NB)
        tmae, _ = parallel_eval(net, F[te], Y[te], M[te], NB)
        torch.save({'model': net.state_dict(), 'bins': NB, 'feat': F.shape[-1],
                    'epoch': ep + 1, 'val': float(vmae)}, rpath)
        tag = ''
        if vmae < best:
            best = vmae
            torch.save({'model': net.state_dict(), 'bins': NB, 'feat': F.shape[-1], 'KEEP': KEEP}, a.out)
            tag = ' *saved'
        print('epoch %d ce %.4f gap %.4f val_MAE %.1fpx test_MAE %.1fpx %ds%s' % (
            ep + 1, tot / n, tot_g / n, vmae, tmae, time.time() - t0, tag), flush=True)

    base = float(np.mean(np.abs(Y[M.astype(bool)]) * 675))
    print('baseline(全猜0) %.1fpx' % base, flush=True)


if __name__ == '__main__':
    main()
