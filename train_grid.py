#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网格参数模型：窗级聚合特征 -> gap 分类 / ncol 分类 / center 回归。
对照多数类基线，判断节奏特征里是否藏着网格选择信号。"""
import os, json, time, argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class GridNet(nn.Module):
    def __init__(self, nf, ngap=5, nncol=7, h=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Dropout(.15),
                                  nn.Linear(h, h), nn.GELU())
        self.gap = nn.Linear(h, ngap)
        self.ncol = nn.Linear(h, nncol)
        self.center = nn.Linear(h, 1)

    def forward(self, f):
        h = self.body(f)
        return self.gap(h), self.ncol(h), self.center(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max-train', type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(2)
    d = np.load(a.data, allow_pickle=True)
    F = d['F'].astype(np.float32)
    yg = d['gap'].astype(np.int64)
    yn = d['ncol'].astype(np.int64)
    yc = d['center'].astype(np.float32)
    fid = d['fid'].astype(str)

    split = json.load(open(a.split))
    val, test = set(split['val']), set(split['test'])
    tr = ~np.isin(fid, list(val | test))
    va = np.isin(fid, list(val))
    te = np.isin(fid, list(test))
    tr_idx = np.where(tr)[0]
    if a.max_train:
        tr_idx = np.random.default_rng(0).choice(tr_idx, min(a.max_train, len(tr_idx)), replace=False)

    # 多数类基线
    g_maj = np.bincount(yg[tr_idx]).argmax()
    n_maj = np.bincount(yn[tr_idx]).argmax()
    print('train %d val %d test %d' % (len(tr_idx), va.sum(), te.sum()))
    print('多数类基线: gap %d (%.1f%%) ncol %d (%.1f%%)' % (
        g_maj, 100 * (yg[tr_idx] == g_maj).mean(),
        n_maj, 100 * (yn[tr_idx] == n_maj).mean()))
    print('val 多数类: gap %.1f%% ncol %.1f%%  center MAE %.3f' % (
        100 * (yg[va] == g_maj).mean(), 100 * (yn[va] == n_maj).mean(),
        np.abs(yc[va]).mean()))

    net = GridNet(F.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    ceg = nn.CrossEntropyLoss(); cen = nn.CrossEntropyLoss()

    def ev(idx):
        net.eval(); gok = nok = 0
        cs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                f = torch.from_numpy(F[idx[i:i+256]])
                lg, ln, lc = net(f)
                gok += (lg.argmax(-1).numpy() == yg[idx[i:i+256]]).sum()
                nok += (ln.argmax(-1).numpy() == yn[idx[i:i+256]]).sum()
                cs.append(np.abs(lc.numpy() - yc[idx[i:i+256]]))
        n = len(idx)
        return gok / n, nok / n, np.concatenate(cs).mean()

    for ep in range(a.epochs):
        t0 = time.time(); net.train()
        dl = DataLoader(TensorDataset(torch.from_numpy(F[tr_idx]),
                                      torch.from_numpy(yg[tr_idx]),
                                      torch.from_numpy(yn[tr_idx]),
                                      torch.from_numpy(yc[tr_idx])),
                        batch_size=a.bs, shuffle=True, drop_last=True)
        tg = tn = tc = n = 0
        for f, g, n_, c in dl:
            opt.zero_grad()
            lg, ln, lc = net(f)
            loss = ceg(lg, g) + cen(ln, n_) + 0.5 * nn.functional.mse_loss(lc, c)
            loss.backward(); opt.step()
            tg += float(ceg(lg, g)); tn += float(cen(ln, n_)); tc += float(nn.functional.mse_loss(lc, c)); n += len(f)
        ag, an, ac = ev(np.where(va)[0])
        tg_, tn_, tc_ = ev(np.where(te)[0])
        print('ep%d 损失 gap %.3f ncol %.3f center %.3f | val gap %.1f%% ncol %.1f%% centerMAE %.3f | test gap %.1f%% ncol %.1f%% %.3f | %ds' % (
            ep + 1, tg / n, tn / n, tc / n, 100 * ag, 100 * an, ac, 100 * tg_, 100 * tn_, tc_, time.time() - t0), flush=True)

    torch.save({'model': net.state_dict(), 'nf': F.shape[-1]}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
