#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因果 Transformer 版逐音排键：跟 train_ar.py 同数据同特征，
区别只是把 GRU 换成带因果掩码的 Transformer（每个位置只能看自己和左边）。

用来验证"Transformer 更强"在这个任务上是否成立。
"""
import os
import json
import glob
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ar import NB, DIM, PEB, MAXLEN, feats, load_song, x2g, g2x

SEG = 256


class TFAR(nn.Module):
    def __init__(self, h=192, layers=3, nhead=4):
        super().__init__()
        self.prev = nn.Embedding(NB + 1, PEB)
        self.inp = nn.Linear(DIM + PEB, h)
        self.pos = nn.Parameter(torch.zeros(1, SEG, h))
        nn.init.normal_(self.pos, std=.02)
        enc = nn.TransformerEncoderLayer(h, nhead, h * 2, batch_first=True, dropout=.2)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.o = nn.Linear(h, NB)

    def forward(self, f, pb, pad=None):
        n = f.size(1)
        x = torch.relu(self.inp(torch.cat([f, self.prev(pb)], -1))) + self.pos[:, :n]
        cm = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        x = self.enc(x, mask=cm, src_key_padding_mask=pad)
        return self.o(x)


def seg_songs(paths, limit, seg=SEG):
    """把整曲切成 seg 音的段，段首的 prev 用上一段的最后一音"""
    out = []
    files = sorted(glob.glob(paths + '/*.json'))
    random.seed(0); random.shuffle(files)
    for p in files[:limit]:
        try:
            ns = load_song(p)
        except Exception:
            continue
        if len(ns) < 8:
            continue
        f = feats(ns)
        y = np.array([x2g(z['x_abs']) for z in ns], np.int64)
        for lo in range(0, len(ns), seg):
            hi = min(lo + seg, len(ns))
            if hi - lo < 4:
                continue
            pb = np.full(hi - lo, NB, np.int64)
            pb[1:] = y[lo:hi - 1]
            out.append((f[lo:hi], y[lo:hi], pb))
    return out


def batchify(segs, idx, noise=0.15):
    n = max(len(segs[i][0]) for i in idx)
    B = len(idx)
    F = np.zeros((B, n, DIM), np.float32)
    PB = np.full((B, n), NB, np.int64)
    Y = np.zeros((B, n), np.int64)
    M = np.zeros((B, n), np.float32)
    for j, i in enumerate(idx):
        f, y, pb = segs[i]
        t = len(f)
        F[j, :t] = f; Y[j, :t] = y; M[j, :t] = 1.0; PB[j, :t] = pb
        if noise > 0:
            r = np.random.rand(t) < noise
            if r.any():
                PB[j, :t][r] = np.random.randint(0, NB, int(r.sum()))
            j2 = np.random.rand(t) < noise * 2
            if j2.any():
                PB[j, :t][j2] = np.clip(PB[j, :t][j2] + np.random.randint(-1, 2, int(j2.sum())), 0, NB)
    pad = torch.from_numpy((M < 0.5))
    return torch.from_numpy(F), torch.from_numpy(PB), torch.from_numpy(Y), torch.from_numpy(M), pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pars', default='/home/user/Doubao/chats/38453392661469954/_run_real_traj/abs_full/parsed')
    ap.add_argument('--limit', type=int, default=1500)
    a = ap.parse_args()
    segs = seg_songs(a.pars, a.limit)
    print('段数', len(segs), '总音', sum(len(s[0]) for s in segs), flush=True)
    n = len(segs)
    vidx = list(range(int(n * .95), n)); tridx = list(range(0, int(n * .95)))
    model = TFAR()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=.03)
    BS = 16
    EPOCH = int(os.environ.get('EPOCH', 4))
    for ep in range(EPOCH):
        model.train(); random.shuffle(tridx)
        tot = 0.; nb = 0
        for it in range(0, len(tridx), BS):
            F, PB, Y, M, pad = batchify(segs, tridx[it:it + BS])
            lg = model(F, PB, pad)
            loss = (ce(lg.reshape(-1, NB), Y.reshape(-1)) * M.reshape(-1)).sum() / M.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            if it % 200 == 0:
                print('  ep%d %d/%d loss %.4f' % (ep, it, len(tridx), tot / max(1, nb)), flush=True)
        print('ep%d 训练损失 %.4f' % (ep, tot / max(1, nb)), flush=True)
        model.eval()
        with torch.no_grad():
            vl = 0.; vn = 0; xs = []
            for it in range(0, len(vidx), 64):
                F, PB, Y, M, pad = batchify(segs, vidx[it:it + 64], noise=0)
                lg = model(F, PB, pad)
                loss = (ce(lg.reshape(-1, NB), Y.reshape(-1)) * M.reshape(-1)).sum() / M.sum()
                vl += float(loss); vn += 1
                pred = lg.argmax(-1); xs.append(g2x(pred[M > 0].numpy()))
            x = np.concatenate(xs)
            print('  验证 |x|中位%.0f(真253) <135占%.1f%%(真23.2) >400占%.1f%%(真21.9) loss %.4f' % (
                np.median(np.abs(x)), np.mean(np.abs(x) < 135) * 100, np.mean(np.abs(x) > 400) * 100, vl / vn), flush=True)
        torch.save(model.state_dict(), os.environ.get('CKPT_OUT', 'tfar.pt'))
        print('  已存', os.environ.get('CKPT_OUT', 'tfar.pt'), flush=True)


if __name__ == '__main__':
    main()
