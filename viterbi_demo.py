#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Viterbi 解码：把每拍独立的分类分布，串成一条连贯的位置序列"""
import json
import sys
import numpy as np
import torch
from train_seg import SegNet, featurize, g2x, NB, x2g
from build_seg_index import beats_of, fp_of
from seg_detect import detect

PRIOR = 'delta_prior.npz'


def build_trans():
    z = np.load(PRIOR)
    nz = z['nz'].astype(int)
    same_p = float(z['same_p'])
    h = np.bincount(nz, minlength=NB).astype(np.float64)
    h[0] = 0.0
    tot = h[1:].sum()
    p = np.full(NB, 0.5 / (tot + 0.5 * (NB - 1)))
    p[1:] = (h[1:] + 0.5) / (tot + 0.5 * (NB - 1)) * (1.0 - same_p)
    p[0] = same_p
    p = p / p.sum()
    T = np.zeros((NB, NB), dtype=np.float64)
    for i in range(NB):
        for j in range(NB):
            T[i, j] = np.log(p[abs(i - j)] + 1e-12)
    return T


def viterbi(logits, T, use_trans=True):
    lg = logits.numpy().astype(np.float64)
    if lg.max() > 50:
        lg = lg - lg.max()
    n, C = lg.shape
    dp = lg[0].copy()
    backs = []
    for t in range(1, n):
        score = dp[:, None] + (T if use_trans else np.zeros_like(T))
        arg = score.argmax(0)
        dp = score[arg, np.arange(C)] + lg[t]
        backs.append(arg)
    path = [int(dp.argmax())]
    for arg in reversed(backs):
        path.append(int(arg[path[-1]]))
    return np.array(path[::-1])


def main():
    chart_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831/38153333.json'
    chart = json.load(open(chart_path))
    seq = beats_of(chart)
    bounds = detect(seq)
    m = SegNet(); m.load_state_dict(torch.load('segnet.pt')); m.eval()
    T = build_trans()
    px_all, gt_all, prev_all = [], [], []
    for si, (lo, hi) in enumerate(bounds):
        row = fp_of(seq, lo, hi, 'x')
        if row['zero'] > 0.3:
            continue
        F, y = featurize(row)
        with torch.no_grad():
            lg = m(torch.from_numpy(F[None]))[0][:len(y)]
        pv = viterbi(lg, T, True)
        pf = viterbi(lg, T, False)
        px = [round(g2x(g)) for g in pv]
        py = [round(g2x(g)) for g in pf]
        gt = [round(g2x(g)) for g in y]
        px_all += px; gt_all += gt; prev_all += py
        if si < 8:
            print('#%02d' % si)
            print('   viterbi %s' % px[:14])
            print('   逐拍采 %s' % py[:14])
            print('   真谱   %s' % gt[:14])
    for tag, arr in (('viterbi', px_all), ('逐拍采', prev_all), ('真谱', gt_all)):
        a = np.array(arr)
        d = np.abs(np.diff(a)); d = d[d > 1]
        same = (np.abs(np.diff(a)) < 1).mean() * 100
        print('%-6s |x|中位%3.0f 中间带%4.1f%% 步长%3.0f 同列%4.1f%% 用格%d' % (
            tag, np.median(np.abs(a)), 100 * (np.abs(a) < 135).mean(),
            np.median(d) if len(d) else 0, same, len(set(x2g(np.array(arr).clip(-675, 675))) )))


if __name__ == '__main__':
    main()
