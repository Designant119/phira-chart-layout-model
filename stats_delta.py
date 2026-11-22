#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计真谱段内相邻拍的位移分布，给 Viterbi 解码当转移概率"""
import gzip
import json
import sys
import numpy as np
from collections import Counter

NB = 81
HALF = 675.0


def x2g(x):
    return int(np.clip(round((x + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1))


def main():
    path = '/home/user/Doubao/chats/38453392661469954/_run_real_traj/seg_index.jsonl.gz'
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    zero = Counter()      # 位移 0 的次数
    nz = []               # 非零位移绝对值（格）
    rows = 0
    with gzip.open(path, 'rt') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r['zero'] > 0.3:
                continue
            ans = r['ans']; chord = r['chord']
            pos = []
            k = 0
            for c in chord:
                pos.append(np.mean(ans[k:k + c]) if c else ans[min(k, len(ans) - 1)])
                k += c
            g = [x2g(p) for p in pos]
            for i in range(len(g) - 1):
                d = g[i + 1] - g[i]
                if d == 0:
                    zero['0'] += 1
                else:
                    nz.append(abs(d))
            rows += 1
            if cap and rows >= cap:
                break
    nz = np.array(nz)
    tot = sum(zero.values()) + len(nz)
    print('段 %d 位移样本 %d | 同列 %.1f%%' % (rows, tot, 100 * zero['0'] / tot))
    print('非零位移 |d| 格: 中位 %.1f, 25%%=%.1f, 75%%=%.1f, 90%%=%.1f' % (
        np.median(nz), np.percentile(nz, 25), np.percentile(nz, 75), np.percentile(nz, 90)))
    h = Counter(nz.astype(int))
    top = sorted(h.items())[:0]
    print('常见 |d|: ' + ', '.join('%d格:%.1f%%' % (k, 100 * v / len(nz)) for k, v in sorted(h.items())[:14]))
    np.savez('delta_prior.npz', nz=nz, same_p=zero['0'] / tot)
    print('已存 delta_prior.npz')


if __name__ == '__main__':
    main()
