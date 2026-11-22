#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""谱面自动分段：在音符序列上算局部特征，找特征突变点切段。

特征窗口不按固定音数切，而是看每一小段"手感"变了没有：
多押密度、drag/flick 占比、时间密度、横向步长、换向率。
"""
import json
import sys
import argparse
import numpy as np
sys.path.insert(0, '.')
from phira_fetch import rpe_time_to_beat
from collections import defaultdict

W = 8          # 局部窗口音数
MIN_SEG = 16   # 一段最少音数


def beat_notes(path):
    d = json.load(open(path))
    raw = []
    for li, jl in enumerate(d['judgeLineList']):
        for n in jl.get('notes', []):
            if n.get('isFake', 0) == 1:
                continue
            b = rpe_time_to_beat(n['startTime'])
            raw.append((b, float(n['positionX']), int(n.get('type', 1))))
    raw.sort(key=lambda r: r[0])
    # 同拍聚合成一个"拍"
    beats = defaultdict(list)
    for b, x, t in raw:
        beats[round(b, 3)].append((x, t))
    order = sorted(beats)
    seq = []
    for b in order:
        v = beats[b]
        xs = [x for x, _ in v]
        ts = [t for _, t in v]
        seq.append({'b': b, 'n': len(v), 'x': float(np.mean(xs)),
                    'drag': sum(1 for t in ts if t == 3),
                    'flick': sum(1 for t in ts if t == 4),
                    'span': max(xs) - min(xs)})
    return seq


def feats(seq, lo, hi):
    """[lo,hi) 拍区间的手感特征"""
    seg = seq[lo:hi]
    if len(seg) < 2:
        return np.zeros(6)
    ns = np.array([s['n'] for s in seg], dtype=float)
    dt = np.array([seq[i + 1]['b'] - seq[i]['b'] for i in range(lo, hi - 1)], dtype=float)
    dc = np.array([abs(seq[i + 1]['x'] - seq[i]['x']) for i in range(lo, hi - 1)], dtype=float)
    d = np.diff([s['x'] for s in seg])
    d = d[np.abs(d) > 1e-6]
    alt = float((np.sign(d[:-1]) * np.sign(d[1:]) < 0).mean()) if len(d) >= 2 else 0.5
    return np.array([
        ns.mean(),                                    # 平均每拍音数
        sum(s['drag'] for s in seg) / max(1, ns.sum()),  # drag 占比
        sum(s['flick'] for s in seg) / max(1, ns.sum()),  # flick 占比
        np.median(dt) if len(dt) else 0.0,             # 时间密度
        np.median(dc) if len(dc) else 0.0,             # 横向步长
        alt,                                          # 换向率
    ])


def detect(seq):
    n = len(seq)
    if n < MIN_SEG * 2:
        return [(0, n)]
    # 每个位置的左右窗口特征差
    score = np.zeros(n)
    for i in range(W, n - W):
        a = feats(seq, i - W, i)
        b = feats(seq, i, i + W)
        # 各特征量纲不同，先按整体尺度归一
        score[i] = float(np.sum(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-6)))
    # 找局部极大且高于阈值的切点
    cuts = []
    last = 0
    for i in range(W, n - W):
        if score[i] < 0.35:
            continue
        lo, hi = max(W, i - W), min(n - W, i + W)
        if score[i] >= score[lo:hi].max() - 1e-9 and i - last >= MIN_SEG:
            cuts.append(i)
            last = i
    cuts.append(n)
    bounds = []
    prev = 0
    for c in cuts:
        if c - prev >= MIN_SEG:
            bounds.append((prev, c))
            prev = c
    if prev < n:
        if bounds and n - prev < MIN_SEG:
            bounds[-1] = (bounds[-1][0], n)
        else:
            bounds.append((prev, n))
    return bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--top', type=int, default=40)
    a = ap.parse_args()
    seq = beat_notes(a.chart)
    bounds = detect(seq)
    print('总拍 %d，自动切成 %d 段' % (len(seq), len(bounds)))
    print('%-4s %-8s %-8s %-6s %-5s %-5s %-6s %-7s %-5s %-5s' % (
        '#', '起始拍', '音数', '拍数', '每拍音', 'drag%', 'flick%', '间隔', '步长', '换向'))
    for i, (lo, hi) in enumerate(bounds[:a.top]):
        f = feats(seq, lo, hi)
        ncount = sum(s['n'] for s in seq[lo:hi])
        print('%-4d %-8.1f %-8d %-6d %-5.2f %-5.1f %-6.1f %-7.3f %-5.0f %.2f' % (
            i, seq[lo]['b'], ncount, hi - lo, f[0], 100 * f[1], 100 * f[2], f[3], f[4], f[5]))


if __name__ == '__main__':
    main()
