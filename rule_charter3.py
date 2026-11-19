#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规则排键v3：沿用 v2 的拍驱动手法，把 4 个宽键位换成真谱统计出来的密网格。
- 网格：11 列，间距 5 格（约 84px），跨度 50 格（约 844px），关于屏幕中心对称
- 单音：按拍奇偶切左半/右半区，半拍决定区内具体列位
- 双押：对称配对（列 i 配 列 10-i）
- 三押：两端 + 正中
用法: python3 rule_charter3.py <chart.json> <out.json> [--seed N]"""
import os, sys, json, argparse, random, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phira_fetch import rpe_time_to_beat

NB = 81
HALF = 675.0
NCOL = 11          # 真谱列数中位
GAP = 5.0          # 真谱间距中位 4-5 格
SWAP_P = 0.25      # 每段换网格概率
SAME_P = 0.176     # 原地同列连敲概率


def g2x(c):
    return -HALF + c / (NB - 1) * 2 * HALF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('chart')
    ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    d = json.load(open(a.chart, encoding='utf-8'))
    notes = []
    for li, jl in enumerate(d['judgeLineList']):
        for ni, n in enumerate(jl.get('notes', [])):
            if n.get('isFake', 0) == 1:
                continue
            notes.append({'b': round(rpe_time_to_beat(n['startTime']), 4),
                          'li': li, 'ni': ni, 'x': 0.0})
    groups = defaultdict(list)
    for r in notes:
        groups[r['b']].append(r)
    order = sorted(groups)

    center = (NB - 1) / 2.0
    half = (NCOL - 1) / 2.0
    cols = [center + (i - half) * GAP for i in range(NCOL)]

    def px(i):
        return g2x(cols[int(max(0, min(NCOL - 1, i)))])

    gbase = GAP
    last_single = None
    for b in order:
        items = groups[b]
        k = len(items)
        beat = int(math.floor(b + 1e-6))
        sub = int(math.floor(b * 2 + 1e-6)) % 2
        if rng.random() < SWAP_P and k == 1:
            gbase = rng.choice([4.0, 4.0, 5.0, 5.0, 6.0])
            cols = [center + (i - half) * gbase for i in range(NCOL)]

        if k == 1:
            r = items[0]
            if last_single is not None and rng.random() < SAME_P:
                r['x'] = px(last_single)
            else:
                side = beat % 2
                if side == 0:
                    i = sub * 2 % 5            # 0..4
                else:
                    i = NCOL - 1 - (sub * 2 % 5)  # 10..6
                r['x'] = px(i)
                last_single = i
        elif k == 2:
            i = (beat % 5)
            pair = [i, NCOL - 1 - i]
            for j, r in enumerate(items[:2]):
                r['x'] = px(pair[j])
        elif k == 3:
            i = (beat % 5)
            trio = [i, 5, NCOL - 1 - i]
            for j, r in enumerate(items[:3]):
                r['x'] = px(trio[j])
        else:
            lo, hi = min(beat % 5, NCOL - 1 - (beat % 5)), max(beat % 5, NCOL - 1 - (beat % 5))
            pos = [int(round(v)) for v in
                   [lo + (hi - lo) * j / (k - 1) for j in range(k)]]
            for j, r in enumerate(items):
                r['x'] = px(pos[j])

    for r in notes:
        d['judgeLineList'][r['li']]['notes'][r['ni']]['positionX'] = round(
            float(max(-HALF, min(HALF, r['x']))), 2)
    json.dump(d, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)

    xs = [r['x'] for r in sorted(notes, key=lambda r: r['b'])]
    import numpy as np
    xs = np.array(xs)
    dx = np.abs(np.diff(xs))
    s = np.sign(np.diff(xs))
    alt = float((s[:-1] * s[1:] < 0).mean()) if len(s) > 1 else 0
    q = np.clip(np.round((xs + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1).astype(int)
    u, c2 = np.unique(q, return_counts=True)
    print('音 %d | 步长中位 %.0f 交替 %.2f 占用 %d 格 单格最多 %d 同列率 %.1f%%' % (
        len(xs), np.median(dx), alt, len(u), c2.max(), 100 * (dx < 20).mean()))


if __name__ == '__main__':
    main()
