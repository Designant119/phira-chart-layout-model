#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按真谱的网格规律重排键位：
- 每 24 音一段，段内选一个等间距列网格（间距 3-7 格，列数按真谱分布抽，关于屏幕中心对称）
- 单颗音在列位之间跳，17.6% 原地同列连敲，59% 换向
- 同拍多押关于中心对分，占两端列位
- 段与段之间换网格"""
import os, sys, json, argparse, random
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phira_fetch import rpe_time_to_beat

NB = 81
HALF = 675.0
SEG = 24            # 每段音数
SAME_CELL_P = 0.28 # 原地同列概率
FLIP_P = 0.59       # 换向概率
GAP_CHOICES = [3, 4, 5, 5, 5, 6, 6, 7]   # 间距档(格)，4 格最常见
NCOL_CHOICES = [8, 9, 10, 10, 11, 11, 12, 12, 13, 14]


def g2x(c):
    """格坐标 -> px"""
    return -HALF + c / (NB - 1) * 2 * HALF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    d = json.load(open(a.chart, encoding='utf-8'))
    notes = []
    for li, jl in enumerate(d['judgeLineList']):
        for ni, n in enumerate(jl.get('notes', [])):
            if n.get('isFake', 0) == 1:
                continue
            notes.append({'b': round(rpe_time_to_beat(n['startTime']), 4),
                          'li': li, 'ni': ni, 'out': 0.0})
    groups = defaultdict(list)
    for r in notes:
        groups[r['b']].append(r)
    order = sorted(groups)

    center = (NB - 1) / 2.0
    base_gap = rng.choice(GAP_CHOICES)
    base_ncol = rng.choice(NCOL_CHOICES)
    # 按音序切段（同拍组不拆开更自然，这里按组数切，约 SEG 音一段）
    segs = []
    cur = []
    cnt = 0
    for b in order:
        k = len(groups[b])
        if cnt + k > SEG and cur:
            segs.append(cur)
            cur = []
            cnt = 0
        cur.append(b)
        cnt += k
    if cur:
        segs.append(cur)

    for seg in segs:
        if rng.random() < 0.25:
            gap = rng.choice(GAP_CHOICES)
            ncol = rng.choice(NCOL_CHOICES)
        else:
            gap, ncol = base_gap, base_ncol
        span = (ncol - 1) * gap
        if span > 62:
            ncol = max(4, int(62 / gap) + 1)
            span = (ncol - 1) * gap
        half = (ncol - 1) / 2.0
        i = int(round(rng.uniform(0, ncol - 1)))
        dirn = 1

        for b in seg:
            items = groups[b]
            k = len(items)
            if k == 1:
                pos = [i]
            elif k == 2:
                pos = [i, ncol - 1 - i]
            elif k == 3:
                pos = [i, int(round(half)), ncol - 1 - i]
            else:
                lo, hi = min(i, ncol - 1 - i), max(i, ncol - 1 - i)
                pos = [int(round(v)) for v in np.linspace(lo, hi, k)]
            # 去重后兜底
            seen = set()
            clean = []
            for p in pos:
                p = int(np.clip(p, 0, ncol - 1))
                if p in seen:
                    p = int(np.clip(p + 1, 0, ncol - 1))
                seen.add(p)
                clean.append(p)
            for kk, r in enumerate(items):
                c = clean[kk] if kk < len(clean) else clean[-1]
                r['out'] = g2x(center + (c - half) * gap)

            # 走向下一组
            if k == 1:
                if rng.random() < SAME_CELL_P:
                    step = 0
                else:
                    step = rng.choice([2, 3, 3, 4, 4, 5, 5, 6])
                if rng.random() < FLIP_P:
                    dirn = -dirn
                ni_ = i + dirn * step
                if ni_ < 0 or ni_ > ncol - 1:
                    dirn = -dirn
                    ni_ = i + dirn * step
                    ni_ = int(np.clip(ni_, 0, ncol - 1))
                i = ni_

    for r in notes:
        x = float(np.clip(r['out'], -HALF, HALF))
        d['judgeLineList'][r['li']]['notes'][r['ni']]['positionX'] = round(x, 2)

    json.dump(d, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)

    xs = np.array([r['out'] for r in sorted(notes, key=lambda r: r['b'])])
    dx = np.abs(np.diff(xs))
    s = np.sign(np.diff(xs))
    alt = (s[:-1] * s[1:] < 0).mean() if len(s) > 1 else 0
    q = np.clip(np.round((xs + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1).astype(int)
    u, cnt2 = np.unique(q, return_counts=True)
    spans = [xs[i:i + SEG].max() - xs[i:i + SEG].min() for i in range(0, max(1, len(xs) - SEG), SEG)]
    print('段 %d 总音 %d' % (len(segs), len(notes)))
    print('步长中位 %.0f 交替 %.2f 占用 %d 格 单格最多 %d 同列率 %.1f%% 24音跨度中位 %.0f' % (
        np.median(dx), alt, len(u), cnt2.max(), 100 * (dx < 20).mean(), np.median(spans)))


if __name__ == '__main__':
    main()
