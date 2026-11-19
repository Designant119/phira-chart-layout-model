#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规则排键 v4：整首一套对齐中心的等间距网格，段内只在这套列位上选。
- 基础网格：间距 5 格、13 列、关于屏心对称；少数段换间距/列数但仍对齐中心
- 列位按距中心加权：中列高频、边缘低频
- 单音：左右交替为主，允许同列连敲、同侧连走 2 步
- 双押对称配对，三押两端+中，四押均分
用法: python3 rule_charter4.py <chart.json> <out.json> [--seed N]"""
import os, sys, json, argparse, random, math
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phira_fetch import rpe_time_to_beat

NB = 81
HALF = 675.0
SEG = 24
SAME_CELL_P = 0.22      # 原地同列概率（真谱全量 13%，这首 21%）
FLIP_P = 0.45           # 换向概率
GAP_CHOICES = [5, 5, 5, 4, 6]
NCOL_CHOICES = [13, 13, 13, 12, 14]
SWAP_P = 0.0            # 段间换网格概率：整首一套对齐网格


def g2x(c):
    return -HALF + c / (NB - 1) * 2 * HALF


def build_grid(gap, ncol, center):
    half = (ncol - 1) / 2.0
    return [center + (i - half) * gap for i in range(ncol)]


def col_weights(ncol):
    half = (ncol - 1) / 2.0
    w = [1.0 / (1.0 + abs(i - half) * 0.28) for i in range(ncol)]
    return np.array(w, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--out', required=True)
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
                          'li': li, 'ni': ni, 'out': 0.0})
    groups = defaultdict(list)
    for r in notes:
        groups[r['b']].append(r)
    order = sorted(groups)

    center = (NB - 1) / 2.0
    # 切段
    segs, cur, cnt = [], [], 0
    for b in order:
        k = len(groups[b])
        if cnt + k > SEG and cur:
            segs.append(cur); cur = []; cnt = 0
        cur.append(b); cnt += k
    if cur:
        segs.append(cur)

    gap = rng.choice(GAP_CHOICES)
    ncol = rng.choice(NCOL_CHOICES)
    cols = build_grid(gap, ncol, center)
    i = ncol // 2
    dirn = 1

    for seg in segs:
        if rng.random() < SWAP_P:
            gap = rng.choice(GAP_CHOICES)
            ncol = rng.choice(NCOL_CHOICES)
            span = (ncol - 1) * gap
            if span > 64:
                ncol = max(4, int(64 / gap) + 1)
            cols = build_grid(gap, ncol, center)
            i = ncol // 2
        w = col_weights(ncol)
        half = (ncol - 1) / 2.0

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
            seen, clean = set(), []
            for p in pos:
                p = int(np.clip(p, 0, ncol - 1))
                if p in seen:
                    p = int(np.clip(p + 1, 0, ncol - 1))
                seen.add(p); clean.append(p)
            for kk, r in enumerate(items):
                c = clean[kk] if kk < len(clean) else clean[-1]
                r['out'] = g2x(cols[c])

            if k == 1:
                if rng.random() < SAME_CELL_P:
                    step = 0
                else:
                    step = rng.choice([3, 4, 4, 5, 5, 6])
                if rng.random() < FLIP_P:
                    dirn = -dirn
                ni_ = i + dirn * step
                if ni_ < 0 or ni_ > ncol - 1:
                    dirn = -dirn
                    ni_ = int(np.clip(i + dirn * step, 0, ncol - 1))
                i = ni_

    for r in notes:
        d['judgeLineList'][r['li']]['notes'][r['ni']]['positionX'] = round(
            float(np.clip(r['out'], -HALF, HALF)), 2)
    json.dump(d, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)

    xs = np.array([r['out'] for r in sorted(notes, key=lambda r: r['b'])])
    dx = np.abs(np.diff(xs)); s = np.sign(np.diff(xs))
    alt = float((s[:-1] * s[1:] < 0).mean()) if len(s) > 1 else 0
    q = np.clip(np.round((xs + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1).astype(int)
    u, c2 = np.unique(q, return_counts=True)
    spans = [xs[j:j + SEG].max() - xs[j:j + SEG].min() for j in range(0, max(1, len(xs) - SEG), SEG)]
    print('段 %d 总音 %d' % (len(segs), len(notes)))
    print('步长中位 %.0f 交替 %.2f 占用 %d 格 单格最多 %d 同列率 %.1f%% 24音跨度中位 %.0f' % (
        np.median(dx), alt, len(u), c2.max(), 100 * (dx < 20).mean(), np.median(spans)))


if __name__ == '__main__':
    main()
