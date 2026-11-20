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
from phira_fetch import rpe_time_to_beat, parse_chart

NB = 81
HALF = 675.0
SEG = 24
SAME_CELL_P = 0.28      # 基准：原地同列概率（0.183s 间隔谱面口径，运行时按密度自适应）
FLIP_P = 0.53           # 基准：单步换向概率（运行时按密度自适应）
GAP_CHOICES = [5, 5, 5, 4, 6]
NCOL_CHOICES = [13, 13, 13, 12, 14]
SWAP_P = 0.0            # 段间换网格概率：整首一套对齐网格
# 非零步长的跳列数权重（按真谱 15831 实测计数）
STEP_W = [15, 63, 128, 285, 90, 183, 72, 109, 23, 11, 0, 23]
STEP_IDX = list(range(1, 13))


def g2x(c):
    return -HALF + c / (NB - 1) * 2 * HALF


def build_grid(gap, ncol, center):
    half = (ncol - 1) / 2.0
    return [center + (i - half) * gap for i in range(ncol)]


def col_weights(ncol):
    half = (ncol - 1) / 2.0
    w = [1.0 / (1.0 + abs(i - half) * 0.28) for i in range(ncol)]
    return np.array(w, dtype=np.float64)


def wchoice(rng, idx, w):
    w = np.array(w, dtype=np.float64)
    w = w / w.sum()
    return idx[int(np.searchsorted(w.cumsum(), rng.random()))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    d = json.load(open(a.chart, encoding='utf-8'))
    _, psegs = parse_chart(d)

    def b2s(beat):
        total = 0.0
        for i, (sb, bpm) in enumerate(psegs):
            if i + 1 < len(psegs):
                nb = psegs[i + 1][0]
                if beat < nb:
                    return total + (beat - sb) * 60.0 / bpm
                total += (nb - sb) * 60.0 / bpm
            else:
                return total + (beat - sb) * 60.0 / bpm
        return beat * 0.5

    notes = []
    for li, jl in enumerate(d['judgeLineList']):
        for ni, n in enumerate(jl.get('notes', [])):
            if n.get('isFake', 0) == 1:
                continue
            beat = rpe_time_to_beat(n['startTime'])
            notes.append({'b': round(beat, 4), 't': b2s(beat),
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
    half = (ncol - 1) / 2.0
    h = int(round(half))

    # 密度自适应：音越密，位移越小、同列越多（真谱 15831 间隔0.183s→步长338px，15728 间隔0.083s→225px）
    ts = sorted(set(r['t'] for r in notes))
    ivs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    med_iv = float(np.median(ivs)) if ivs else 0.183
    tgt_px = min(max(225.0 + 1130.0 * (med_iv - 0.083), 170.0), 400.0)
    step_scale = tgt_px / 338.0
    scp = min(max(0.28 - (0.183 - med_iv) * 0.9, 0.12), 0.30)
    flip_p = min(max(0.53 + (0.183 - med_iv) * 3.0, 0.45), 0.86)
    print(f'# 间隔中位 {med_iv:.3f}s 目标步长 {tgt_px:.0f}px 缩放 {step_scale:.2f} 同列概率 {scp:.2f} 换向 {flip_p:.2f}')

    for seg in segs:
        # 单音走位移状态机游走；多押按押型独立选对称宽度
        i = h
        dirn = 1

        for bt in seg:
            items = groups[bt]
            k = len(items)
            if k == 1:
                if rng.random() < scp:
                    c = i
                else:
                    step = wchoice(rng, STEP_IDX, STEP_W)
                    step = max(1, int(round(step * step_scale)))
                    if rng.random() < flip_p:
                        dirn = -dirn
                    nc = i + dirn * step
                    if nc < 0:
                        nc = -nc; dirn = 1
                    elif nc > ncol - 1:
                        nc = 2 * (ncol - 1) - nc; dirn = -1
                    c = int(np.clip(nc, 0, ncol - 1))
                    if c == h and rng.random() < 0.6:
                        c = int(np.clip(h + dirn, 0, ncol - 1))
                i = c
                pos = [c]
            elif k == 2:
                ch = int(np.clip(h + rng.choice([-2, -1, 0, 0, 1, 2]), 0, ncol - 1))
                w2 = rng.choice([2, 3, 3, 3, 3, 4, 6])
                l = int(np.clip(ch - w2, 0, ncol - 1)); r = int(np.clip(ch + w2, 0, ncol - 1))
                pos = [l, r]; i = r if dirn > 0 else l
            elif k == 3:
                ch = int(np.clip(h + rng.choice([-2, -1, 0, 0, 1, 2]), 0, ncol - 1))
                w3 = rng.choice([4, 4, 5, 5])
                l = int(np.clip(ch - w3, 0, ncol - 1)); r = int(np.clip(ch + w3, 0, ncol - 1))
                mid = int(np.clip(ch + rng.choice([-1, 0, 0, 1]), 0, ncol - 1))
                pos = [l, mid, r]; i = h
            else:
                ch = int(np.clip(h + rng.choice([-2, -1, 0, 0, 1, 2]), 0, ncol - 1))
                w4 = rng.choice([2, 3, 3, 3, 3, 4, 6])
                l = int(np.clip(ch - w4, 0, ncol - 1)); r = int(np.clip(ch + w4, 0, ncol - 1))
                if rng.random() < 0.6 or h < 2:
                    pos = [l, l, r, r]
                else:
                    am = wchoice(rng, list(range(1, h)), [1.0] * (h - 1))
                    pos = [l, am, ncol - 1 - am, r]
                pos = pos[:k]; i = r if dirn > 0 else l
            seen, clean = set(), []
            for p in pos:
                p = int(np.clip(p, 0, ncol - 1))
                if k == 1 and p in seen:
                    p = int(np.clip(p + 1, 0, ncol - 1))
                seen.add(p); clean.append(p)
            for kk, r in enumerate(items):
                c = clean[kk] if kk < len(clean) else clean[-1]
                r['out'] = g2x(cols[c])

    # hold 重叠避让：时间交叠的长押分居对侧，间距不小于阈值
    from gen_chart_from_model import collect
    rec = collect(d)
    out_map = {(r['li'], r['ni']): r['out'] for r in notes}
    holds = [r for r in rec if r['type'] == 2 and r['dur'] > 0.1]
    holds.sort(key=lambda r: r['t'])

    def ov(A, B):
        return B['t'] < A['t'] + A['dur'] and A['t'] < B['t'] + B['dur']

    MIN_HOLD_GAP = 336.0
    for i in range(len(holds)):
        for j in range(i + 1, len(holds)):
            A, B = holds[i], holds[j]
            if not ov(A, B):
                continue
            x1 = out_map[(A['li'], A['ni'])]
            x2 = out_map[(B['li'], B['ni'])]
            if abs(x1 - x2) < MIN_HOLD_GAP:
                x2 = -x1
                if abs(x2 - x1) < MIN_HOLD_GAP:
                    x2 = -np.sign(x1 if x1 != 0 else 1) * 600.0
                out_map[(B['li'], B['ni'])] = float(np.clip(x2, -HALF, HALF))
    for r in notes:
        r['out'] = out_map[(r['li'], r['ni'])]

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
