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
# 非零步长的跳列数权重（按真谱 15831 实测计数）
STEP_W = [15, 63, 128, 285, 90, 183, 72, 109, 23, 11, 0, 23]
STEP_IDX = list(range(1, 13))


def g2x(c):
    return -HALF + c / (NB - 1) * 2 * HALF


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

    # 切段
    segs, cur, cnt = [], [], 0
    for b in order:
        k = len(groups[b])
        if cnt + k > SEG and cur:
            segs.append(cur); cur = []; cnt = 0
        cur.append(b); cnt += k
    if cur:
        segs.append(cur)

    # 全格位游走：候选列位放开到全部 81 格，不再锁十几列等间距网格
    ncol = NB
    cols = list(range(NB))
    half = (NB - 1) / 2.0
    h = int(round(half))
    COL_STEP = 4  # 原基准列宽换算格步用，跳列权重 → 格步

    # 密度自适应：基准取全量 9746 首谱面分层中位（full_baseline.py）
    # 换向全量恒定 0.70；步长/同列率随密度略变；|x| 中位恒定 225px
    ts = sorted(set(r['t'] for r in notes))
    ivs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    med_iv = float(np.median(ivs)) if ivs else 0.13
    tgt_px = min(max(263.0 + 245.0 * (med_iv - 0.094), 250.0), 300.0)
    step_scale = tgt_px / 270.0
    scp = min(max(0.15 + 0.47 * (med_iv - 0.094), 0.13), 0.22)
    flip_p = 0.70
    chord_follow_p = min(max((0.13 - med_iv) * 4.0, 0.0), 0.8)  # 密集谱多押中心跟随游走
    # 多押半宽（px）按全量分层中位插值：2押随密度略收，3/4押随密度加宽
    # 锚点 稀疏 iv=0.200 / 密集 iv=0.083，全宽中位 → 半宽 px
    td = min(max((0.200 - med_iv) / (0.200 - 0.083), 0.0), 1.0)
    colw = 2 * HALF / (NB - 1)  # 每格 px
    w2_px = 225.0 - 26.5 * td
    w3_px = 253.0 + 58.5 * td
    w4_px = 270.0 + 67.5 * td
    print(f'# 间隔中位 {med_iv:.3f}s 目标步长 {tgt_px:.0f}px 缩放 {step_scale:.2f} 同列概率 {scp:.2f} 换向 {flip_p:.2f} 密度档 {td:.2f}')

    for seg in segs:
        # 单音走位移状态机游走；多押按押型独立选对称宽度
        i = h
        dirn = 1
        last_use = [-1e9] * ncol  # 每列最近一次落点时间，用于生成端避让
        L = R = 0                # 段内左右落点计数，平衡拉力用

        for bt in seg:
            items = groups[bt]
            k = len(items)
            cur_t = items[0]['t']
            cool = 1.0 * med_iv

            def avoid(c):
                # 目标列刚被占用过，就在邻近列里挑冷却最久的，生成时自然错开
                if last_use[c] > cur_t - cool:
                    lo, hi = max(0, c - 2), min(ncol - 1, c + 2)
                    cand = list(range(lo, hi + 1))
                    return min(cand, key=lambda q: last_use[q])
                return c

            # 多押中心偏移：一侧偏多时整组往偏少侧挪
            if L - R >= 2:
                bias = rng.choice([0, 1, 2])
            elif R - L >= 2:
                bias = rng.choice([-2, -1, 0])
            else:
                bias = rng.choice([-2, -1, 0, 0, 1, 2])

            if k == 1:
                if rng.random() < scp:
                    c = i
                else:
                    step = wchoice(rng, STEP_IDX, STEP_W)
                    step = max(1, int(round(step * COL_STEP * step_scale)))
                    # 平衡拉力：本段一侧落太多，就把方向掰去偏少侧
                    if i <= h and L - R >= 2:
                        dirn = 1
                    elif i >= h and R - L >= 2:
                        dirn = -1
                    elif rng.random() < flip_p:
                        dirn = -dirn
                    nc = i + dirn * step
                    if nc < 0:
                        nc = -nc; dirn = 1
                    elif nc > ncol - 1:
                        nc = 2 * (ncol - 1) - nc; dirn = -1
                    c = int(np.clip(nc, 0, ncol - 1))
                    if c == h and rng.random() < 0.6:
                        c = int(np.clip(h + dirn, 0, ncol - 1))
                    c = avoid(c)
                i = c
                if c < h:
                    L += 1
                elif c > h:
                    R += 1
                pos = [c]
            elif k == 2:
                w2 = max(1, int(round(w2_px / colw + rng.choice([-0.5, 0, 0, 0.5]))))
                ch_raw = (i if rng.random() < chord_follow_p else h) + bias
                ch = int(np.clip(ch_raw, w2, ncol - 1 - w2))
                ch = avoid(ch)
                l = int(np.clip(ch - w2, 0, ncol - 1)); r = int(np.clip(ch + w2, 0, ncol - 1))
                pos = [l, r]; i = r if dirn > 0 else l
            elif k == 3:
                w3 = max(2, int(round(w3_px / colw + rng.choice([-0.5, 0, 0, 0.5]))))
                ch_raw = (i if rng.random() < chord_follow_p else h) + bias
                ch = int(np.clip(ch_raw, w3, ncol - 1 - w3))
                ch = avoid(ch)
                l = int(np.clip(ch - w3, 0, ncol - 1)); r = int(np.clip(ch + w3, 0, ncol - 1))
                mid = int(np.clip(ch + rng.choice([-1, 0, 0, 1]), 0, ncol - 1))
                pos = [l, mid, r]; i = h
            else:
                w4 = max(1, int(round(w4_px / colw + rng.choice([-0.5, 0, 0, 0.5]))))
                ch_raw = (i if rng.random() < chord_follow_p else h) + bias
                ch = int(np.clip(ch_raw, w4, ncol - 1 - w4))
                ch = avoid(ch)
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
                if p in seen:
                    for dd in range(1, ncol):
                        if p - dd >= 0 and p - dd not in seen:
                            p = p - dd; break
                        if p + dd < ncol and p + dd not in seen:
                            p = p + dd; break
                seen.add(p); clean.append(p)
            for p in clean:
                last_use[p] = cur_t
                if p < h:
                    L += 1
                elif p > h:
                    R += 1
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

    # 判定线全部静止水平：清掉移动/旋转/速度动画与位移控制，不保留原谱线动效
    static_pos = [{"easing": 1, "pos": 1.0, "x": 0.0}, {"easing": 1, "pos": 1.0, "x": 9999999.0}]
    static_y = [{"easing": 1, "x": 0.0, "y": 1.0}, {"easing": 1, "x": 9999999.0, "y": 1.0}]
    static_size = [{"easing": 1, "size": 1.0, "x": 0.0}, {"easing": 1, "size": 1.0, "x": 9999999.0}]
    static_skew = [{"easing": 1, "skew": 0.0, "x": 0.0}, {"easing": 1, "skew": 0.0, "x": 9999999.0}]
    def _static_ev(val):
        return [{"bezier": 0, "bezierPoints": [0, 0, 0, 0], "easingLeft": 0.0,
                 "easingRight": 1.0, "easingType": 1, "end": val, "endTime": [1, 0, 1],
                 "linkgroup": 0, "start": val, "startTime": [0, 0, 1]}]
    static_layer = [{
        "alphaEvents": _static_ev(255),
        "moveXEvents": _static_ev(0), "moveYEvents": _static_ev(0),
        "rotateEvents": _static_ev(0),
        "speedEvents": _static_ev(10.0)}]
    for l in d['judgeLineList']:
        l['eventLayers'] = static_layer
        l['posControl'] = static_pos
        l['yControl'] = static_y
        l['sizeControl'] = static_size
        l['skewControl'] = static_skew
    json.dump(d, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)

    # 统计口径与 full_baseline 对齐：单音序列、非零步算换向；同列率也在单音序列上
    from collections import defaultdict as _dd
    _g = _dd(list)
    for r in notes:
        _g[round(r['b'], 3)].append(r['out'])
    _singles = [v[0] for v in _g.values() if len(v) == 1]
    if len(_singles) >= 3:
        _s = np.array(_singles, dtype=float)
        _dx = np.abs(np.diff(_s))
        same = float((_dx < 20).mean())
        _d = np.diff(_s); _d = _d[np.abs(_d) > 1e-6]
        alt = float((np.sign(_d[:-1]) * np.sign(_d[1:]) < 0).mean()) if len(_d) >= 2 else 0.5
        step_med = float(np.median(_dx[_dx > 1e-6])) if (_dx > 1e-6).any() else 0.0
    else:
        alt, same, step_med = 0.5, 0.0, 0.0
    xs = np.array([r['out'] for r in sorted(notes, key=lambda r: r['b'])])
    q = np.clip(np.round((xs + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1).astype(int)
    u, c2 = np.unique(q, return_counts=True)
    spans = [xs[j:j + SEG].max() - xs[j:j + SEG].min() for j in range(0, max(1, len(xs) - SEG), SEG)]
    print('段 %d 总音 %d' % (len(segs), len(notes)))
    print('步长中位 %.0f 交替 %.2f 占用 %d 格 单格最多 %d 同列率 %.1f%% 24音跨度中位 %.0f' % (
        step_med, alt, len(u), c2.max(), 100 * same, np.median(spans)))
    _byk = _dd(list)
    for v in _g.values():
        if len(v) >= 2:
            _byk[len(v)].append(max(v) - min(v))
    chord_str = ' '.join('%d押%.0f(n%d)' % (k, np.median(_byk[k]), len(_byk[k])) for k in (2, 3, 4) if _byk[k])
    _all = np.array([r['out'] for r in notes])
    lr = '左%.2f 右%.2f 边%.2f' % ((_all < -100).mean(), (_all > 100).mean(), (np.abs(_all) > 400).mean())
    print('多押', chord_str, '|', lr)


if __name__ == '__main__':
    main()
