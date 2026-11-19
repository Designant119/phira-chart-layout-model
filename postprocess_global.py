#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局槽位硬分配：不再跟着模型的塌缩位置走，按真谱统计的尺度重排整条横向轨迹。
规则：
1. 同拍组整组处理，组内按 GAP 均匀铺开，组中心才是被调度的对象。
2. 组中心推进的步长从真谱相邻步长分布采样（中位 386px），方向优先听模型的相对倾向，模型没态度时按交替率翻向。
3. 撞到真谱活动范围（±520）就反弹，天然形成「一段在一侧打、然后甩到另一侧」的段落感。
4. 最终钳到 ±675 硬边界。"""
import os, sys, json, argparse, random
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phira_fetch import rpe_time_to_beat

HALF = 675.0
SOFT = 520.0        # 真谱 95% 音落在这个范围里，撞到就反弹
GAP = 62.0          # 同拍相邻中心间距
# 真谱相邻步长分位(400 张谱统计)：10% 0 / 25% 225 / 50% 386 / 75% 562 / 90% 788
STEP_Q = [0.0, 225.0, 386.0, 562.0, 788.0]
FLIP = 0.5          # 换向概率，对上真谱交替率 0.50
MODEL_W = 0.10      # 模型组中心偏离超过这个量才认为它有方向意见


def reflect(x, lo, hi):
    """撞墙反弹，来回折，不贴边。"""
    while x > hi or x < lo:
        if x > hi:
            x = 2 * hi - x
        else:
            x = 2 * lo - x
    return x



def sample_step(rng):
    a, b = sorted(rng.sample(STEP_Q, 2))
    return rng.uniform(a, b)


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
                          'x': float(n.get('positionX', 0)), 'li': li, 'ni': ni})

    groups = defaultdict(list)
    for r in notes:
        groups[r['b']].append(r)
    order = sorted(groups)

    prev_c = None      # 上一组中心
    prev_dir = 1
    prev_mx = None     # 上一组模型均值
    for gi, b in enumerate(order):
        items = groups[b]
        mx = float(np.mean([r['x'] for r in items]))
        k = len(items)
        span = (k - 1) * GAP

        if prev_c is None:
            c = float(np.clip(mx, -SOFT, SOFT))
        else:
            # 方向：模型有态度就听它，没态度就按换向概率翻
            if prev_mx is not None and abs(mx - prev_mx) > MODEL_W:
                dirn = 1 if mx > prev_mx else -1
            else:
                dirn = -prev_dir if rng.random() < FLIP else prev_dir
            step = sample_step(rng)
            nx = prev_c + dirn * step
            if abs(nx) > SOFT:
                # 撞墙不折返压步长，直接翻向再走满这一步
                dirn = -dirn
                nx = prev_c + dirn * step
                if abs(nx) > SOFT:
                    nx = float(np.clip(nx, -SOFT, SOFT))
            c = nx
            prev_dir = dirn
            prev_c, prev_mx = c, mx

        # 组内铺开，整体居中在 c，超界就整体平移
        lo = c - span / 2
        if lo < -SOFT:
            c += -SOFT - lo
        hi = c + span / 2
        if hi > SOFT:
            c -= hi - SOFT
        c = float(np.clip(c, -SOFT, SOFT))
        prev_c = c
        items.sort(key=lambda r: r['x'])
        for kk, r in enumerate(items):
            r['out'] = c - span / 2 + kk * GAP

    for r in notes:
        x = float(np.clip(r['out'], -HALF, HALF))
        d['judgeLineList'][r['li']]['notes'][r['ni']]['positionX'] = round(x, 2)

    json.dump(d, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)

    xs = np.array([r['out'] for r in sorted(notes, key=lambda r: r['b'])])
    dx = np.abs(np.diff(xs))
    s = np.sign(np.diff(xs))
    alt = (s[:-1] * s[1:] < 0).mean() if len(s) > 1 else 0
    q = np.clip(np.round((xs + HALF) / (2 * HALF) * 80), 0, 80).astype(int)
    u, cnt = np.unique(q, return_counts=True)
    print('组数 %d 总音 %d' % (len(order), len(notes)))
    print('步长中位 %.0f 交替 %.2f 占用 %d 格 单格最多 %d 近距 %.1f%%' % (
        np.median(dx), alt, len(u), cnt.max(), 100 * (dx < 40).mean()))


if __name__ == '__main__':
    main()
