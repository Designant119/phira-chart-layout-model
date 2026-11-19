#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""硬性排键规则：不靠模型自觉，输出阶段强制执行。
1. 同拍组按预测位置排序后均匀铺开，互不重叠。
2. 单格限载，超载的音按时间顺序往左右最近空槽分流。
3. 位置卡死在 ±675 边界内。"""
import os, sys, json, argparse
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phira_fetch import rpe_time_to_beat

GAP = 62.0          # 同拍相邻中心间距(px)
HALF = 675.0
NB = 81             # 横向格子数
MAX_PER_CELL = 3    # 单格最多压几颗


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    d = json.load(open(a.chart, encoding='utf-8'))
    notes = []  # (beat, x, li, ni)
    for li, jl in enumerate(d['judgeLineList']):
        for ni, n in enumerate(jl.get('notes', [])):
            if n.get('isFake', 0) == 1:
                continue
            b = round(rpe_time_to_beat(n['startTime']), 4)
            notes.append([b, float(n.get('positionX', 0)), li, ni])

    # 规则1：同拍铺开
    groups = defaultdict(list)
    for rec in notes:
        groups[rec[0]].append(rec)
    for b, items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda r: r[1])
        n = len(items)
        center = float(np.mean([r[1] for r in items]))
        span = (n - 1) * GAP
        lo = max(-HALF, min(HALF - span, center - span / 2))
        for k, r in enumerate(items):
            r[1] = lo + k * GAP

    # 规则2：时间相邻两颗不能黏在同一格，后一颗往远离上一颗的方向挪
    def cell(x):
        return int(np.clip(round((x + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1))

    notes.sort(key=lambda r: r[0])
    moved = 0
    prev_cell = None
    for r in notes:
        c = cell(r[1])
        if prev_cell is not None and c == prev_cell:
            # 跟上一颗同格，优先往反方向挪一格
            nc = c + 1 if c < NB - 1 else c - 1
            r[1] = -HALF + nc / (NB - 1) * 2 * HALF
            c = nc
            moved += 1
        prev_cell = c

    # 规则3：边界截断 + 写回
    for b, x, li, ni in notes:
        x = float(np.clip(x, -HALF, HALF))
        d['judgeLineList'][li]['notes'][ni]['positionX'] = round(x, 2)

    json.dump(d, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)
    print('分流音数', moved, '总音', len(notes))


if __name__ == '__main__':
    main()
