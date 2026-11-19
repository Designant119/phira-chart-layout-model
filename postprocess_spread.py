#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后处理避让：同一拍的几颗音按预测位置排序后均匀铺开，互不重叠。
不改训练，只在生成谱上重排同拍组内位置。"""
import sys, os, json, argparse, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phira_fetch import rpe_time_to_beat

GAP = 62.0          # 同拍相邻音中心间距(px)，略大于音符宽
HALF = 675.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = json.load(open(a.chart, encoding='utf-8'))
    groups = defaultdict(list)   # beat -> [(li, ni, x)]
    for li, jl in enumerate(d['judgeLineList']):
        for ni, n in enumerate(jl.get('notes', [])):
            if n.get('isFake', 0) == 1:
                continue
            b = round(rpe_time_to_beat(n['startTime']), 4)
            groups[b].append((li, ni, float(n.get('positionX', 0))))

    moved = 0
    for b, items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda r: r[2])
        n = len(items)
        center = float(np.mean([r[2] for r in items]))
        span = (n - 1) * GAP
        lo = center - span / 2
        lo = max(-HALF, min(HALF - span, lo))
        for k, (li, ni, _) in enumerate(items):
            nx = round(lo + k * GAP, 2)
            d['judgeLineList'][li]['notes'][ni]['positionX'] = nx
            moved += 1
    json.dump(d, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)
    print('同拍重排音数', moved, '组数', sum(1 for v in groups.values() if len(v) >= 2))

if __name__ == '__main__':
    main()
