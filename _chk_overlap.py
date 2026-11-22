#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查生成谱面的音符重叠情况"""
import json, sys
from collections import defaultdict
sys.path.insert(0, '.')
from phira_fetch import rpe_time_to_beat


def go(tag, f):
    d = json.load(open(f))
    g = defaultdict(list)
    for jl in d['judgeLineList']:
        for n in jl.get('notes', []):
            b = round(rpe_time_to_beat(n['startTime']), 3)
            x = round(n['positionX'], 2)
            t = int(n.get('type'))
            g[b].append((x, t))

    same = near = tot = 0
    for b, v in g.items():
        ts = [x for x, t in v if t == 1]
        ds = [x for x, t in v if t == 3]
        for dv in ds:
            for tv in ts:
                tot += 1
                gap = abs(dv - tv)
                if gap < 1:
                    same += 1
                elif gap < 40:
                    near += 1
    print(tag, '同拍 tap-drag 配对 %d | 完全同x %d (%.1f%%) | 40px内 %d' % (
        tot, same, 100 * same / tot if tot else 0, near))

    # 相邻拍之间是否叠在一起
    order = sorted(g)
    close = 0
    pairs = 0
    for i in range(len(order) - 1):
        dt = order[i + 1] - order[i]
        if dt > 0.12:
            continue
        for x1, t1 in g[order[i]]:
            for x2, t2 in g[order[i + 1]]:
                pairs += 1
                if abs(x1 - x2) < 25:
                    close += 1
    print(tag, '相邻拍(<0.12s) 音符对 %d | 25px内 %d (%.1f%%)' % (
        pairs, close, 100 * close / pairs if pairs else 0))


go('生成', '/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831_gen/v4/generated_chart.json')
go('真谱', '/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831/38153333.json')
