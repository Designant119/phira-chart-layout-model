#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位剩余的重叠：同拍完全同x、同拍很近、跨拍很近"""
import json, sys
from collections import defaultdict
sys.path.insert(0, '.')
from phira_fetch import rpe_time_to_beat


def load(f):
    d = json.load(open(f))
    g = defaultdict(list)
    for li, jl in enumerate(d['judgeLineList']):
        for n in jl.get('notes', []):
            b = round(rpe_time_to_beat(n['startTime']), 3)
            g[b].append({'x': round(n['positionX'], 1), 't': int(n.get('type', 1)), 'li': li})
    return g


def go(tag, f):
    g = load(f)
    same = []
    near = 0
    tot_pairs = 0
    for b, v in g.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                tot_pairs += 1
                d = abs(v[i]['x'] - v[j]['x'])
                if d < 1:
                    same.append((b, v[i], v[j]))
                elif d < 66:
                    near += 1
    print(tag, '同拍配对数 %d | 完全同x %d | 25px内 %d' % (tot_pairs, len(same), near))
    for b, a, c in same[:12]:
        print('   拍%.2f  type%d@%.0f vs type%d@%.0f  线%d/%d' % (
            b, a['t'], a['x'], c['t'], c['x'], a['li'], c['li']))


go('生成', '/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831_gen/v4/generated_chart.json')
go('真谱', '/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831/38153333.json')
