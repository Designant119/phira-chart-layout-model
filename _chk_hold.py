#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 hold 是否压住别的音符"""
import json, sys
sys.path.insert(0, '.')
from phira_fetch import rpe_time_to_beat


def t2s(v):
    # [beat, num, den] 形式的 startTime 用 beat 近似比较即可
    return rpe_time_to_beat(v)


def go(tag, f):
    d = json.load(open(f))
    holds = []
    others = []
    for jl in d['judgeLineList']:
        for n in jl.get('notes', []):
            t = int(n.get('type'))
            b0 = t2s(n['startTime'])
            b1 = t2s(n['endTime'])
            if t == 2 and b1 > b0:
                holds.append((n['positionX'], b0, b1))
            else:
                others.append((n['positionX'], b0, t))
    hit = tot = 0
    for hx, b0, b1 in holds:
        for ox, ob, ot in others:
            tot += 1
            if b0 - 1e-6 <= ob <= b1 + 1e-6 and abs(ox - hx) < 80:
                hit += 1
    print(tag, 'hold %d 条，区间内与 hold 横向 80px 内的音符 %d / 候选 %d' % (len(holds), hit, tot))


go('生成', '/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831_gen/v4/generated_chart.json')
go('真谱', '/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831/38153333.json')
