#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规则排键v2：固定对称键位，单音流动，双押对称配对。不叠坨。
用法: python3 rule_charter2.py <fid> <infer骨架.json> <out.json>"""
import json, sys
from collections import defaultdict
import numpy as np

KEYS = [-450.0, -150.0, 150.0, 450.0]  # 4个对称键位


def charter(notes, bps):
    """notes: 骨架音符(t,type,dur,line_id)。返回 out notes（x 规则分配）。"""
    # 按时间分组同刻
    by_t = defaultdict(list)
    for n in notes:
        by_t[round(n['t'], 3)].append(n)
    times = sorted(by_t.keys())
    # 单音流动状态：在 KEYS 间按节奏走
    flow_idx = 0  # 当前流动到哪个键位
    # 流动方向：按半拍格奇偶决定左右移动
    last_half = -1
    out = []
    for t in times:
        ns = by_t[t]
        k = len(ns)
        half = int((t * bps * 2) % 2)
        if k == 1:
            # 单音：在 KEYS 间锯齿流动（左右交替）
            # 按半拍格决定：左区(-450/-150) 或 右区(+150/+450)
            side = 0 if (int(t * bps) % 2) == 0 else 1  # 左右区交替
            # 区内再按半拍选键位
            keys_side = KEYS[:2] if side == 0 else KEYS[2:]
            sub = half
            x = keys_side[sub]
        elif k == 2:
            # 双押：对称配对（-450/+450 或 -150/+150），交替用
            pair = 0 if (int(t * bps) % 4) < 2 else 1
            x = [KEYS[pair], KEYS[3 - pair]]
        elif k == 3:
            x = [KEYS[0], KEYS[1] if (int(t*bps)%2)==0 else KEYS[2], KEYS[3]]
        else:
            x = KEYS[:k]
        if k == 1:
            out.append({"t": ns[0]["t"], "type": ns[0].get("type",1), "x": x,
                        "line_id": ns[0]["line_id"], "dur": ns[0].get("dur",0.0)})
        else:
            for j, n in enumerate(ns):
                out.append({"t": n["t"], "type": n.get("type",1), "x": x[j % len(x)],
                            "line_id": n["line_id"], "dur": n.get("dur",0.0)})
    out.sort(key=lambda n: n["t"])
    return out


if __name__ == '__main__':
    fid, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    d = json.load(open(inp))
    notes = d['notes']
    bpm = json.load(open(f'/home/user/Doubao/chats/38445055719679234/phira_charts_dl/bpm/{fid}.json'))['segs'][0][1]
    res = charter(notes, bpm/60.0)
    json.dump({"fid": fid, "mode": "rule2", "notes": res}, open(outp, 'w'), ensure_ascii=False)
    xs = [n['x'] for n in res]
    print(f'{outp}: {len(res)}音 | 键位分布: {sorted(set(xs))}')
