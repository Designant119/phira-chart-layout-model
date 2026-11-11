#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规则排键 demo：确定性、谱师手法。
- 同刻 k 音：均匀铺开在 [-R, R]（双押 R=225 / 3押 R=450 / 4押 R=600），中心 = 小节流动位置
- 小节中心：乐句级流动（交替 -300/+300 或阶梯）
- 单音：中心附近的锯齿流动（奇数/偶数格错开）
输出 infer json（与模型输出同格式）。"""
import json, sys
from collections import defaultdict
import numpy as np


def rule_chart(notes, bps):
    """notes: 原谱音符（t,type,dur,line_id）；返回 {t: x} 映射（同线同刻只留一个x）。
    简化：本 demo 按"同刻组"排，忽略 line_id（RPE 里同刻不同线 = 同屏多音）。"""
    by_t = defaultdict(list)
    for i, n in enumerate(notes):
        by_t[round(n['t'], 3)].append(i)
    out = {}
    # 小节中心流动：每 4 拍一个中心，交替 -320 / +320（2小节一个周期），也可加阶梯
    for t, idxs in by_t.items():
        bar = int(t * bps / 4.0)
        m = -320.0 if (bar // 2) % 2 == 0 else 320.0
        if (bar % 2) == 1:
            m = -m * 0.6  # 奇数小节回撤
        k = len(idxs)
        if k == 1:
            # 单音：锯齿，半拍格奇偶错开
            half = int((t * bps * 2) % 2)
            x = m + (180.0 if half == 1 else -180.0)
        elif k == 2:
            R = 225.0
            x1, x2 = m - R, m + R
        elif k == 3:
            R = 450.0
            x1, x2, x3 = m - R, m, m + R
        else:
            R = 600.0
            xs_ = np.linspace(m - R, m + R, k)
            xs_ = [float(v) for v in xs_]
        # 越界 clamp
        def cl(v):
            return float(np.clip(v, -650, 650))
        if k == 1:
            out[round(t, 3)] = cl(x)
        elif k == 2:
            out[round(t, 3)] = cl(x1)
            for idx in idxs[1:]:
                out[round(t, 3) + idx * 1e-9] = cl(x2)  # 第二个音近似同 t
        else:
            for j, idx in enumerate(idxs):
                out[round(t, 3) + idx * 1e-9] = cl(xs_[j])
    return out


if __name__ == '__main__':
    fid, infer_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    d = json.load(open(infer_path))
    # 从模型 infer 拿音符骨架（t/type/dur/line_id），x 用规则重排
    notes = d['notes']
    bpm = json.load(open(f'/home/user/Doubao/chats/38445055719679234/phira_charts_dl/bpm/{fid}.json'))['segs'][0][1]
    xmap = rule_chart(notes, bpm / 60.0)
    out_notes = []
    for n in notes:
        key = round(n['t'], 3)
        cand = [k for k in xmap if abs(k - key) < 1e-6]
        x = xmap.get(key, 0.0)
        out_notes.append({"t": n["t"], "type": n.get("type", 1), "x": x,
                          "line_id": n["line_id"], "dur": n.get("dur", 0.0)})
    json.dump({"fid": fid, "mode": "rule", "notes": out_notes}, open(out_path, 'w'))
    print(f'规则排键已写入 {out_path}: {len(out_notes)} 音符')
