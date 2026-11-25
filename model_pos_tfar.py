#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因果 Transformer 的推理桥：逐音递推（每步只能看到自己及左边，prev 用自己的输出）。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ar import NB, feats, g2x
from train_ar_tf import TFAR


def _load(ckpt='tfar3.pt'):
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt)
    m = TFAR()
    m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    m.eval()
    return m


def build(notes, seed=0, ckpt='tfar3.pt', temp=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = _load(ckpt)
    ns = [n for n in notes if abs(n.get('x', 0)) <= 725]
    ns.sort(key=lambda z: z['b'])
    if not ns:
        return {}
    F = torch.from_numpy(feats(ns)[None])
    out = {}
    prev = NB
    with torch.no_grad():
        for i in range(len(ns)):
            # 完整前缀 + 绝对位置编码，跟训练时的位置口径一致（不再截窗口，避免错位）
            lo = max(0, i - 1023)
            lg = m(F[:, lo:i + 1], torch.full((1, i + 1 - lo), prev, dtype=torch.long))
            p = torch.softmax(lg[0, -1] / temp, -1).numpy()
            b = int(rng.choice(NB, p=p))
            prev = b
            x = float(g2x(b) + rng.uniform(-20, 20))
            out.setdefault(round(ns[i]['b'], 3), []).append(x)
    return out


if __name__ == '__main__':
    import json
    import time
    t0 = time.time()
    d = json.load(open('/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831/38153333.json'))
    notes = []
    for ln in d['judgeLineList']:
        for n in ln.get('notes', []):
            st = n['startTime']
            notes.append({'b': st[1] + st[0] * st[2], 't': st[1] + st[0] * st[2],
                          'x': float(n.get('positionX', 0)), 'type': int(n.get('type', 1)), 'dur': 0})
    x = build(notes, seed=1)
    xs = np.array([v for k in x for v in x[k]])
    print('音', len(xs), '|x|中位', round(float(np.median(np.abs(xs)))), '耗时', round(time.time() - t0, 1), 's')
