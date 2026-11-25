#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐音自回归模型的推理桥：给整曲音符，逐音递推出每颗 x。

跟 placement3 一样是"上一个音的位置 -> 下一个音"的递推，区别是 81 格 + x_abs 训练。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ar import ARNet, feats, g2x, x2g, NB


def _load(ckpt='arnet.pt'):
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt)
    m = ARNet()
    m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    m.eval()
    return m


def build(notes, seed=0, ckpt='arnet.pt', temp=1.0):
    """notes: [{'b','t','type','dur'}]；返回 {拍: [x...]}"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = _load(ckpt)
    ns = [n for n in notes if abs(n.get('x', 0)) <= 725]
    ns.sort(key=lambda z: z['b'])
    if not ns:
        return {}
    F = feats(ns)
    ft = torch.from_numpy(F[None])
    out = {}
    prev = NB
    st = None
    with torch.no_grad():
        for i in range(len(ns)):
            lg, st = m(ft[:, i:i + 1], torch.tensor([[prev]]), st)
            p = torch.softmax(lg[0, 0] / temp, -1).numpy()
            b = int(rng.choice(NB, p=p))
            prev = b
            x = float(g2x(b) + rng.uniform(-20, 20))
            out.setdefault(round(ns[i]['b'], 3), []).append(x)
    return out
