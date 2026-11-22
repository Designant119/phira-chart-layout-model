#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用段级模型给整首谱的每一拍算出位置（单点中心），交给规则展开多押"""
import json
import os
import sys
from collections import defaultdict
import numpy as np
import torch
from train_seg import SegNet, featurize, g2x
from seg_detect import detect
from build_seg_index import fp_of

_MODEL = None


def _get():
    global _MODEL
    if _MODEL is None:
        m = SegNet()
        m.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'segnet.pt'),
                                     map_location='cpu'))
        m.eval()
        _MODEL = m
    return _MODEL


def build(notes, seed=0):
    """notes: [{b,t,type,x,...}]，返回 {拍(3位小数): x_px}"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = defaultdict(list)
    for r in notes:
        g[round(r['b'], 3)].append(r)
    seq = []
    for b in sorted(g):
        v = g[b]
        xsv = [float(r['x']) for r in v]
        seq.append({'b': b, 'n': len(v), 'x': float(np.mean(xsv)),
                    'drag': sum(1 for r in v if r['type'] == 3),
                    'flick': sum(1 for r in v if r['type'] == 4),
                    'span': max(xsv) - min(xsv), 'xs': xsv,
                    'ts': [int(r['type']) for r in v]})
    bounds = detect(seq)
    m = _get()
    out = {}
    for lo, hi in bounds:
        row = fp_of(seq, lo, hi, 'g')
        if row['zero'] > 0.3:       # 多判定线的 x=0 段，模型学不到，交回规则
            continue
        F, y = featurize(row)
        with torch.no_grad():
            lg = m(torch.from_numpy(F[None]))[0][:len(y)]
        for t in range(len(y)):
            prob = torch.softmax(lg[t], dim=-1)
            gidx = int(torch.multinomial(prob, 1))
            out[round(seq[lo + t]['b'], 3)] = float(g2x(gidx))
    return out
