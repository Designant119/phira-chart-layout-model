#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 模型的推理桥：给整曲音符，返回每拍该落几颗、各在哪。

跟旧 model_pos 的区别：返回的是"每拍一组 x"，不是单点。多押不拆开、交给模型，
规则只在模型没覆盖的段（多判定线、扫线 drag 段）补。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_detect import detect as detect_segments
from build_abs_index import beats_of, fp_of
from train_seg2 import SegNet2, MAXK, NB, HALF, MAXLEN, featurize, g2x

_MODEL = None


def _load(ckpt='segnet2.pt'):
    global _MODEL
    if _MODEL is None:
        m = SegNet2()
        m.load_state_dict(torch.load(ckpt, map_location='cpu'))
        m.eval()
        _MODEL = m
    return _MODEL


def build(notes, seed=0, ckpt='segnet2.pt'):
    """notes: [{'b','x','type'}...] 已解析的整曲；返回 {拍: [x1, x2, ...]}"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _load(ckpt)
    # 拟一个 chart 结构给 beats_of 用
    chart = {'notes': [{'t': n['b'], 'x_abs': n['x'], 'type': n['type']}
                       for n in notes if abs(n.get('x', 0)) <= HALF + 50]}
    seq = beats_of(chart)
    bounds = detect_segments(seq)
    out = {}
    with torch.no_grad():
        for lo, hi in bounds:
            if hi - lo < 4:
                continue
            r = fp_of(seq, lo, hi, 'gen')
            if r['beats'] > MAXLEN:
                continue
            F, _, _ = featurize(r)
            x = torch.from_numpy(F[None])
            lg = model(x, x[:, :, 9])
            prob = torch.softmax(lg[0], dim=-1)  # (T,4,81)
            samp = torch.multinomial(prob.reshape(-1, NB), 1).reshape(-1, MAXK)
            for i in range(len(seq[lo:hi])):
                b = round(seq[lo + i]['b'], 3)
                k = int(min(seq[lo + i]['n'], MAXK))
                xs = sorted(float(g2x(int(samp[i, j]))) for j in range(k))
                # 同拍相邻靠太近（note 宽约 177 坐标）就合并成一个点，真谱里重合本来就是常规写法
                merged = []
                for x in xs:
                    if merged and x - merged[-1] < 177.0:
                        merged[-1] = (merged[-1] + x) / 2.0
                    else:
                        merged.append(x)
                out[b] = merged
    return out
