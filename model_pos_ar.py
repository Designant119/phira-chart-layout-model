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
from train_ar import ARNet, feats, g2x, x2g, NB, HALF


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
    # 屏幕边界：note 宽 10.5 格，摆到 ±675 就半边出框；真谱 99% 的音在 ±450 内
    cap = float(os.environ.get('XCAP', '450'))
    JIT = float(os.environ.get('XJIT', '8'))
    GRID = float(os.environ.get('XGRID', '0'))
    GAPMIN = float(os.environ.get('XGAP', '0'))  # 跨拍禁用 20..GAPMIN 的半叠距离
    GX = np.array([g2x(j) for j in range(NB)])
    last_x = None
    last_dx = 0.0
    lo_g = x2g(-cap) if cap < 675 else 0
    hi_g = x2g(cap) if cap < 675 else NB - 1
    HF = float(os.environ.get('XHF', '0'))
    # 真谱位置分布是中间最高的钟形（9 桶 -675..675，单位 %）
    TB = np.array([0.0, 0.11, 0.12, 0.16, 0.21, 0.13, 0.15, 0.12, 0.0])
    TB = TB / TB.sum()
    bq = np.clip(np.floor((np.array([g2x(i) for i in range(NB)]) + HALF) / (2 * HALF) * 9), 0, 8).astype(int)
    cnt = np.zeros(9, dtype=float)
    m = _load(ckpt)
    ns = [n for n in notes if abs(n.get('x', 0)) <= 725]
    ns.sort(key=lambda z: z['b'])
    if not ns:
        return {}
    F = feats(ns)
    ft = torch.from_numpy(F[None])
    out = {}
    prev = NB
    j_in_beat = 0
    st = None
    with torch.no_grad():
        for i in range(len(ns)):
            b_cur = round(ns[i]['b'], 3)
            b_cur_prev = round(ns[i - 1]['b'], 3) if i > 0 else None
            ft[:, i, 11] = last_dx / HALF  # 用自己刚走过的位移，不泄漏原谱
            lg, st = m(ft[:, i:i + 1], torch.tensor([[prev]]), st)
            p = torch.softmax(lg[0, 0] / temp, -1).numpy()
            if cap < 675:
                p[:lo_g] = 0.0
                p[hi_g + 1:] = 0.0
                s = p.sum()
                p /= s if s > 0 else 1.0
            if GAPMIN > 0 and last_x is not None and b_cur_prev is not None and b_cur != b_cur_prev:
                d = np.abs(GX - last_x)
                p[(d > 20) & (d < GAPMIN)] = 0.0
                s2 = p.sum()
                p /= s2 if s2 > 0 else 1.0
            if HF > 0 and i > 0:
                # 配额校正：已超配的桶压低，欠配的桶抬高，把整曲分布拉回真谱钟形
                used = cnt / max(cnt.sum(), 1.0)
                ratio = (TB + 1e-4) / (used + 1e-4)
                wgt = np.power(ratio[bq], HF)
                p = p * wgt
                s = p.sum()
                p /= s if s > 0 else 1.0
            b = int(rng.choice(NB, p=p))
            cnt[bq[b]] += 1
            if i > 0 and b_cur == round(ns[i - 1]['b'], 3) and j_in_beat > 0:
                # 同拍第二颗起：用刚才那颗的镜像位置喂给模型，逼它往对侧摆，
                # 真谱双押跨度 437，顺着递推只会 270
                prev = int(np.clip(NB - 1 - b, 0, NB))
            else:
                prev = b
            j_in_beat = j_in_beat + 1 if (i > 0 and b_cur == round(ns[i - 1]['b'], 3)) else 0
            x = float(g2x(b) + rng.uniform(-JIT, JIT))
            if i > 0 and b_cur != b_cur_prev:
                last_dx = x - last_x
            last_x = x
            if GRID > 0:
                x = float(np.clip(np.round(x / GRID) * GRID, -cap, cap))
            out.setdefault(b_cur, []).append(x)
    return out
