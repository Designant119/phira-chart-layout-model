#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把段级模型套到某一首真谱上，看它给每段排出什么位置（先不生成谱面）"""
import json
import sys
import numpy as np
import torch
from train_seg import SegNet, featurize, g2x, NB, HALF
from build_seg_index import beats_of, fp_of
from seg_detect import detect


def main():
    chart_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/user/Doubao/chats/38453392661469954/_run_real_traj/chart_15831/38153333.json'
    chart = json.load(open(chart_path))
    seq = beats_of(chart)
    bounds = detect(seq)
    m = SegNet()
    m.load_state_dict(torch.load('segnet.pt'))
    m.eval()
    COLW = 2 * HALF / (NB - 1)
    print('段数 %d' % len(bounds))
    allx, allg = [], []
    for si, (lo, hi) in enumerate(bounds):
        row = fp_of(seq, lo, hi, 'x')
        F, y = featurize(row)
        with torch.no_grad():
            lg = m(torch.from_numpy(F[None]))
        k = len(y)
        med_iv = float(np.median(row['iv'])) if row['iv'] else 0.2
        tgt_g = float(np.clip(263 + 245 * (med_iv - 0.094), 250, 300)) / COLW
        gs, prev = [], None
        for t in range(k):
            lp = lg[0, t]
            if prev is not None:
                grid = torch.arange(NB, dtype=torch.float32)
                d = (grid - prev).abs()
                lp = lp - ((d - tgt_g) / max(3.0, tgt_g * 0.45)) ** 2 - torch.clamp(4.0 - d, min=0) * 3
            s = int(torch.multinomial(torch.softmax(lp, dim=-1), 1))
            gs.append(s); prev = s
        px = [round(g2x(g)) for g in gs]
        gx = [round(g2x(g)) for g in y]
        print('#%02d 拍%d 预测 %s' % (si, k, px[:14]))
        print('        真谱 %s' % (gx[:14],))
        allx += px; allg += gx
    xs = np.array(allx); gs2 = np.array(allg)
    d = np.abs(np.diff(xs)); ga = np.abs(np.diff(gs2))
    print('\n整曲对比：|x|中位 %d/%d  中间带 %.1f%%/%.1f%%  步长 %d/%d' % (
        np.median(np.abs(xs)), np.median(np.abs(gs2)),
        100 * (np.abs(xs) < 135).mean(), 100 * (np.abs(gs2) < 135).mean(),
        np.median(d[d > 1]), np.median(ga[ga > 1])))


if __name__ == '__main__':
    main()
