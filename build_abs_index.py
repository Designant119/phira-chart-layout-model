#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 x_abs（屏幕绝对坐标）重建段库：多判定线的音符也按真实屏幕位置存。

跟旧 build_seg_index 的区别：
1. 位置取 x_abs，不再是判定线相对坐标，x=0 假象基本消失
2. 每拍存 unfolded 的每颗 x（ans 已经是扁平的，这里额外存每拍的颗数和顺序 x）
"""
import json
import glob
import gzip
import os
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_detect import detect as detect_segments

PARS = '/home/user/Doubao/chats/38453392661469954/_run_real_traj/abs_full/parsed'
HALF = 675.0


def beats_of(chart):
    """按拍归并，位置一律取 x_abs；超出屏幕的裁掉"""
    g = defaultdict(list)
    for n in chart.get('notes', []):
        t = float(n['t'])
        x = float(n.get('x_abs', n.get('x_rel', 0.0)))
        if abs(x) > HALF + 50:
            continue
        g[round(t, 4)].append((x, int(n.get('type', 1)))
                              )
    seq = []
    for t in sorted(g):
        v = g[t]
        xs = [x for x, _ in v]
        ts = [k for _, k in v]
        seq.append({'b': t, 'n': len(v), 'x': float(np.mean(xs)),
                    'drag': sum(1 for k in ts if k == 3),
                    'flick': sum(1 for k in ts if k == 4),
                    'span': max(xs) - min(xs), 'xs': xs, 'ts': ts})
    return seq


def fp_of(seq, lo, hi, fid):
    seg = seq[lo:hi]
    chord = [s['n'] for s in seg]
    dflag = []
    for s in seg:
        if s['drag'] and s['flick']:
            dflag.append(34)
        elif s['drag']:
            dflag.append(3)
        elif s['flick']:
            dflag.append(4)
        else:
            dflag.append(0)
    iv = [round(seg[i + 1]['b'] - seg[i]['b'], 3) for i in range(len(seg) - 1)]
    # 每拍按位置排序后的各颗 x，最多 4 颗
    per = []
    for s in seg:
        xs = sorted(s['xs'])[:4]
        per.append([round(x, 1) for x in xs])
    # 上一拍落在哪（序列结构：真谱 28% 同列、31% 对侧，全靠这一维喂给模型）
    prev = []
    for k in range(len(seg)):
        if k == 0:
            px = sorted(seq[lo - 1]['xs']) if lo - 1 >= 0 else []
        else:
            px = sorted(seg[k - 1]['xs'])
        prev.append(round(float(np.median(px)), 1) if px else None)
    return {
        'fid': fid, 'seg': 0,
        'n': int(sum(chord)), 'beats': len(seg),
        'chord': chord, 'dflag': dflag, 'iv': iv,
        'per': per, 'prev': prev,
        'span': round(max((max(p) - min(p) for p in per if p), default=0.0), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pars', default=PARS)
    ap.add_argument('--out', default='seg_index_abs.jsonl.gz')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(a.pars + '/*.json'))
    if a.limit:
        files = files[:a.limit]
    rows = 0
    with gzip.open(a.out, 'wt') as f:
        for k, p in enumerate(files):
            fid = os.path.basename(p)[:-5]
            try:
                chart = json.load(open(p))
                seq = beats_of(chart)
                bounds = detect_segments(seq)
            except Exception:
                continue
            for si, (lo, hi) in enumerate(bounds):
                if hi - lo < 4 or sum(s['n'] for s in seq[lo:hi]) < 8:
                    continue
                row = fp_of(seq, lo, hi, fid)
                row['seg'] = si
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows += 1
            if k % 1000 == 0:
                print('  %d/%d 首，%d 段' % (k, len(files), rows), flush=True)
    print('完成：%d 段 -> %s' % (rows, a.out))


if __name__ == '__main__':
    main()
