#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给全量真谱的每个段打"节奏指纹"，建成可检索的库。

指纹只用时间维度（几拍、每拍几颗、哪拍是 drag/flick、拍子快慢），
不带任何位置信息 —— 位置只存在库侧的 answers 里，等匹配上了再借。
"""
import json
import glob
import gzip
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
from phira_fetch import rpe_time_to_beat
from seg_detect import detect as detect_segments

PARS = '/home/user/Doubao/chats/38453392661469954/_run_real_traj/phira_all/parsed'


def beats_of(chart):
    """把一个谱面压成有序的"拍"序列，每拍记全部音符的 x 和类型
    兼容两种格式：RPE 原始谱（judgeLineList）和已解析（notes 扁平）"""
    g = defaultdict(list)
    if 'judgeLineList' in chart:
        for jl in chart['judgeLineList']:
            for n in jl.get('notes', []):
                if n.get('isFake', 0) == 1:
                    continue
                b = round(rpe_time_to_beat(n['startTime']), 3)
                g[b].append((float(n['positionX']), int(n.get('type', 1)))
                            )
    else:
        for n in chart.get('notes', []):
            b = round(float(n['t']), 3)
            g[b].append((float(n['x']), int(n.get('type', 1)))
                        )
    order = sorted(g)
    seq = []
    for b in order:
        v = g[b]
        xs = [x for x, _ in v]
        ts = [t for _, t in v]
        seq.append({'b': b, 'n': len(v), 'x': float(np.mean(xs)),
                    'drag': sum(1 for t in ts if t == 3),
                    'flick': sum(1 for t in ts if t == 4),
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
    xs = [x for s in seg for x in s['xs']]
    zero_rate = float(np.mean([abs(x) < 1 for x in xs])) if xs else 1.0
    return {
        'fid': fid,
        'n': int(sum(chord)), 'beats': len(seg),
        'chord': chord, 'dflag': dflag,
        'iv': iv,
        'zero': round(zero_rate, 3),
        'ans': [round(x, 1) for x in xs],
        'span': round(max(xs) - min(xs), 1) if xs else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='seg_index.jsonl.gz')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(PARS + '/*.json'))
    if a.limit:
        files = files[:a.limit]
    rows = 0
    with gzip.open(a.out, 'wt') as f:
        for k, p in enumerate(files):
            fid = p.split('/')[-1][:-5]
            try:
                chart = json.load(open(p))
            except Exception:
                continue
            try:
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
    print('完成：%d 段 → %s' % (rows, a.out))


if __name__ == '__main__':
    main()
