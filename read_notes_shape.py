#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐音读谱：全部谱、全部 note 逐一处理，把每 24 颗音横排成一行形状，
位置落到 81 列格子上，用符号标类型。输出形状文本供人工逐行阅读。"""
import os, sys, json, glob, argparse
import numpy as np

NB = 81
HALF = 675.0
CH = {1: 'o', 2: 'H', 3: 'D', 4: 'F'}


def cell(x):
    return int(np.clip(round((x + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1))


def read_notes(fp, ang=5.0):
    d = json.load(open(fp))
    out = []
    for n in d.get('notes', []):
        th = ((float(n.get('theta', 0.0) or 0.0) + 180) % 360) - 180
        te = ((float(n.get('theta_end', th) or th) + 180) % 360) - 180
        if abs(th) > ang or abs(te) > ang:
            continue
        out.append((float(n['t']), float(n.get('x', 0.0)),
                    int(n.get('type', 1)), float(n.get('dur', 0.0))))
    out.sort(key=lambda r: r[0])
    return out


def shape_of(notes):
    """返回该段形状摘要：跨度/换向/同列重复/最大步长。"""
    if len(notes) < 2:
        return None
    t = np.array([r[0] for r in notes])
    x = np.array([r[1] for r in notes])
    dx = np.diff(x)
    same_t = np.diff(t) < 1e-6
    step = np.abs(dx)
    dirs = np.sign(dx)
    flips = 0
    for i in range(1, len(dirs)):
        if dirs[i] != 0 and dirs[i - 1] != 0 and dirs[i] != dirs[i - 1]:
            flips += 1
    return {
        'span': float(x.max() - x.min()),
        'med_step': float(np.median(step)),
        'max_step': float(step.max()),
        'flip': flips / max(1, len(dirs) - 1),
        'hold_same': float((step < 20).mean()),
        'chord_ratio': float(same_t.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shape-out', required=True)
    ap.add_argument('--per', type=int, default=24)
    ap.add_argument('--max-charts', type=int, default=0)
    ap.add_argument('--charts-out', default='', help='只对这些 fid 输出完整形状')
    ap.add_argument('--auto-pick', type=int, default=0, help='按音数分档各挑几张输出形状')
    a = ap.parse_args()

    roots = [r.strip() for r in a.roots.split(',') if r.strip()]
    files = []
    for r in roots:
        files.extend(sorted(glob.glob(os.path.join(r, '*.json'))))
    if a.max_charts:
        files = files[:a.max_charts]

    only = set()
    if a.charts_out:
        only = {x.strip() for x in a.charts_out.split(',') if x.strip()}

    shapes = []
    nch = nnote = 0
    sizes = {}
    fout = open(a.out, 'w', encoding='utf-8')
    sout = open(a.shape_out, 'w', encoding='utf-8')
    for fp in files:
        fid = os.path.basename(fp)[:-5]
        try:
            notes = read_notes(fp)
        except Exception:
            continue
        if len(notes) < 8:
            continue
        nch += 1
        nnote += len(notes)
        sizes[fid] = len(notes)
        # 每 per 颗切一段形状
        for s in range(0, len(notes) - 1, a.per):
            seg = notes[s:s + a.per]
            if len(seg) < 4:
                continue
            sh = shape_of(seg)
            if sh is None:
                continue
            sh['fid'] = fid
            sh['at'] = s
            shapes.append(sh)
        if only and fid not in only:
            continue
        if not only:
            continue
        # 输出完整逐音形状
        fout.write('\n=== %s  %d 音 ===\n' % (fid, len(notes)))
        for s in range(0, len(notes), a.per):
            seg = notes[s:s + a.per]
            row = ['.'] * NB
            for r in seg:
                c = cell(r[1])
                row[c] = CH.get(r[2], 'o')
            fout.write('%5d |%s|\n' % (s, ''.join(row)))
            # 下一行标相对上一颗的走向
            mark = ['.'] * NB
            for i in range(1, len(seg)):
                c0 = cell(seg[i - 1][1])
                c1 = cell(seg[i][1])
                if c1 != c0:
                    mid = (c0 + c1) // 2
                    mark[mid] = '<' if c1 < c0 else '>'
            fout.write('      |%s|\n' % ''.join(mark))
    fout.close()

    if a.auto_pick and sizes:
        srt = sorted(sizes.items(), key=lambda kv: kv[1])
        n = len(srt)
        picks = []
        for frac in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
            f, k = srt[int(frac * (n - 1))]
            picks.append((f, k))
        fout = open(a.out, 'w', encoding='utf-8')
        for f, k in picks:
            fp = None
            for root in roots:
                cand = os.path.join(root, f + '.json')
                if os.path.exists(cand):
                    fp = cand
                    break
            if fp is None:
                continue
            notes = read_notes(fp)
            fout.write('\n=== %s  %d 音 ===\n' % (f, len(notes)))
            for s in range(0, len(notes), a.per):
                seg = notes[s:s + a.per]
                row = ['.'] * NB
                for r in seg:
                    row[cell(r[1])] = CH.get(r[2], 'o')
                fout.write('%5d |%s|\n' % (s, ''.join(row)))
                mark = ['.'] * NB
                for i in range(1, len(seg)):
                    c0 = cell(seg[i - 1][1])
                    c1 = cell(seg[i][1])
                    if c1 != c0:
                        mark[(c0 + c1) // 2] = '<' if c1 < c0 else '>'
                fout.write('      |%s|\n' % ''.join(mark))
        fout.close()

    # 形状聚合（全量）
    import collections
    keys = ['span', 'med_step', 'max_step', 'flip', 'hold_same', 'chord_ratio']
    arr = {k: np.array([s[k] for s in shapes]) for k in keys}
    sout.write('charts %d notes %d segments %d\n' % (nch, nnote, len(shapes)))
    for k in keys:
        sout.write('%-12s 分位 %s\n' % (k, np.percentile(arr[k], [5, 25, 50, 75, 95]).round(3)))
    # 形状聚类：按 (跨度大小, 换向率高低, 同列率) 粗分
    lab = []
    for s in shapes:
        big = s['span'] > 500
        mid = 250 < s['span'] <= 500
        fl = s['flip'] > 0.45
        hs = s['hold_same'] > 0.15
        lab.append(('大跨' if big else ('中跨' if mid else '小跨'),
                    '常换' if fl else '顺推', '含同列' if hs else '无同列'))
    cnt = collections.Counter(lab)
    sout.write('\n=== 24音形状类别 ===\n')
    for k, v in cnt.most_common():
        sout.write('%-8s %-6s %-8s %6.2f%%\n' % (k[0], k[1], k[2], 100 * v / len(shapes)))
    sout.close()
    print('charts %d notes %d segments %d' % (nch, nnote, len(shapes)))


if __name__ == '__main__':
    main()
