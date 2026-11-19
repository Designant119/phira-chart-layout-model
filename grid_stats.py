#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫全部谱，估每 24 音段用的网格：间距、列数、是否关于中心对称。"""
import os, json, glob, argparse
import numpy as np

NB = 81
HALF = 675.0
PER = 24


def cell(x):
    return int(np.clip(round((x + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1))


ap = argparse.ArgumentParser()
ap.add_argument('--roots', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()

files = []
for r in [s.strip() for s in a.roots.split(',') if s.strip()]:
    files.extend(sorted(glob.glob(os.path.join(r, '*.json'))))

gaps = []
ncols = []
sym = []
span = []
nch = nnote = 0
for fp in files:
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    ns = []
    for n in d.get('notes', []):
        th = ((float(n.get('theta', 0.0) or 0.0) + 180) % 360) - 180
        te = ((float(n.get('theta_end', th) or th) + 180) % 360) - 180
        if abs(th) > 5 or abs(te) > 5:
            continue
        ns.append((float(n['t']), float(n.get('x', 0.0))))
    if len(ns) < 8:
        continue
    ns.sort()
    nch += 1
    nnote += len(ns)
    x = np.array([r[1] for r in ns])
    for s in range(0, len(x) - PER + 1, PER):
        seg = x[s:s + PER]
        cs = np.array(sorted(set(cell(v) for v in seg)))
        if len(cs) < 2:
            continue
        dif = np.diff(cs)
        g = float(np.median(dif))
        gaps.append(g)
        ncols.append(len(cs))
        span.append(float(cs[-1] - cs[0]))
        # 对称性：最大最小之和约等于 2*40
        sym.append(abs((cs[0] + cs[-1]) - 2 * (NB - 1) / 2))

gaps = np.array(gaps); ncols = np.array(ncols); span = np.array(span); sym = np.array(sym)
print('charts %d notes %d segs %d' % (nch, nnote, len(gaps)))
print('网格间距(格) 分位', np.percentile(gaps, [5, 10, 25, 50, 75, 90, 95]).round(2))
print('间距(格) 直方:')
for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 9), (9, 99)]:
    print('  %d-%d: %.1f%%' % (lo, hi, 100 * ((gaps >= lo) & (gaps < hi)).mean()))
print('列数 分位', np.percentile(ncols, [5, 25, 50, 75, 95]).round(1))
print('列数分布:', {k: round(float((ncols == k).mean()) * 100, 1) for k in range(2, 11)})
print('段跨度(格) 中位 %.1f' % np.median(span))
print('首末列对称误差(格) 中位 %.1f, <2格占比 %.1f%%' % (
    np.median(sym), 100 * (sym < 2).mean()))
np.savez(a.out, gaps=gaps, ncols=ncols, span=span, sym=sym)
print('saved', a.out)
