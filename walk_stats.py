#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫全部真谱，统计单音走向手感：步长、同列率、换向间隔、同侧连走步数、多押间距。"""
import os, json, glob, argparse
import numpy as np

HALF = 675.0
ANG = 5.0


def norm_deg(v):
    return ((float(v or 0.0) + 180.0) % 360.0) - 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.roots, '*.json')))

    steps, same, side_run, chord_pairs = [], [], [], {2: [], 3: [], 4: []}
    nch = nnote = 0
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        ns = []
        for n in d.get('notes', []):
            th = norm_deg(n.get('theta', 0.0)); te = norm_deg(n.get('theta_end', th))
            if abs(th) > ANG or abs(te) > ANG:
                continue
            ns.append((float(n['t']), float(n.get('x', 0.0))))
        if len(ns) < 8:
            continue
        ns.sort()
        nch += 1; nnote += len(ns)
        x = np.array([v for _, v in ns])
        dx = np.diff(x)
        steps.extend(np.abs(dx).tolist())
        same.extend((np.abs(dx) < 20).tolist())
        # 方向：0 原地，+1 右，-1 左
        sgn = np.sign(dx)
        # 同侧连走步数（忽略原地）
        run = 1
        for i in range(1, len(sgn)):
            if sgn[i] == 0 or sgn[i - 1] == 0:
                continue
            if sgn[i] == sgn[i - 1]:
                run += 1
            else:
                side_run.append(run); run = 1
        side_run.append(run)
        # 多押：按时间分组
        from collections import defaultdict
        grp = defaultdict(list)
        for t, v in ns:
            grp[round(t, 4)].append(v)
        for g in grp.values():
            k = len(g)
            if k in chord_pairs:
                g = sorted(g)
                chord_pairs[k].extend(np.diff(g).tolist())

    steps = np.array(steps); same = np.array(same); side_run = np.array(side_run)
    print('charts %d notes %d' % (nch, nnote))
    print('步长(px) 分位', np.percentile(steps, [10, 25, 50, 75, 90]).round(0))
    print('同列率(|dx|<20px) %.1f%%' % (100 * same.mean()))
    print('同侧连走步数 分位', np.percentile(side_run, [25, 50, 75, 90]).round(1))
    print('同侧步数分布:', {k: round(float((side_run == k).mean()) * 100, 1) for k in range(1, 7)})
    for k, v in chord_pairs.items():
        if v:
            v = np.array(v)
            print('%d押间距(px) 中位 %.0f 分位 %s' % (k, np.median(v),
                  np.percentile(v, [25, 75]).round(0)))


if __name__ == '__main__':
    main()
