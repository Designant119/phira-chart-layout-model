#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重新提取训练数据：
1. 剔除横移脏线（线内中位 |x|>675 或 >10% 音超界）
2. 标签 clip 到 ±675，归一化 -1..1
3. 角度 ±5° 过滤；特征沿用 17 维，分组见 KEEP 注释
输出与 singleline npz 同构，可直接喂 train_parallel_grp.py。"""
import os, glob, json, argparse
import numpy as np

W = 128
STRIDE = 64
X_SCALE = 675.0
ANG = 5.0


def norm_deg(v):
    v = float(v or 0.0) % 360.0
    if v > 180.0:
        v -= 360.0
    return v


def chord_counts(t):
    n = len(t)
    out = np.zeros(n, np.float32)
    if n == 0:
        return out
    uq, inv = np.unique(t, return_inverse=True)
    return np.bincount(inv).astype(np.float32)[inv]


def dens_counts(t, win=4.0):
    n = len(t)
    out = np.zeros(n, np.float32)
    if n == 0:
        return out
    s = np.searchsorted(t, t - win, side='left')
    e = np.searchsorted(t, t + win, side='right')
    return (e - s).astype(np.float32)


def chord_group(x, t):
    n = len(x)
    rank = np.zeros(n, np.float32)
    span = np.zeros(n, np.float32)
    meand = np.zeros(n, np.float32)
    uq, inv = np.unique(t, return_inverse=True)
    for g in range(len(uq)):
        idx = np.where(inv == g)[0]
        if len(idx) == 1:
            continue
        xs = x[idx]
        o = np.argsort(xs)
        ranks = np.linspace(-1.0, 1.0, len(idx))
        for pos, ii in enumerate(o):
            rank[idx[ii]] = ranks[pos]
        span[idx] = xs.max() - xs.min()
        meand[idx] = float(np.mean(np.abs(xs[:, None] - xs[None, :])))
    return rank, span, meand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    split = json.load(open(a.split))
    val, test = set(split['val']), set(split['test'])

    files = sorted(glob.glob(os.path.join(a.roots, '*.json')))
    if a.limit:
        files = files[:a.limit]

    rows = []
    n_dirty_line = n_dirty_note = 0
    for fp in files:
        fid = os.path.basename(fp)[:-5]
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        raw = d.get('notes', [])
        if not raw:
            continue
        # 逐线判脏
        by_line = {}
        for n in raw:
            by_line.setdefault(int(n.get('line_id', 0)), []).append(n)
        clean = []
        for li, ns in by_line.items():
            xs = np.array([float(n.get('x', 0.0)) for n in ns])
            dirty = np.median(np.abs(xs)) > X_SCALE or (np.abs(xs) > X_SCALE).mean() > 0.10
            if dirty:
                n_dirty_line += 1
                n_dirty_note += len(ns)
                continue
            clean.extend(ns)
        keep = []
        for n in clean:
            th = norm_deg(n.get('theta', 0.0))
            te = norm_deg(n.get('theta_end', th))
            if abs(th) > ANG or abs(te) > ANG:
                continue
            keep.append(n)
        if len(keep) < 8:
            continue
        keep.sort(key=lambda n: float(n['t']))
        x = np.clip(np.array([float(n['x']) for n in keep], np.float32),
                    -X_SCALE, X_SCALE) / X_SCALE
        t = np.array([float(n['t']) for n in keep], np.float32)
        ty = np.array([int(n.get('type', 1)) for n in keep], np.int32)
        dur = np.array([float(n.get('dur', 0)) for n in keep], np.float32)
        ln = np.array([int(n.get('line_id', 0)) for n in keep], np.int16)
        rows.append((fid, x, t, ty, dur, ln))

    print('charts', len(rows), 'notes', sum(len(r[1]) for r in rows),
          'dirty_lines', n_dirty_line, 'dirty_notes', n_dirty_note, flush=True)

    F, Y, M, FID, ST, LN, G = [], [], [], [], [], [], []
    for fid, x, t, ty, dur, ln in rows:
        n = len(x)
        chord = chord_counts(t)
        dens = dens_counts(t, 4.0)
        crank, cspan, cmeand = chord_group(x, t)
        _, grp = np.unique(t, return_inverse=True)
        for s in range(0, max(1, n - W + 1), STRIDE):
            e = min(n, s + W)
            m = np.zeros(W, np.float32)
            m[:e - s] = 1
            f = np.zeros((W, 17), np.float32)
            y = np.zeros(W, np.float32)
            y[:e - s] = x[s:e]
            l = np.zeros(W, np.int16)
            l[:e - s] = ln[s:e]
            g = np.zeros(W, np.int64)
            g[:e - s] = grp[s:e]
            prev = prev2 = 0.0
            run = 0
            for k in range(W):
                i = s + k
                if i >= e:
                    break
                dt = 0.0 if k == 0 else max(0.0, float(t[i] - t[i - 1]))
                if k == 0:
                    lc, lgap, run = 0.0, 0.0, 1
                else:
                    lc = 0.0 if int(ln[i]) == int(ln[i - 1]) else 1.0
                    lgap = np.log1p(abs(int(ln[i]) - int(ln[i - 1])) / 10.0) if lc else 0.0
                    run = 1 if lc else run + 1
                f[k] = [np.log1p(dt), *(ty[i] == np.arange(1, 5)), np.log1p(max(0.0, dur[i])),
                        np.log1p(chord[i]), np.log1p(dens[i]), prev, prev2, prev - prev2,
                        lc, lgap, np.log1p(run),
                        crank[i], np.log1p(cspan[i] * X_SCALE), np.log1p(cmeand[i] * X_SCALE)]
                prev2 = prev
                prev = x[i]
            F.append(f); Y.append(y); M.append(m); LN.append(l)
            FID.append(fid); ST.append(s); G.append(g)
            if e >= n:
                break

    F = np.stack(F); Y = np.stack(Y); M = np.stack(M)
    LN = np.stack(LN); G = np.stack(G)
    FID = np.array(FID); ST = np.array(ST)
    np.savez_compressed(a.out, F=F, Y=Y, M=M, fid=FID, start=ST, line=LN, G=G)
    print('windows', len(F), 'feat', F.shape[-1], 'notes_in', int(M.sum()), flush=True)
    # 标签分布自检
    ym = Y[M.astype(bool)]
    print('标签 |x| 分位', np.percentile(np.abs(ym), [50, 90, 99, 99.9]).round(3),
          '超界 %.2f%%' % (100 * (np.abs(ym) > 1).mean()), flush=True)


if __name__ == '__main__':
    main()
