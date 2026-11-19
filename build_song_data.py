#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整曲序列数据（扁平 offs 格式，匹配 train_song.py）：两趟预分配，避免 list 拼接双份内存。
清洗：剔横移脏线、角度 ±5°、标签 clip ±675；超长截断 MAXLEN。
输出 F(N,nf) / Y(N,) / offs(n+1) / lengths(n) / fid / split(0tr 1va 2te) / maxlen。"""
import os, glob, json, argparse
import numpy as np

MAXLEN = 2048
X_SCALE = 675.0
ANG = 5.0


def norm_deg(v):
    v = float(v or 0.0) % 360.0
    if v > 180.0:
        v -= 360.0
    return v


def parse_chart(d):
    """返回清洗排序后的 note list；不合格返回 None。"""
    raw = d.get('notes', [])
    if not raw:
        return None
    by_line = {}
    for n in raw:
        by_line.setdefault(int(n.get('line_id', 0)), []).append(n)
    clean = []
    for li, ns in by_line.items():
        xs = np.array([float(n.get('x', 0.0)) for n in ns])
        if np.median(np.abs(xs)) > X_SCALE or (np.abs(xs) > X_SCALE).mean() > 0.10:
            continue  # 脏线只剔除该线，不丢整首
        clean.extend(ns)
    keep = []
    for n in clean:
        th = norm_deg(n.get('theta', 0.0))
        te = norm_deg(n.get('theta_end', th))
        if abs(th) > ANG or abs(te) > ANG:
            continue
        keep.append(n)
    if len(keep) < 8:
        return None
    keep.sort(key=lambda n: float(n['t']))
    return keep[:MAXLEN]


def feats(keep):
    x = np.clip(np.array([float(n['x']) for n in keep], np.float32),
                -X_SCALE, X_SCALE) / X_SCALE
    t = np.array([float(n['t']) for n in keep], np.float32)
    ty = np.array([int(n.get('type', 1)) for n in keep], np.int32)
    dur = np.array([float(n.get('dur', 0)) for n in keep], np.float32)
    ln = np.array([int(n.get('line_id', 0)) for n in keep], np.int16)
    n = len(x)

    # 同拍统计
    uq, inv = np.unique(t, return_inverse=True)
    chord = np.bincount(inv).astype(np.float32)[inv]
    s = np.searchsorted(t, t - 4.0, side='left')
    e = np.searchsorted(t, t + 4.0, side='right')
    dens = (e - s).astype(np.float32)

    rank = np.zeros(n, np.float32)
    span = np.zeros(n, np.float32)
    meand = np.zeros(n, np.float32)
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

    f = np.zeros((n, 17), np.float32)
    prev = prev2 = 0.0
    run = 0
    for k in range(n):
        dt = 0.0 if k == 0 else max(0.0, float(t[k] - t[k - 1]))
        if k == 0:
            lc, lgap, run = 0.0, 0.0, 1
        else:
            lc = 0.0 if int(ln[k]) == int(ln[k - 1]) else 1.0
            lgap = np.log1p(abs(int(ln[k]) - int(ln[k - 1])) / 10.0) if lc else 0.0
            run = 1 if lc else run + 1
        f[k] = [np.log1p(dt), *(ty[k] == np.arange(1, 5)), np.log1p(max(0.0, dur[k])),
                np.log1p(chord[k]), np.log1p(dens[k]), prev, prev2, prev - prev2,
                lc, lgap, np.log1p(run),
                rank[k], np.log1p(span[k] * X_SCALE), np.log1p(meand[k] * X_SCALE)]
        prev2 = prev
        prev = x[k]
    return f, x


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

    # pass1：统计有效长度
    kept_files, lengths, fids, splits = [], [], [], []
    n_dirty_line = n_short = n_trunc = 0
    for fp in files:
        fid = os.path.basename(fp)[:-5]
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        keep = parse_chart(d)
        if keep is None:
            n_short += 1
            continue
        if len(keep) == MAXLEN:
            n_trunc += 1
        kept_files.append(fp)
        lengths.append(len(keep))
        fids.append(fid)
        splits.append(1 if fid in val else (2 if fid in test else 0))

    lengths = np.array(lengths, np.int64)
    splits = np.array(splits, np.int8)
    total = int(lengths.sum())
    offs = np.concatenate([[0], lengths.cumsum()])
    F = np.zeros((total, 17), np.float32)
    Y = np.zeros(total, np.float32)

    # pass2：回填
    for bi, fp in enumerate(kept_files):
        d = json.load(open(fp))
        keep = parse_chart(d)
        f, x = feats(keep)
        s, e = offs[bi], offs[bi + 1]
        F[s:e] = f
        Y[s:e] = x

    np.savez_compressed(a.out, F=F, Y=Y, offs=offs, lengths=lengths,
                        fid=np.array(fids), split=splits, maxlen=MAXLEN)
    print('charts', len(lengths), 'notes', total, 'short', n_short, 'trunc', n_trunc, flush=True)
    print('split', np.bincount(splits), '中位长度', int(np.median(lengths)), flush=True)
    print('Y |x|>1 %.2f%%' % (100 * (np.abs(Y) > 1).mean()), flush=True)
    print('saved', a.out)


if __name__ == '__main__':
    main()
