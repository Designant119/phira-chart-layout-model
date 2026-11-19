#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整曲序列数据：一首歌一条完整 note 序列，不切窗口。
扁平存储：F_flat/Y_flat + offsets，训练时按谱取切片、batch 内动态 padding。
口径：脏线剔除、角度 ±5°、标签 clip ±675 归一化；特征只用纯节奏（不含 x 衍生）。
特征 11 维：0 dt 1-4 type 5 dur 6 chord 7 dens 8 换线 9 跨线距 10 同线连跑。"""
import os, glob, json, argparse
import numpy as np

X_SCALE = 675.0
ANG = 5.0
MAXLEN = 2048
MINLEN = 40


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


def feats(t, ty, dur, ln):
    n = len(t)
    chord = chord_counts(t)
    dens = dens_counts(t, 4.0)
    f = np.zeros((n, 11), np.float32)
    run = 0
    for i in range(n):
        dt = 0.0 if i == 0 else max(0.0, float(t[i] - t[i - 1]))
        if i == 0:
            lc, lgap, run = 0.0, 0.0, 1
        else:
            lc = 0.0 if int(ln[i]) == int(ln[i - 1]) else 1.0
            lgap = np.log1p(abs(int(ln[i]) - int(ln[i - 1])) / 10.0) if lc else 0.0
            run = 1 if lc else run + 1
        f[i] = [np.log1p(dt), *(ty[i] == np.arange(1, 5)), np.log1p(max(0.0, dur[i])),
                np.log1p(chord[i]), np.log1p(dens[i]), lc, lgap, np.log1p(run)]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    split = json.load(open(a.split))
    val, test = set(split['val']), set(split['test'])

    F_list, Y_list, offs, lengths, fids, split_tag = [], [], [0], [], [], []
    n_chart = n_skip_len = n_dirty = 0
    for fp in sorted(glob.glob(os.path.join(a.roots, '*.json'))):
        fid = os.path.basename(fp)[:-5]
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        raw = d.get('notes', [])
        if not raw:
            continue
        by_line = {}
        for n in raw:
            by_line.setdefault(int(n.get('line_id', 0)), []).append(n)
        clean = []
        for li, ns in by_line.items():
            xs = np.array([float(n.get('x', 0.0)) for n in ns])
            if np.median(np.abs(xs)) > X_SCALE or (np.abs(xs) > X_SCALE).mean() > 0.10:
                n_dirty += len(ns)
                continue
            clean.extend(ns)
        keep = [n for n in clean
                if abs(norm_deg(n.get('theta', 0.0))) <= ANG
                and abs(norm_deg(n.get('theta_end', norm_deg(n.get('theta', 0.0))))) <= ANG]
        if len(keep) < MINLEN:
            n_skip_len += 1
            continue
        keep.sort(key=lambda n: float(n['t']))
        if len(keep) > MAXLEN:
            keep = keep[:MAXLEN]
        x = np.clip([float(n['x']) for n in keep], -X_SCALE, X_SCALE) / X_SCALE
        t = np.array([float(n['t']) for n in keep], np.float32)
        ty = np.array([int(n.get('type', 1)) for n in keep], np.int32)
        dur = np.array([float(n.get('dur', 0)) for n in keep], np.float32)
        ln = np.array([int(n.get('line_id', 0)) for n in keep], np.int16)
        f = feats(t, ty, dur, ln)
        F_list.append(f)
        Y_list.append(x.astype(np.float32))
        offs.append(offs[-1] + len(keep))
        lengths.append(len(keep))
        fids.append(fid)
        split_tag.append(2 if fid in test else (1 if fid in val else 0))
        n_chart += 1

    F = np.concatenate(F_list)
    Y = np.concatenate(Y_list)
    offs = np.array(offs, np.int64)
    lengths = np.array(lengths, np.int64)
    fids = np.array(fids)
    split_tag = np.array(split_tag, np.int8)
    np.savez_compressed(a.out, F=F, Y=Y, offs=offs, lengths=lengths,
                         fid=fids, split=split_tag, maxlen=MAXLEN)
    print('谱 %d 跳过短谱 %d 脏音剔除 %d 总音 %d' % (n_chart, n_skip_len, n_dirty, len(Y)))
    print('长度 分位', np.percentile(lengths, [25, 50, 75, 90]).round(0))
    print('train %d val %d test %d' % ((split_tag == 0).sum(),
          (split_tag == 1).sum(), (split_tag == 2).sum()))
    print('saved', a.out, 'F', F.shape)


if __name__ == '__main__':
    main()
