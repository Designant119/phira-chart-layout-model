#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全部判定线都保留，按判定线角度筛音，切成 128 音窗口。特征 14 维（含线号变化信息）。"""
import os, glob, json, argparse, numpy as np

W = 128
STRIDE = 64
X_SCALE = 675.0


def norm_deg(v):
    v = float(v or 0.0) % 360.0
    if v > 180.0:
        v -= 360.0
    return v


def chord_counts(t):
    """同一时刻有几个音。"""
    n = len(t)
    out = np.zeros(n, np.float32)
    if n == 0:
        return out
    uq, inv = np.unique(t, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float32)
    return cnt[inv]


def dens_counts(t, win=4.0):
    """前后 4 秒内有多少音。"""
    n = len(t)
    out = np.zeros(n, np.float32)
    if n == 0:
        return out
    s = np.searchsorted(t, t - win, side='left')
    e = np.searchsorted(t, t + win, side='right')
    return (e - s).astype(np.float32)


ap = argparse.ArgumentParser()
ap.add_argument('--roots', required=True)
ap.add_argument('--split', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--ang', type=float, default=10.0)
a = ap.parse_args()
ANG_LIM = a.ang

roots = [r.strip() for r in a.roots.split(',') if r.strip()]
split = json.load(open(a.split))
val = set(split.get('val', []))
test = set(split.get('test', []))

seen = set()
rows = []
for r in roots:
    sub = os.path.join(r, 'parsed')
    if not os.path.isdir(sub):
        sub = r
    for fp in sorted(glob.glob(os.path.join(sub, '*.json'))):
        fid = os.path.basename(fp)[:-5]
        if fid in seen:
            continue
        seen.add(fid)
        if a.limit and len(rows) >= a.limit:
            break
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        notes = d.get('notes', [])
        if not notes:
            continue
        keep = []
        for n in notes:
            th = norm_deg(n.get('theta', 0.0))
            if abs(th) > ANG_LIM:
                continue
            ty = int(n.get('type', 1))
            if ty == 2 and abs(norm_deg(n.get('theta_end', th))) > ANG_LIM:
                continue
            keep.append(n)
        if len(keep) < 8:
            continue
        keep.sort(key=lambda n: float(n['t']))
        x = np.array([float(n['x']) for n in keep], np.float32) / X_SCALE
        t = np.array([float(n['t']) for n in keep], np.float32)
        ty = np.array([int(n.get('type', 1)) for n in keep], np.int32)
        dur = np.array([float(n.get('dur', 0)) for n in keep], np.float32)
        ln = np.array([int(n.get('line_id', 0)) for n in keep], np.int16)
        rows.append((fid, x, t, ty, dur, ln, len(notes)))

print('charts', len(rows), 'notes', sum(len(r[1]) for r in rows),
      'val', sum(r[0] in val for r in rows), 'test', sum(r[0] in test for r in rows), flush=True)

F, Y, M, FID, ST, LN = [], [], [], [], [], []
for fid, x, t, ty, dur, ln, raw_n in rows:
    n = len(x)
    chord = chord_counts(t)
    dens = dens_counts(t, 4.0)
    for s in range(0, max(1, n - W + 1), STRIDE):
        e = min(n, s + W)
        m = np.zeros(W, np.float32)
        m[:e - s] = 1
        f = np.zeros((W, 14), np.float32)
        y = np.zeros(W, np.float32)
        y[:e - s] = x[s:e]
        l = np.zeros(W, np.int16)
        l[:e - s] = ln[s:e]
        prev = 0.0
        prev2 = 0.0
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
                    lc, lgap, np.log1p(run)]
            prev2 = prev
            prev = x[i]
        F.append(f)
        Y.append(y)
        M.append(m)
        LN.append(l)
        FID.append(fid)
        ST.append(s)
        if e >= n:
            break

F = np.stack(F)
Y = np.stack(Y)
M = np.stack(M)
LN = np.stack(LN)
FID = np.array(FID)
ST = np.array(ST)
np.savez_compressed(a.out, F=F, Y=Y, M=M, fid=FID, start=ST, line=LN)
print('windows', len(F), 'dim', F.shape[-1], 'notes_in_windows', int(M.sum()), flush=True)
