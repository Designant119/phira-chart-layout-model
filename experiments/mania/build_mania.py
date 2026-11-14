#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mania .osu -> 逐音符特征/标签（4K），供分类模型用。"""
import json, os, glob, re, random
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "osu")


def parse(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"CircleSize:\s*(\d+)", txt)
    if not m:
        return None
    keys = int(m.group(1))
    if keys not in (4, 7):
        return None
    body = txt.split("[HitObjects]", 1)[-1]
    notes = []
    for line in body.splitlines():
        p = line.strip().split(",")
        if len(p) < 5:
            continue
        try:
            x, t, typ = float(p[0]), int(p[2]), int(p[3])
        except Exception:
            continue
        col = min(max(int(np.floor(x * keys / 512.0)), 0), keys - 1)
        hold = bool(typ & 128)
        end = int(p[5]) if hold and len(p) > 5 and p[5].strip().isdigit() else t
        notes.append([t, col, 2 if hold else 1, max(0, end - t)])
    notes.sort(key=lambda n: (n[0], n[1]))
    return keys, notes


def feats(notes, keys):
    n = len(notes)
    t = np.array([q[0] for q in notes], dtype=np.float64) / 1000.0
    col = np.array([q[1] for q in notes], dtype=np.int64)
    typ = np.array([q[2] for q in notes], dtype=np.int64)
    dur = np.array([q[3] for q in notes], dtype=np.float64) / 1000.0
    dt = np.zeros(n); dt[1:] = np.clip(np.log1p(t[1:] - t[:-1]) / 3.0, 0, 1)
    # 同刻和弦大小
    chord = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        chord[i:j + 1] = min(j - i + 1, 4) / 4.0
        i = j + 1
    # 密度 ±0.5s
    dens = np.zeros(n)
    j0 = 0
    for i in range(n):
        while t[i] - t[j0] > 0.5:
            j0 += 1
        j1 = i
        while j1 + 1 < n and t[j1 + 1] - t[i] <= 0.5:
            j1 += 1
        dens[i] = min((j1 - j0 + 1) / 20.0, 1.0)
    prev = np.full(n, -1); prev2 = np.full(n, -1)
    prev[1:] = col[:-1]; prev2[2:] = col[:-2]
    streak = np.zeros(n)
    for i in range(1, n):
        streak[i] = streak[i - 1] + 1 if col[i] == col[i - 1] else 0
    # 手位：4K 左 0/1，右 2/3；7K 左 0-2，右 4-6
    if keys == 4:
        side = np.where(col < 2, 0, 1)
    else:
        side = np.where(col < 3, 0, 1)
    lastL = np.full(n, -1.0); lastR = np.full(n, -1.0)
    l = r = -1.0
    for i in range(n):
        lastL[i] = l; lastR[i] = r
        if side[i] == 0:
            l = col[i]
        else:
            r = col[i]
    def onehot(v, k):
        oh = np.zeros((len(v), k + 1), dtype=np.float32)
        oh[np.arange(len(v)), np.clip(v + 1, 0, k)] = 1
        return oh
    X = np.concatenate([
        dt[:, None], chord[:, None], dens[:, None],
        (typ == 2).astype(np.float32)[:, None], np.clip(dur / 2.0, 0, 1)[:, None],
        np.clip(streak / 10.0, 0, 1)[:, None],
        onehot(prev, keys), onehot(prev2, keys),
        lastL[:, None] / max(keys - 1, 1), lastR[:, None] / max(keys - 1, 1),
    ], axis=1)
    return X.astype(np.float32), col.astype(np.int64), t


charts = []
for f in glob.glob(os.path.join(OUT, "*.osu")):
    r = parse(f)
    if r and len(r[1]) >= 100:
        charts.append((os.path.basename(f)[:-4], r[0], r[1]))
print("charts", len(charts))
random.seed(123); random.shuffle(charts)
for keys in (4, 7):
    sub = [c for c in charts if c[1] == keys]
    print(keys, "K", len(sub))
    if len(sub) < 2:
        continue
    n_tr = int(0.7 * len(sub)); n_va = int(0.15 * len(sub))
    splits = {"train": sub[:n_tr], "val": sub[n_tr:n_tr + n_va], "test": sub[n_tr + n_va:]}
    out = {}
    for sp, ss in splits.items():
        if not ss:
            continue
        Xs, ys, ids = [], [], []
        for fid, k, notes in ss:
            X, y, t = feats(notes, k)
            Xs.append(X); ys.append(y); ids += [fid] * len(y)
        out[sp] = (np.concatenate(Xs), np.concatenate(ys), np.array(ids))
        print(sp, out[sp][0].shape, "charts", len(ss))
    if len(out) >= 3:
        payload = {}
        for sp in out:
            payload[f"{sp}_X"] = out[sp][0]
            payload[f"{sp}_y"] = out[sp][1]
            payload[f"{sp}_ids"] = out[sp][2]
        print("保存keys", keys, sorted(payload), flush=True)
        np.savez(os.path.join(ROOT, f"mania_{keys}k.npz"), **payload)
    json.dump({sp: [c[0] for c in ss] for sp, ss in splits.items()},
              open(os.path.join(ROOT, f"split_{keys}k.json"), "w"), indent=1)
