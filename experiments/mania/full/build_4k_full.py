#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量 4K：osu/*.osu -> mania_4k_full.npz + split_4k_full.json"""
import json, os, glob, re, random
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "osu")


def parse(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"CircleSize:\s*(\d+)", txt)
    if not m: return None
    keys = int(m.group(1))
    body = txt.split("[HitObjects]", 1)[-1]
    notes = []
    for line in body.splitlines():
        p = line.strip().split(",")
        if len(p) < 5: continue
        try: x, t, typ = float(p[0]), int(p[2]), int(p[3])
        except Exception: continue
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
    chord = np.zeros(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]: j += 1
        chord[i:j + 1] = min(j - i + 1, 4) / 4.0; i = j + 1
    dens = np.zeros(n); j0 = 0
    for i in range(n):
        while t[i] - t[j0] > 0.5: j0 += 1
        j1 = i
        while j1 + 1 < n and t[j1 + 1] - t[i] <= 0.5: j1 += 1
        dens[i] = min((j1 - j0 + 1) / 20.0, 1.0)
    prev = np.full(n, -1); prev2 = np.full(n, -1)
    prev[1:] = col[:-1]; prev2[2:] = col[:-2]
    # 只给“进入当前音之前”的连打长度，不能用当前列号（否则标签泄漏）
    streak = np.zeros(n); run = 0
    for i in range(n):
        streak[i] = run
        run = run + 1 if (i > 0 and col[i] == col[i - 1]) else 1
    side = np.where(col < 2, 0, 1)
    lastL = np.full(n, -1.0); lastR = np.full(n, -1.0); l = r = -1.0
    for i in range(n):
        lastL[i] = l; lastR[i] = r
        if side[i] == 0: l = col[i]
        else: r = col[i]
    def onehot(v, k):
        oh = np.zeros((len(v), k + 1), dtype=np.float32); oh[np.arange(len(v)), np.clip(v + 1, 0, k)] = 1
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
    if r and r[0] == 4 and len(r[1]) >= 100:
        charts.append((os.path.basename(f)[:-4], 4, r[1]))
print("4K charts", len(charts), "notes", sum(len(c[2]) for c in charts))
random.seed(123); random.shuffle(charts)
n_tr = int(0.8 * len(charts)); n_va = int(0.1 * len(charts))
splits = {"train": charts[:n_tr], "val": charts[n_tr:n_tr + n_va], "test": charts[n_tr + n_va:]}
payload = {}
for sp, ss in splits.items():
    Xs, ys, ids = [], [], []
    for fid, k, notes in ss:
        X, y, t = feats(notes, k); Xs.append(X); ys.append(y); ids += [fid] * len(y)
    payload[f"{sp}_X"] = np.concatenate(Xs); payload[f"{sp}_y"] = np.concatenate(ys)
    payload[f"{sp}_ids"] = np.array(ids)
    print(sp, payload[f"{sp}_X"].shape, "charts", len(ss))
np.savez(os.path.join(ROOT, "mania_4k_full.npz"), **payload)
json.dump({sp: [c[0] for c in ss] for sp, ss in splits.items()},
          open(os.path.join(ROOT, "split_4k_full.json"), "w"), indent=1)
print("saved mania_4k_full.npz")
