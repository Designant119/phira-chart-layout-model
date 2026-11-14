#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 mania .osu：列号、时间、类型、长条；输出统计与多数类基线。"""
import json, os, glob, re
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "osu")
meta = json.load(open(os.path.join(ROOT, "meta.json")))


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
        col = int(np.floor(x * keys / 512.0))
        col = min(max(col, 0), keys - 1)
        hold = bool(typ & 128)
        end = int(p[5]) if hold and len(p) > 5 and p[5].strip().isdigit() else t
        notes.append((t, col, 2 if hold else 1, end))
    notes.sort()
    return keys, notes


charts = []
for f in glob.glob(os.path.join(OUT, "*.osu")):
    r = parse(f)
    if r and len(r[1]) >= 100:
        charts.append((os.path.basename(f)[:-4], r[0], r[1]))
print("charts", len(charts))
for keys in (4, 7):
    sub = [c for c in charts if c[1] == keys]
    if not sub:
        continue
    notes = [n for c in sub for n in c[2]]
    cols = np.array([n[1] for n in notes])
    dist = np.bincount(cols, minlength=keys) / len(cols)
    print(f"{keys}K: {len(sub)} 谱 / {len(notes)} 音符 / 最长 {max(len(c[2]) for c in sub)}")
    print("  列分布", np.round(dist, 3).tolist(), "多数类基线", round(float(dist.max()), 3))
    # 和弦率 & jack 率（相邻同列）
    chord = 0; jack = 0
    for c in sub:
        ts = [n[0] for n in c[2]]
        chord += sum(1 for i in range(1, len(ts)) if ts[i] == ts[i - 1])
        cols2 = [n[1] for n in c[2]]
        jack += sum(1 for i in range(1, len(cols2)) if cols2[i] == cols2[i - 1])
    print("  同刻和弦占比 %.3f / 相邻同列 %.3f" % (chord / len(notes), jack / len(notes)))
json.dump({"charts": [(c[0], c[1], len(c[2])) for c in charts]}, open(os.path.join(ROOT, "stat.json"), "w"), indent=1)
