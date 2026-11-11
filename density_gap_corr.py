#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：双押间距是否由局部密度可预测（决定模型能否自学"密集段双押窄"）。
对每个双押时刻 t：密度 = t±1s 窗口内音符数（跨线）。
按密度分层统计双押间距中位 → 若高密度→窄，则模型 density 特征足够。"""
import json, glob, os
from collections import defaultdict
import numpy as np

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
os.chdir(BASE)

parsed = glob.glob('parsed/*.json')
np.random.seed(0)
sample = np.random.choice(parsed, min(400, len(parsed)), replace=False)

# 每首歌收集：双押时刻(间距, 局部密度)
rows = []  # (gap, density)
for pf in sample:
    d = json.load(open(pf))
    notes = d.get('notes', [])
    if not notes:
        continue
    by_t = defaultdict(list)
    for n in notes:
        by_t[round(n['t'], 3)].append(n['x'])
    times = sorted(by_t.keys())
    for i, t in enumerate(times):
        xs = sorted(by_t[t])
        if len(xs) != 2:
            continue
        gap = abs(xs[1] - xs[0])
        # 局部密度：t±1s 内的音符总数
        dens = sum(len(by_t[t2]) for t2 in times if abs(t2 - t) <= 1.0)
        rows.append((gap, dens))

rows = np.array(rows, dtype=np.float64)
gaps = rows[:, 0]; dens = rows[:, 1]
print(f'双押样本: {len(rows)}')
# 按密度四分位分层
qs = np.quantile(dens, [0, 0.25, 0.5, 0.75, 1.0])
print(f'密度四分位: {qs.astype(int)}')
for i in range(4):
    m = (dens >= qs[i]) & (dens < qs[i + 1]) if i < 3 else (dens >= qs[i]) & (dens <= qs[i + 1])
    g = gaps[m]
    print(f'  密度[{qs[i]:.0f},{qs[i+1]:.0f}): n={len(g)} 双押间距中位 {np.median(g):.0f}px')
# 相关性
from scipy import stats
r, p = stats.spearmanr(dens, gaps)
print(f'Spearman(密度, 间距) = {r:.3f} (p={p:.1e})')
