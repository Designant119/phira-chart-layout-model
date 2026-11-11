#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按难度分层统计多押间距：<16.4 vs >=16.4；双押(2音) vs 多押(3+音)总跨度。"""
import json, glob, zipfile, io, os, re
from collections import defaultdict
import numpy as np

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
os.chdir(BASE)

# 一次建全量 pez 索引 {fid: path}
pez_index = {}
for p in glob.glob('subsets/unstable_*/*.pez'):
    fid = os.path.basename(p).split('_')[0]
    pez_index.setdefault(fid, p)
print(f'pez 索引: {len(pez_index)} 首', flush=True)

def get_level(fid):
    p = pez_index.get(fid)
    if not p:
        return None
    try:
        z = zipfile.ZipFile(p)
        jn = [n for n in z.namelist() if n.endswith('.json')][0]
        dj = json.load(io.BytesIO(z.read(jn)))
        lv = (dj.get('META') or {}).get('level', '')
        m = re.search(r'Lv\.?([\d.]+)', lv)
        return float(m.group(1)) if m else None
    except Exception:
        return None

lo = defaultdict(list)
hi = defaultdict(list)
no_lv = 0
parsed = glob.glob('parsed/*.json')
np.random.seed(0)
sample = np.random.choice(parsed, min(500, len(parsed)), replace=False)
for i, pf in enumerate(sample):
    fid = os.path.basename(pf)[:-5]
    lv = get_level(fid)
    if lv is None:
        no_lv += 1
        continue
    d = json.load(open(pf))
    notes = d.get('notes', [])
    if not notes:
        continue
    by_t = defaultdict(list)
    for n in notes:
        by_t[round(n['t'], 3)].append(n['x'])
    grp = lo if lv < 16.4 else hi
    for t, xs in by_t.items():
        xs = sorted(xs)
        if len(xs) == 2:
            grp['gap2'].append(abs(xs[1] - xs[0]))
        elif len(xs) >= 3:
            grp['gap3'].append(abs(xs[-1] - xs[0]))
    if (i + 1) % 400 == 0:
        print(f'{i+1}/{len(sample)}', flush=True)

out = {}
for name, g in [('low<16.4', lo), ('high>=16.4', hi)]:
    g2 = np.array(g['gap2']); g3 = np.array(g['gap3'])
    line = f'--- {name} ---\n'
    if len(g2):
        line += f'  双押(2音)间距: n={len(g2)} 中位{np.median(g2):.0f}px P25={np.percentile(g2,25):.0f} P75={np.percentile(g2,75):.0f}\n'
    else:
        line += '  双押: 无样本\n'
    if len(g3):
        line += f'  多押(3+音)总跨度: n={len(g3)} 中位{np.median(g3):.0f}px P25={np.percentile(g3,25):.0f} P75={np.percentile(g3,75):.0f}\n'
    print(line, flush=True)
    out[name] = {'gap2_median': float(np.median(g2)) if len(g2) else None,
                 'gap2_p25': float(np.percentile(g2, 25)) if len(g2) else None,
                 'gap2_p75': float(np.percentile(g2, 75)) if len(g2) else None,
                 'gap3_median': float(np.median(g3)) if len(g3) else None,
                 'gap3_n': int(len(g3))}
print(f'无难度信息: {no_lv}', flush=True)
json.dump(out, open('chord_gap_by_level.json', 'w'), indent=1)
