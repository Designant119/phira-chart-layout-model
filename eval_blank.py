#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""blank 模型验收：15710 双押/单音 MAE + lost_requiem x 范围。"""
import json, statistics
from collections import Counter, defaultdict
import numpy as np

def r3(t): return round(float(t), 3)

def load_notes(path):
    return json.load(open(path))['notes']

def mae_compare(pred_file, orig_file):
    orig = load_notes(orig_file)
    pred = load_notes(pred_file)
    orig_idx = defaultdict(list)
    for n in orig:
        orig_idx[(r3(n['t']), n.get('line_id', -1))].append(n)
    orig_tcount = Counter(r3(n['t']) for n in orig)
    pairs = []
    unmatched = 0
    used_orig = defaultdict(int)
    for p in pred:
        key = (r3(p['t']), p.get('line_id', -1))
        lst = orig_idx.get(key)
        if not lst:
            unmatched += 1; continue
        i = used_orig[key]
        if i >= len(lst):
            unmatched += 1; continue
        o = lst[i]; used_orig[key] += 1
        pairs.append((float(p['x']), float(o['x']), r3(p['t'])))
    dbl, sgl = [], []
    for px, ox, t3 in pairs:
        (dbl if orig_tcount[t3] >= 2 else sgl).append(abs(px - ox))
    m = lambda a: statistics.mean(a) if a else float('nan')
    return {'n_pred': len(pred), 'n_orig': len(orig), 'n_paired': len(pairs),
            'n_unmatched': unmatched, 'dbl_n': len(dbl), 'dbl_mae': m(dbl),
            'sgl_n': len(sgl), 'sgl_mae': m(sgl)}

import sys
tag = sys.argv[1] if len(sys.argv) > 1 else 'blank'
print('='*60)
for fid in ['15710', '15372']:
    r = mae_compare(f'infer_traj_{tag}_{fid}.json', f'infer_orig_{fid}.json')
    print(f"[{tag} {fid}] paired={r['n_paired']} unmatched={r['n_unmatched']} | "
          f"双押MAE={r['dbl_mae']:.1f}px(n={r['dbl_n']})  单音MAE={r['sgl_mae']:.1f}px(n={r['sgl_n']})")
print('='*60)
model = load_notes(f'infer_traj_{tag}_lost_requiem.json')
allx = np.array([float(n['x']) for n in model])
print(f"[lost_requiem] n={len(model)}  x范围=[{allx.min():.0f}, {allx.max():.0f}]  "
      f"span={allx.max()-allx.min():.0f}")
# 双押间距
pg = defaultdict(list)
for n in model: pg[r3(n['t'])].append(float(n['x']))
sp = [abs(v[1]-v[0]) for v in pg.values() if len(v) == 2]
if sp:
    sp = np.array(sp)
    print(f"  双押间距 |x1-x2|: median={np.median(sp):.0f}  mean={sp.mean():.0f}  n={len(sp)}")
