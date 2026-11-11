#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step4 评估 traj v2（新口径：只统计 Tap(1)/Hold(2)，排除 Drag/Flick）。
用法：python3 eval_traj_v2.py
- 15710/15372: infer_traj_v2_*.json vs infer_orig_*.json 算双押/单音位置 MAE。
- lost_requiem: 从 infer_traj_v2_lost_requiem.json 统计 x 范围与双押间距。
"""
import json, statistics, os
from collections import Counter, defaultdict
import numpy as np

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
os.chdir(BASE)


def r3(t):
    return round(float(t), 3)


def load_notes(path):
    notes = json.load(open(path))['notes']
    # 新口径：只保留 Tap/Hold
    return [n for n in notes if int(n.get('type', 1)) in (1, 2)]


def mae_compare(fid, pred_file, orig_file):
    orig = load_notes(orig_file)
    pred = load_notes(pred_file)
    orig_idx = defaultdict(list)
    for n in orig:
        orig_idx[(r3(n['t']), n.get('line_id', -1))].append(n)
    orig_tcount = Counter(r3(n['t']) for n in orig)
    pairs, unmatched = [], 0
    used = defaultdict(int)
    for p in pred:
        key = (r3(p['t']), p.get('line_id', -1))
        lst = orig_idx.get(key)
        if not lst:
            unmatched += 1
            continue
        i = used[key]
        if i >= len(lst):
            unmatched += 1
            continue
        o = lst[i]; used[key] += 1
        pairs.append((float(p['x']), float(o['x']), r3(p['t'])))
    dbl, sgl = [], []
    for px, ox, t3 in pairs:
        (dbl if orig_tcount[t3] >= 2 else sgl).append(abs(px - ox))
    m = lambda a: statistics.mean(a) if a else float('nan')
    return dict(fid=fid, n_pred=len(pred), n_orig=len(orig), n_paired=len(pairs),
                n_unmatched=unmatched, dbl_n=len(dbl), dbl_mae=m(dbl),
                sgl_n=len(sgl), sgl_mae=m(sgl))


print("=" * 72)
print("【MAE 新口径：仅 Tap/Hold；对齐 key=(round(t,3), line_id)；双押=orig 该t音数>=2】")
print("=" * 72)
for fid in ['15710', '15372']:
    orig = f'infer_orig_{fid}.json'
    new = f'infer_traj_v2_{fid}.json'
    if not os.path.exists(new):
        print(f"[{fid}] 缺少 {new}")
        continue
    r = mae_compare(fid, new, orig)
    print(f"[{fid}] pred={r['n_pred']} orig={r['n_orig']} paired={r['n_paired']} unmatched={r['n_unmatched']}")
    print(f"    双押MAE={r['dbl_mae']:.2f}px (n={r['dbl_n']})   单音MAE={r['sgl_mae']:.2f}px (n={r['sgl_n']})")

print("=" * 72)
print("【lost_requiem 定量：infer_traj_v2_lost_requiem.json（仅 Tap/Hold）】")
print("=" * 72)
mp = 'infer_traj_v2_lost_requiem.json'
if os.path.exists(mp):
    model = load_notes(mp)
    allx = np.array([float(n['x']) for n in model])
    print(f"模型输出 x 范围: min={allx.min():.1f}  max={allx.max():.1f}  (跨度={allx.max()-allx.min():.1f})")
    # 双押：按 (t3) 分组，恰好2个 Tap/Hold 视为双押
    g = defaultdict(list)
    for n in model:
        g[r3(n['t'])].append(float(n['x']))
    spacings = [abs(xs[0] - xs[1]) for t, xs in g.items() if len(xs) == 2]
    sp = np.array(spacings) if spacings else np.array([0.0])
    print(f"双押事件数(恰好2音): {len(spacings)}")
    print(f"双押 |x1-x2|: mean={sp.mean():.1f}  median={np.median(sp):.1f}  "
          f"P10={np.percentile(sp,10):.1f}  min={sp.min():.1f}")
else:
    print(f"缺少 {mp}")
