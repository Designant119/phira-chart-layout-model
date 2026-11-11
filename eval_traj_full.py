#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估 ckpt_traj_full：MAE 对齐对比 + lost_requiem 双押间距/平滑度。不修改任何现有脚本。"""
import json, statistics
from collections import Counter, defaultdict
import numpy as np

def r3(t): return round(float(t), 3)

def load_notes(path):
    return json.load(open(path))['notes']

def mae_compare(fid, pred_file, orig_file):
    """按 (round(t,3), line_id) 对齐 pred->orig，分组统计双押/单音 MAE。"""
    orig = load_notes(orig_file)
    pred = load_notes(pred_file)
    # orig 按 (t3, line_id) 索引；同键可能多条，用 list
    orig_idx = defaultdict(list)
    for n in orig:
        orig_idx[(r3(n['t']), n.get('line_id', -1))].append(n)
    # orig 每个 t3 上的音数（双押判定基准）
    orig_tcount = Counter(r3(n['t']) for n in orig)

    pairs = []          # (pred_x, orig_x, t3)
    unmatched = 0
    used_orig = defaultdict(int)
    for p in pred:
        key = (r3(p['t']), p.get('line_id', -1))
        lst = orig_idx.get(key)
        if not lst:
            unmatched += 1
            continue
        # 取该键尚未被消费的第一条（line_id 唯一时即唯一）
        i = used_orig[key]
        if i >= len(lst):
            unmatched += 1
            continue
        o = lst[i]; used_orig[key] += 1
        pairs.append((float(p['x']), float(o['x']), r3(p['t'])))

    dbl, sgl = [], []
    for px, ox, t3 in pairs:
        (dbl if orig_tcount[t3] >= 2 else sgl).append(abs(px - ox))
    def m(a): return statistics.mean(a) if a else float('nan')
    return {
        'fid': fid, 'pred': pred_file,
        'n_pred': len(pred), 'n_orig': len(orig),
        'n_paired': len(pairs), 'n_unmatched': unmatched,
        'dbl_n': len(dbl), 'dbl_mae': m(dbl),
        'sgl_n': len(sgl), 'sgl_mae': m(sgl),
    }

def show(tag, r):
    print(f"[{tag}] {r['fid']}  pred={r['n_pred']} orig={r['n_orig']} paired={r['n_paired']} unmatched={r['n_unmatched']}")
    print(f"    双押MAE={r['dbl_mae']:.2f}px (n={r['dbl_n']})   单音MAE={r['sgl_mae']:.2f}px (n={r['sgl_n']})")

print("="*70)
print("【MAE 对比】对齐规则: key=(round(t,3), line_id)；双押=orig 在该 t 音数>=2，单音===1")
print("="*70)
results = {}
for fid in ['15710','15372']:
    orig = f'infer_orig_{fid}.json'
    new = mae_compare(fid, f'infer_traj_full_{fid}.json', orig)
    v5  = mae_compare(fid, f'infer_v5low_{fid}.json', orig)
    results[fid] = (new, v5)
    show('v5_low ', v5)
    show('NEW    ', new)
    print()

print("="*70)
print("【lost_requiem 定量】")
print("="*70)
parsed = load_notes('parsed/lost_requiem.json')
model  = load_notes('infer_traj_full_lost_requiem.json')
# 模型按 (t3, line_id) 索引
midx = defaultdict(list)
for n in model: midx[(r3(n['t']), n.get('line_id',-1))].append(float(n['x']))
# parsed 按 t3 分组
pg = defaultdict(list)
for n in parsed: pg[r3(n['t'])].append(n)
dbl_ts = [t for t,ns in pg.items() if len(ns)==2]
print("parsed 双押时间戳数(恰好2音):", len(dbl_ts))
spacings = []
for t in dbl_ts:
    xs = []
    for n in pg[t]:
        k=(t, n.get('line_id',-1))
        if k in midx and midx[k]:
            xs.append(midx[k].pop(0))
    if len(xs)==2:
        spacings.append(abs(xs[0]-xs[1]))
sp=np.array(spacings)
print(f"模型成功配对双押 ts: {len(spacings)}/{len(dbl_ts)}")
print(f"双押 |x1-x2|: median={np.median(sp):.1f}  P25={np.percentile(sp,25):.1f}  P75={np.percentile(sp,75):.1f}  mean={sp.mean():.1f}")
# 平滑度：按 line_id 分组，相邻音符 |Δx|
byline = defaultdict(list)
for n in model: byline[n.get('line_id',-1)].append((r3(n['t']), float(n['x'])))
deltas=[]
for lid, lst in byline.items():
    lst.sort()
    for i in range(1,len(lst)):
        deltas.append(abs(lst[i][1]-lst[i-1][1]))
d=np.array(deltas)
print(f"单线相邻|Δx|: P99={np.percentile(d,99):.1f}  max={d.max():.1f}  median={np.median(d):.1f}")
# x 范围越界检查
allx=np.array([float(n['x']) for n in model])
print(f"模型输出 x 范围: [{allx.min():.1f}, {allx.max():.1f}]  (合法 [-675,675])")
