#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按"全部判定线保留 + 角度 0±10"重抽谱面，转成预览用的 json。"""
import os, json, argparse

ANG_LIM = 10.0


def norm_deg(v):
    v = float(v or 0.0) % 360.0
    if v > 180.0:
        v -= 360.0
    return v


ap = argparse.ArgumentParser()
ap.add_argument('--parsed', required=True)
ap.add_argument('--fids', required=True, help='逗号分隔')
ap.add_argument('--titles', default='')
ap.add_argument('--out', required=True)
ap.add_argument('--ang', type=float, default=10.0)
a = ap.parse_args()
ANG_LIM = a.ang

os.makedirs(os.path.join(a.out, 'parsed'), exist_ok=True)
fids = [x.strip() for x in a.fids.split(',') if x.strip()]
titles = [x.strip() for x in a.titles.split('|')] if a.titles else []

meta = []
for i, fid in enumerate(fids):
    p = os.path.join(a.parsed, fid + '.json')
    if not os.path.exists(p):
        print('miss', fid)
        continue
    d = json.load(open(p))
    keep = []
    for n in d.get('notes', []):
        th = norm_deg(n.get('theta', 0.0))
        if abs(th) > ANG_LIM:
            continue
        ty = int(n.get('type', 1))
        if ty == 2 and abs(norm_deg(n.get('theta_end', th))) > ANG_LIM:
            continue
        keep.append(n)
    keep.sort(key=lambda n: float(n['t']))
    out = {'fid': fid, 'notes': keep}
    json.dump(out, open(os.path.join(a.out, 'parsed', fid + '.json'), 'w'), ensure_ascii=False)
    cnt = {}
    for n in keep:
        k = str(int(n.get('type', 1)))
        cnt[k] = cnt.get(k, 0) + 1
    lines = len({int(n.get('line_id', 0)) for n in keep})
    meta.append({'fid': fid, 'title': titles[i] if i < len(titles) else fid,
                 'notes': len(keep), 'counts': cnt, 'lines': lines})
    print(fid, len(keep), cnt, 'lines', lines)

json.dump(meta, open(os.path.join(a.out, 'sample_meta.json'), 'w'), ensure_ascii=False, indent=1)
print('done', len(meta))
