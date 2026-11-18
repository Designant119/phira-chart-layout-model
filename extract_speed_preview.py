#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从原始谱面里算出每颗 note 时刻的判定线流速（speed），供预览按流速布局。"""
import os, sys, glob, json, zipfile, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from easing_lib import interp_events
from phira_fetch import rpe_time_to_beat
import statistics, math
import build_angle_parsed as BAP

ap = argparse.ArgumentParser()
ap.add_argument('--zips', required=True)
ap.add_argument('--fids', required=True)
ap.add_argument('--ang', type=float, default=10.0)
ap.add_argument('--out', required=True)
a = ap.parse_args()

os.makedirs(os.path.join(a.out, 'parsed'), exist_ok=True)
fids = [x.strip() for x in a.fids.split(',') if x.strip()]


def find_zip(fid):
    for p in glob.glob(os.path.join(a.zips, '**', fid + '_*.zip'), recursive=True):
        return p
    for p in glob.glob(os.path.join(a.zips, '**', fid + '*.zip'), recursive=True):
        return p
    return None


meta = []
for fid in fids:
    zp = find_zip(fid)
    if not zp:
        print('miss', fid)
        continue
    with zipfile.ZipFile(zp) as zf:
        ch = BAP.pick_chart(zf)
    if not ch:
        print('nochart', fid)
        continue
    segs = sorted([(rpe_time_to_beat(b['startTime']), float(b['bpm']))
                   for b in ch.get('BPMList', [])], key=lambda x: x[0])
    notes = []
    speeds = []
    for li, jl in enumerate(ch.get('judgeLineList', [])):
        layers = jl.get('eventLayers') or []
        sp_ev = []
        if layers and isinstance(layers[0], dict):
            sp_ev = layers[0].get('speedEvents') or []
        for n in jl.get('notes', []):
            if n.get('isFake', 0) == 1:
                continue
            typ = int(n.get('type', 1))
            tb = rpe_time_to_beat(n['startTime'])
            t = BAP.beat_to_sec(tb, segs)
            st = BAP.line_state_at(jl, ch, tb, {})
            th = ((st[2] + 180) % 360) - 180
            px_line = float(st[0])
            nx = float(n.get('positionX', 0))
            abs_x = px_line + nx * math.cos(math.radians(th))
            dur = 0.0
            the = th
            if typ == 2 and isinstance(n.get('endTime'), list):
                eb = rpe_time_to_beat(n['endTime'])
                dur = max(0.0, BAP.beat_to_sec(eb, segs) - t)
                the = BAP.line_state_at(jl, ch, eb, {})[2]
                the = ((the + 180) % 360) - 180
            sp = float(interp_events(sp_ev, tb)) * float(n.get('speed', 1.0) or 1.0)
            speeds.append(sp)
            notes.append({'t': round(t, 6), 'x': nx, 'abs_x': round(abs_x, 3), 'line_x': round(px_line, 3), 'type': typ,
                          'dur': round(dur, 6), 'line_id': li,
                          'theta': round(float(th), 3), 'theta_end': round(float(the), 3),
                          'speed': round(sp, 3)})
    keep = [n for n in notes
            if abs(n['theta']) <= a.ang and abs(n['theta_end']) <= a.ang]
    keep.sort(key=lambda n: float(n['t']))
    json.dump({'fid': fid, 'notes': keep},
              open(os.path.join(a.out, 'parsed', fid + '.json'), 'w'), ensure_ascii=False)
    sp_kept = [n['speed'] for n in keep] or [1.0]
    med = statistics.median(sp_kept)
    meta.append({'fid': fid, 'title': fid, 'notes': len(keep), 'speed_median': round(med, 3),
                 'speed_min': round(min(sp_kept), 3), 'speed_max': round(max(sp_kept), 3)})
    print(fid, 'notes', len(keep), 'speed 中位 %.2f 范围 %.2f~%.2f' % (med, min(sp_kept), max(sp_kept)))

json.dump(meta, open(os.path.join(a.out, 'sample_meta.json'), 'w'), ensure_ascii=False, indent=1)
print('done', len(meta))
