#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从原始 zip 重解析，修正 x 口径：
- x_rel = note.positionX（相对判定线中心偏移，旧口径）
- x_abs = 线中心横移 px(t) + positionX（屏幕绝对横向位置，新口径）
line_state_at 已能返回线中心 px（moveXEvents 累加 + 父线叠加），旧解析只取了角度。
输出 angle_abs/{parsed,filtered}，note 含 t/x_rel/x_abs/type/dur/line_id/theta/theta_end。"""
import os, glob, json, zipfile, argparse, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from easing_lib import line_state_at
from phira_fetch import rpe_time_to_beat

ANG_LIM = 5.0


def beat_to_sec(beat, segs):
    total = 0.0
    for i, (sb, bpm) in enumerate(segs):
        if i + 1 < len(segs):
            nb = segs[i + 1][0]
            if beat < nb:
                return total + (beat - sb) * 60.0 / bpm
            total += (nb - sb) * 60.0 / bpm
        else:
            return total + (beat - sb) * 60.0 / bpm
    return total


def parse_angle_abs(chart):
    bpm_raw = chart.get('BPMList', [])
    segs = sorted([(rpe_time_to_beat(b['startTime']), float(b['bpm']))
                   for b in bpm_raw], key=lambda x: x[0])
    notes = []
    for li, jl in enumerate(chart.get('judgeLineList', []) or []):
        if not isinstance(jl, dict):
            continue
        cache = {}
        for n in jl.get('notes', []) or []:
            if n.get('isFake', 0) == 1:
                continue
            typ = int(n.get('type', 1))
            tb = rpe_time_to_beat(n['startTime'])
            t = beat_to_sec(tb, segs)
            px, py, th = line_state_at(jl, chart, tb, cache)
            th = ((th + 180) % 360) - 180
            dur = 0.0
            the = th
            if typ == 2 and isinstance(n.get('endTime'), list):
                eb = rpe_time_to_beat(n['endTime'])
                dur = max(0.0, beat_to_sec(eb, segs) - t)
                _, _, the = line_state_at(jl, chart, eb, cache)
                the = ((the + 180) % 360) - 180
            x_rel = float(n.get('positionX', 0.0))
            notes.append({'t': round(t, 6), 'x_rel': x_rel,
                          'x_abs': round(px + x_rel, 3),
                          'type': typ, 'dur': round(dur, 6),
                          'line_id': li, 'theta': round(float(th), 3),
                          'theta_end': round(float(the), 3)})
    notes.sort(key=lambda n: n['t'])
    return notes


def pick_chart(zf):
    best, bestn = None, -1
    for nm in zf.namelist():
        base = nm.rsplit('/', 1)[-1].lower()
        if not base.endswith('.json') or base in ('info.json', 'extra.json'):
            continue
        try:
            obj = json.loads(zf.read(nm))
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get('judgeLineList'):
            n = sum(len(j.get('notes', []) or []) for j in obj['judgeLineList']
                    if isinstance(j, dict))
            if n > bestn:
                best, bestn = obj, n
    return best


def one(zp, outdir):
    fid = os.path.basename(zp).split('_', 1)[0]
    if not fid.isdigit():
        return None
    try:
        with zipfile.ZipFile(zp) as zf:
            ch = pick_chart(zf)
    except Exception as e:
        return (fid, 'ERR', repr(e)[:80], 0, 0)
    if not ch:
        return (fid, 'NOCHART', '', 0, 0)
    notes = parse_angle_abs(ch)
    good = [n for n in notes if abs(n['theta']) <= ANG_LIM and abs(n['theta_end']) <= ANG_LIM]
    json.dump({'fid': fid, 'notes': notes},
              open(os.path.join(outdir, 'parsed', fid + '.json'), 'w'), ensure_ascii=False)
    json.dump({'fid': fid, 'notes': good},
              open(os.path.join(outdir, 'filtered', fid + '.json'), 'w'), ensure_ascii=False)
    return (fid, 'OK', '', len(notes), len(good))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zips', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    zps = sorted(glob.glob(os.path.join(a.zips, '*', '*.zip'))) + \
        sorted(glob.glob(os.path.join(a.zips, '*.zip')))
    if a.limit:
        zps = zps[:a.limit]
    os.makedirs(os.path.join(a.out, 'parsed'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'filtered'), exist_ok=True)
    print('zips', len(zps), flush=True)
    from concurrent.futures import ProcessPoolExecutor
    stats = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one, zps, [a.out] * len(zps)):
            if r:
                stats.append(r)
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), ensure_ascii=False)
    print('ok', sum(s[1] == 'OK' for s in stats), 'nochart',
          sum(s[1] == 'NOCHART' for s in stats), 'err', sum(s[1] == 'ERR' for s in stats))
    print('notes', sum(s[3] for s in stats), 'valid', sum(s[4] for s in stats))


if __name__ == '__main__':
    main()
