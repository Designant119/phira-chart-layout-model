#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从模型生成横向位置，组装一份全新的谱面 json。
判定线事件（移动/旋转/缩放/父线等）原样拷贝，只重排音符 positionX。
不改动原谱文件，输出独立目录。
"""
import os, sys, json, copy, shutil, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from easing_lib import line_state_at
from phira_fetch import rpe_time_to_beat
from build_angle_parsed import beat_to_sec
from train_delta_cls import DeltaNet, ar_eval
from train_parallel_cls import to_bin

W, STRIDE, XS = 128, 64, 675.0
ANG = 5.0


def chord_counts(t):
    n = len(t); out = np.zeros(n, np.float32)
    if n == 0: return out
    uq, inv = np.unique(t, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float32)
    return cnt[inv]


def dens_counts(t, win=4.0):
    n = len(t); out = np.zeros(n, np.float32)
    if n == 0: return out
    s = np.searchsorted(t, t - win, side='left')
    e = np.searchsorted(t, t + win, side='right')
    return (e - s).astype(np.float32)


def chord_rank(t):
    """同拍组内按顺序给序号 -1..1；同伴位置在推理时未知，跨度/距离填 0。"""
    n = len(t); rank = np.zeros(n, np.float32)
    uq, inv = np.unique(t, return_inverse=True)
    for g in range(len(uq)):
        idx = np.where(inv == g)[0]
        if len(idx) > 1:
            rank[idx] = np.linspace(-1.0, 1.0, len(idx))
    return rank


def norm_deg(v):
    return ((float(v) + 180.0) % 360.0) - 180.0


def collect(chart):
    bpm = chart.get('BPMList', [])
    segs = sorted([(rpe_time_to_beat(b['startTime']), float(b['bpm'])) for b in bpm], key=lambda x: x[0])
    rec = []
    for li, jl in enumerate(chart.get('judgeLineList', [])):
        cache = {}
        for ni, n in enumerate(jl.get('notes', [])):
            if n.get('isFake', 0) == 1:
                continue
            typ = int(n.get('type', 1))
            tb = rpe_time_to_beat(n['startTime'])
            t = beat_to_sec(tb, segs)
            th = norm_deg(line_state_at(jl, chart, tb, cache)[2])
            dur = 0.0
            the = th
            if typ == 2 and isinstance(n.get('endTime'), list):
                eb = rpe_time_to_beat(n['endTime'])
                dur = max(0.0, beat_to_sec(eb, segs) - t)
                the = norm_deg(line_state_at(jl, chart, eb, cache)[2])
            rec.append(dict(li=li, ni=ni, t=t, x=float(n.get('positionX', 0)), type=typ,
                            dur=dur, theta=th, theta_end=the, nt=n))
    rec.sort(key=lambda r: r['t'])
    return rec


def build_feats(rec, nf=14):
    x = np.array([r['x'] for r in rec], np.float32) / XS
    t = np.array([r['t'] for r in rec], np.float32)
    ty = np.array([r['type'] for r in rec], np.int32)
    dur = np.array([r['dur'] for r in rec], np.float32)
    ln = np.array([r['li'] for r in rec], np.int16)
    chord = chord_counts(t)
    dens = dens_counts(t, 4.0)
    crank = chord_rank(t)
    n = len(x)
    F, IDX = [], []
    for s in range(0, max(1, n - W + 1), STRIDE):
        e = min(n, s + W)
        f = np.zeros((W, nf), np.float32)
        prev = prev2 = 0.0
        run = 0
        for k in range(W):
            i = s + k
            if i >= e:
                break
            dt = 0.0 if k == 0 else max(0.0, float(t[i] - t[i - 1]))
            if k == 0:
                lc, lgap, run = 0.0, 0.0, 1
            else:
                lc = 0.0 if int(ln[i]) == int(ln[i - 1]) else 1.0
                lgap = np.log1p(abs(int(ln[i]) - int(ln[i - 1]) / 10.0)) if lc else 0.0
                run = 1 if lc else run + 1
            feat = [np.log1p(dt), *(ty[i] == np.arange(1, 5)), np.log1p(max(0.0, dur[i])),
                    np.log1p(chord[i]), np.log1p(dens[i]), prev, prev2, prev - prev2,
                    lc, lgap, np.log1p(run), crank[i], 0.0, 0.0]
            f[k] = feat[:nf]
            prev2 = prev
            prev = x[i]
        F.append(f)
        IDX.append((s, e))
        if e >= n:
            break
    return np.stack(F), IDX, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True, help='含 info.yml 的谱面目录')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--alpha', type=float, default=2.0)
    ap.add_argument('--scale', type=float, default=1.15)
    a = ap.parse_args()

    info = {}
    for line in open(os.path.join(a.chart, 'info.yml'), encoding='utf-8'):
        if ':' in line:
            k, v = line.split(':', 1)
            info[k.strip()] = v.strip()
    cj = info.get('chart', 'chart.json')
    chart = json.load(open(os.path.join(a.chart, cj), encoding='utf-8'))

    rec = collect(chart)
    keep = [r for r in rec if abs(r['theta']) <= ANG and abs(r['theta_end']) <= ANG]
    print('总音', len(rec), '可用', len(keep))
    if len(keep) < 16:
        print('音太少，退出'); return

    ck = torch.load(a.model, map_location='cpu')
    nb = ck.get('bins', 81)
    nf = ck.get('nf', 14)
    F, IDX, n = build_feats(keep, nf=nf)
    net = DeltaNet(nb=nb, nf=nf); net.load_state_dict(ck['model'])

    Y = np.zeros((len(F), W), np.float32); M = np.zeros_like(Y)
    for wi, (s, e) in enumerate(IDX):
        Y[wi, :e - s] = [r['x'] / XS for r in keep[s:e]]
        M[wi, :e - s] = 1
    prior = np.bincount(to_bin(Y[M == 1], nb), minlength=nb).astype(np.float64)
    _, P = ar_eval(net, F, Y, M, nb, sample=True, guard=True, prior=prior, alpha=a.alpha, scale=a.scale)

    # 逐窗取前半段，最后一窗取全部
    newx = {}
    for wi, (s, e) in enumerate(IDX):
        cnt = (e - s) if e >= n else STRIDE
        for k in range(cnt):
            idx = s + k
            if idx < n:
                newx[(keep[idx]['li'], keep[idx]['ni'])] = float(P[wi, k] * XS)

    # 全新组装：判定线事件原样深拷贝，只换音符位置
    new_lines = []
    changed = 0
    for li, jl in enumerate(chart.get('judgeLineList', [])):
        nl = {k: copy.deepcopy(v) for k, v in jl.items() if k != 'notes'}
        new_notes = []
        for ni, nt in enumerate(jl.get('notes', [])):
            n2 = copy.deepcopy(nt)
            key = (li, ni)
            if key in newx:
                n2['positionX'] = round(newx[key], 2)
                changed += 1
            new_notes.append(n2)
        nl['notes'] = new_notes
        nl['numOfNotes'] = len(new_notes)
        new_lines.append(nl)

    new_chart = dict(
        BPMList=copy.deepcopy(chart.get('BPMList', [])),
        META=copy.deepcopy(chart.get('META', {})),
        judgeLineGroup=copy.deepcopy(chart.get('judgeLineGroup', [])),
        judgeLineList=new_lines,
        multiLineString=copy.deepcopy(chart.get('multiLineString', '')),
        multiScale=copy.deepcopy(chart.get('multiScale', 1.0)),
        timeTags=copy.deepcopy(chart.get('timeTags', [])),
    )
    print('改写音数', changed)

    os.makedirs(a.out, exist_ok=True)
    # 只带音频和封面，不带原谱 json
    for f in os.listdir(a.chart):
        if f.endswith(('.mp3', '.jpg', '.png', '.ogg')):
            shutil.copy2(os.path.join(a.chart, f), os.path.join(a.out, f))
    newjson = 'generated_chart.json'
    json.dump(new_chart, open(os.path.join(a.out, newjson), 'w'), ensure_ascii=False)

    rep = {}
    for line in open(os.path.join(a.chart, 'info.yml'), encoding='utf-8'):
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        rep[k.strip()] = v.strip()
    rep['chart'] = newjson
    rep['name'] = rep.get('name', 'gen') + ' (model)'
    with open(os.path.join(a.out, 'info.yml'), 'w', encoding='utf-8') as fp:
        for k, v in rep.items():
            fp.write('%s: %s\n' % (k, v if v not in (None, 'null', '') else 'null'))
    print('输出目录', a.out)


if __name__ == '__main__':
    main()
