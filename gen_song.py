#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整曲 SongNet 组装：读谱面目录，整曲算 17 维特征，SongNet 前向出位置，
判定线事件原样深拷贝，只换音符 positionX。开卷特征（真值 x 算 prev/同伴列）。"""
import os, json, copy, shutil, argparse, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_song as T
from gen_chart_from_model import collect, chord_counts, dens_counts, norm_deg
from build_song_data import X_SCALE, ANG


def feats_song(rec):
    x = np.array([r['x'] for r in rec], np.float32) / X_SCALE
    t = np.array([r['t'] for r in rec], np.float32)
    ty = np.array([r['type'] for r in rec], np.int32)
    dur = np.array([r['dur'] for r in rec], np.float32)
    ln = np.array([r['li'] for r in rec], np.int16)
    n = len(x)
    chord = chord_counts(t)
    dens = dens_counts(t, 4.0)
    uq, inv = np.unique(t, return_inverse=True)
    rank = np.zeros(n, np.float32); span = np.zeros(n, np.float32); meand = np.zeros(n, np.float32)
    for g in range(len(uq)):
        idx = np.where(inv == g)[0]
        if len(idx) == 1:
            continue
        xs = x[idx]; o = np.argsort(xs); ranks = np.linspace(-1., 1., len(idx))
        for p, ii in enumerate(o):
            rank[idx[ii]] = ranks[p]
        span[idx] = xs.max() - xs.min()
        meand[idx] = float(np.mean(np.abs(xs[:, None] - xs[None, :])))
    f = np.zeros((n, 17), np.float32)
    prev = prev2 = 0.0; run = 0
    for k in range(n):
        dt = 0.0 if k == 0 else max(0.0, float(t[k] - t[k - 1]))
        if k == 0:
            lc, lgap, run = 0., 0., 1
        else:
            lc = 0. if int(ln[k]) == int(ln[k - 1]) else 1.
            lgap = np.log1p(abs(int(ln[k]) - int(ln[k - 1])) / 10.) if lc else 0.
            run = 1 if lc else run + 1
        f[k] = [np.log1p(dt), *(ty[k] == np.arange(1, 5)), np.log1p(max(0., dur[k])),
                np.log1p(chord[k]), np.log1p(dens[k]), prev, prev2, prev - prev2,
                lc, lgap, np.log1p(run), rank[k], np.log1p(span[k] * X_SCALE),
                np.log1p(meand[k] * X_SCALE)]
        prev2 = prev; prev = x[k]
    return f.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--blind', action='store_true', help='闭卷：历史/同伴位置列置零')
    a = ap.parse_args()

    info = {}
    for line in open(os.path.join(a.chart, 'info.yml'), encoding='utf-8'):
        if ':' in line:
            k, v = line.split(':', 1); info[k.strip()] = v.strip()
    cj = info.get('chart', 'chart.json')
    chart = json.load(open(os.path.join(a.chart, cj), encoding='utf-8'))

    rec = collect(chart)
    keep = [r for r in rec if abs(r['theta']) <= ANG and abs(r['theta_end']) <= ANG]
    print('总音', len(rec), '可用', len(keep))

    ck = torch.load(a.model, map_location='cpu')
    net = T.SongNet(nf=ck.get('nf', 17), hidden=ck.get('hidden', 64), layers=ck.get('layers', 1))
    net.load_state_dict(ck['model']); net.eval()

    f = feats_song(keep)
    if a.blind:
        f[:, [8, 9, 10, 14, 15, 16]] = 0.0
    with torch.no_grad():
        lg = net(torch.from_numpy(f).unsqueeze(0), torch.tensor([len(keep)]))[0].numpy()
    pred = lg.argmax(-1)
    P = pred / (T.NB - 1) * 2.0 - 1.0

    newx = {(keep[i]['li'], keep[i]['ni']): float(P[i] * X_SCALE) for i in range(len(keep))}
    new_lines = []; changed = 0
    for li, jl in enumerate(chart.get('judgeLineList', [])):
        nl = {k: copy.deepcopy(v) for k, v in jl.items() if k != 'notes'}
        new_notes = []
        for ni, nt in enumerate(jl.get('notes', [])):
            n2 = copy.deepcopy(nt)
            if (li, ni) in newx:
                n2['positionX'] = round(newx[(li, ni)], 2); changed += 1
            new_notes.append(n2)
        nl['notes'] = new_notes; nl['numOfNotes'] = len(new_notes)
        new_lines.append(nl)

    new_chart = dict(BPMList=copy.deepcopy(chart.get('BPMList', [])),
                     META=copy.deepcopy(chart.get('META', {})),
                     judgeLineGroup=copy.deepcopy(chart.get('judgeLineGroup', [])),
                     judgeLineList=new_lines,
                     multiLineString=copy.deepcopy(chart.get('multiLineString', '')),
                     multiScale=copy.deepcopy(chart.get('multiScale', 1.0)),
                     timeTags=copy.deepcopy(chart.get('timeTags', [])))
    print('改写音数', changed)

    os.makedirs(a.out, exist_ok=True)
    for fn in os.listdir(a.chart):
        if fn.endswith(('.mp3', '.jpg', '.png', '.ogg')):
            shutil.copy2(os.path.join(a.chart, fn), os.path.join(a.out, fn))
    newjson = 'generated_chart.json'
    json.dump(new_chart, open(os.path.join(a.out, newjson), 'w'), ensure_ascii=False)
    rep = {}
    for line in open(os.path.join(a.chart, 'info.yml'), encoding='utf-8'):
        if ':' not in line:
            continue
        k, v = line.split(':', 1); rep[k.strip()] = v.strip()
    rep['chart'] = newjson; rep['name'] = rep.get('name', 'gen') + ' (song)'
    with open(os.path.join(a.out, 'info.yml'), 'w', encoding='utf-8') as fp:
        for k, v in rep.items():
            fp.write('%s: %s\n' % (k, v if v not in (None, 'null', '') else 'null'))
    print('输出目录', a.out)


if __name__ == '__main__':
    main()
