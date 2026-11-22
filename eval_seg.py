#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证段级模型的采样效果，并试"相邻拍距离约束"能不能把步长收回来。"""
import gzip
import json
import random
import numpy as np
import torch
from train_seg import SegNet, featurize, g2x, NB, MAXLEN

HALF = 675.0
COLW = 2 * HALF / (NB - 1)


def load_raw(path, cap=30000):
    rows = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r['zero'] > 0.3 or r['beats'] < 6 or r['beats'] > MAXLEN:
                continue
            rows.append(r)
            if cap and len(rows) >= cap:
                break
    return rows


def run(model, rows, mode='free', temp=1.0):
    preds, gts = [], []
    for it in range(0, len(rows), 256):
        rs = rows[it:it + 256]
        data = [(*featurize(r), r['fid']) for r in rs]
        n = max(len(d[1]) for d in data)
        Fb = np.zeros((len(rs), n, 9), dtype=np.float32)
        Yb = np.zeros((len(rs), n), dtype=np.int64)
        for i, (f, y, _) in enumerate(data):
            Fb[i, :len(y)] = f
            Yb[i, :len(y)] = y
        with torch.no_grad():
            lg = model(torch.from_numpy(Fb))
        for i, r in enumerate(rs):
            k = len(r['chord'])
            p = lg[i, :k] / temp
            med_iv = float(np.median(r['iv'])) if r['iv'] else 0.2
            tgt_g = float(np.clip(263 + 245 * (med_iv - 0.094), 250, 300)) / COLW
            gs = []
            prev = None
            for t in range(k):
                lp = p[t].clone()
                if mode == 'anchor' and prev is not None:
                    grid = torch.arange(NB, dtype=torch.float32)
                    d = (grid - prev).abs()
                    pen = ((d - tgt_g) / max(3.0, tgt_g * 0.45)) ** 2
                    near = torch.clamp(4.0 - d, min=0) * 3.0
                    lp = lp - pen - near
                prob = torch.softmax(lp, dim=-1)
                s = int(torch.multinomial(prob, 1))
                gs.append(s)
                prev = s
            preds.append(np.array(gs))
            gts.append(Yb[i, :k])
    return np.concatenate(preds), np.concatenate(gts)


def report(tag, ps, gs):
    xs = np.array([g2x(g) for g in ps])
    gx = np.array([g2x(g) for g in gs])
    d = np.abs(np.diff(xs)); d = d[d > 1e-6]
    gd = np.abs(np.diff(gx)); gd = gd[gd > 1e-6]
    alt = float((np.sign(np.diff(xs)[:-1]) * np.sign(np.diff(xs)[1:]) < 0).mean()) if len(d) > 2 else 0.5
    galt = float((np.sign(np.diff(gx)[:-1]) * np.sign(np.diff(gx)[1:]) < 0).mean()) if len(gd) > 2 else 0.5
    print('%-8s |x|%.0f/%.0f  中间带%.1f%%/%.1f%%  步长%.0f/%.0f  换向%.2f/%.2f' % (
        tag, np.median(np.abs(xs)), np.median(np.abs(gx)),
        100 * (np.abs(xs) < 135).mean(), 100 * (np.abs(gx) < 135).mean(),
        np.median(d), np.median(gd), alt, galt))


def main():
    path = '/home/user/Doubao/chats/38453392661469954/_run_real_traj/seg_index.jsonl.gz'
    rows = load_raw(path, 30000)
    fids = sorted(set(r['fid'] for r in rows))
    random.seed(7)
    val_fid = set(random.sample(fids, max(1, len(fids) // 20)))
    va = [r for r in rows if r['fid'] in val_fid]
    print('验证段 %d' % len(va))
    m = SegNet()
    m.load_state_dict(torch.load('segnet.pt'))
    m.eval()
    for mode in ('free', 'anchor'):
        ps, gs = run(m, va, mode)
        report(mode, ps, gs)


if __name__ == '__main__':
    main()
