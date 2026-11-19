#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手感尺子：不看单颗音离多远，看相邻音之间横向怎么动。
四项指标：步长幅度、左右交替率、位移分布熵、占用格子数。
"""
import argparse, json
import numpy as np, torch
from train_parallel_cls import ParNet, parallel_eval
from train_singleline import SeqNet
from train_delta_cls import DeltaNet, ar_eval as delta_ar

KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13]

def metrics(P, M):
    """P: 连续 x(-1..1)，M: mask。返回手感指标"""
    m = M.astype(bool)
    step_all, sign_flip, cover = [], [], []
    for i in range(len(P)):
        p = P[i][m[i]]
        if len(p) < 8:
            continue
        d = np.diff(p)
        ad = np.abs(d)
        # 只统计有明显移动的步（>2px 当量，即 0.003）
        step_all.append(ad)
        moving = ad > 0.003
        if moving.sum() >= 3:
            s = np.sign(d[moving])
            flip = float(np.sum(np.r_[False, s[1:] != s[:-1]]))
            sign_flip.append((flip, len(s) - 1))
    if not step_all:
        return None
    ad = np.concatenate(step_all)
    mean_step = float(np.mean(ad)) * 675
    med_step = float(np.median(ad)) * 675
    p90 = float(np.percentile(ad, 90)) * 675
    if sign_flip:
        f, t = [sum(x[0] for x in sign_flip), sum(x[1] for x in sign_flip)]
        flip_rate = f / max(t, 1)
    else:
        flip_rate = float('nan')
    # 占用格子数（把 x 量化成 81 格）
    q = np.clip(np.round((P + 1) / 2 * 80), 0, 80).astype(int)
    used = [len(np.unique(q[i][m[i]])) for i in range(len(P)) if m[i].sum() >= 8]
    return dict(mean_step=mean_step, med_step=med_step, p90_step=p90,
                flip_rate=flip_rate, grid_used=float(np.mean(used)) if used else 0.0)


def load_model(path, arch):
    a = argparse.Namespace()
    ck = torch.load(path, map_location='cpu')
    nf = ck.get('feat', 11); nb = ck.get('bins', 81)
    if arch == 'parallel':
        net = ParNet(nf=nf, nb=nb)
    elif arch == 'delta':
        net = DeltaNet(nb=nb)
    else:
        net = SeqNet(nf=nf)
    net.load_state_dict(ck['model'])
    net.eval()
    return net, nb


def predict(net, arch, F, Y, M, nb, guard=False, sample=False, prior=None, alpha=0.0, scale=1.0):
    if arch == 'parallel':
        Fp = F[:, :, KEEP] if F.shape[-1] > 11 else F
        return parallel_eval(net, Fp, Y, M, nb)[1]
    if arch == 'delta':
        return delta_ar(net, F, Y, M, nb, sample=sample, guard=guard, prior=prior, alpha=alpha, scale=scale)[1]
    with torch.no_grad():
        return net(torch.from_numpy(F.astype(np.float32))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--arch', default='parallel')
    ap.add_argument('--guard', type=int, default=0)
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--alpha', type=float, default=0.0)
    ap.add_argument('--scale', type=float, default=1.0)
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True)
    F = d['F'].astype(np.float32); Y = d['Y'].astype(np.float32); M = d['M'].astype(np.float32)
    fid = d['fid'].astype(str)
    split = json.load(open(a.split))
    va = np.isin(fid, list(split['val'])); te = np.isin(fid, list(split['test']))
    rows = np.where(va | te)[0]
    Fs, Ys, Ms = F[rows], Y[rows], M[rows]
    net, nb = load_model(a.model, a.arch)
    from train_parallel_cls import to_bin
    tr = ~(va | te)
    prior = np.bincount(to_bin(Y[tr], nb).ravel(), minlength=nb).astype(np.float64)
    P = predict(net, a.arch, Fs, Ys, Ms, nb, guard=bool(a.guard), sample=bool(a.sample), prior=prior, alpha=a.alpha, scale=a.scale)

    mt = metrics(Ys, Ms)
    mg = metrics(P.astype(np.float32), Ms)
    print('指标              真谱        生成      说明')
    print('平均步长(px)   %8.1f %10.1f   相邻两音的横向跳幅' % (mt['mean_step'], mg['mean_step']))
    print('中位步长(px)   %8.1f %10.1f   一半的移动小于这个' % (mt['med_step'], mg['med_step']))
    print('90分位(px)     %8.1f %10.1f   大幅跳动的量级' % (mt['p90_step'], mg['p90_step']))
    print('左右交替率     %8.2f %10.2f   方向反转的步数占比' % (mt['flip_rate'], mg['flip_rate']))
    print('占用格子数     %8.1f %10.1f   128 音用到了几个横向格子' % (mt['grid_used'], mg['grid_used']))
    out = dict(truth=mt, gen=mg)
    out['mae'] = float(np.mean(np.abs(P.astype(np.float32)[Ms.astype(bool)] - Ys[Ms.astype(bool)]) * 675))
    print('MAE(px)        %8.1f' % out['mae'])
    json.dump(out, open('/home/user/Doubao/chats/38453392661469954/_run_real_traj/handfeel_metric.json', 'w'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
