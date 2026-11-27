#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐音自回归排键模型：学 placement3 的 GRU 结构（上一音位置作为输入），
但换成 x_abs 绝对坐标 + 81 格分辨率重训。

placement3 在真谱节奏上实测 |x|中位240/同列17.4%/单列127，手感被用户认可；
本文件的差别只有数据口径（x_abs 全量）和分辨率（81 格 vs 27 桶）。
"""
import os
import sys
import json
import glob
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

NB = 21           # 屏幕内 21 个标准格位：-450..450，每格 45 坐标
HALF = 450.0
GRID = 45
BANW = float(os.environ.get('BANW', '2.0'))
GAPLO = 20.0      # 半叠禁带下界：note 宽约 177 坐标，20-135 的间距会半叠
GAPHI = 135.0
SBAN = float(os.environ.get('SBAN', '180'))   # 同拍两音最小间距：note 宽 177，低于这个就叠死
SBW = float(os.environ.get('SBW', '3.0'))
DROPDX = float(os.environ.get('DROPDX', '0.3'))
DIM = int(os.environ.get('DIMF', '16'))  # 16=带过去4拍走势(段记忆)，11=老版
# 真谱 21 格占比（40 首统计）：中间格 15.7%，两端 4~6%，逗号格 1.7~2%
PRIORC = np.array([4.4,6.2,4.7,1.9,5.1,4.1,4.3,5.5,3.7,1.7,15.7,1.7,3.6,5.4,4.4,4.3,5.2,1.9,4.9,6.4,4.6], np.float32)
PRIORW = float(os.environ.get('PRIORW', '1.0'))
STANDW = float(os.environ.get('STANDW', '0.0'))  # 原地档惩罚：纵连一张一串，数量翻倍就满屏竖线
PEB = 48          # prev 位置 embedding 维
MAXLEN = 4096     # 单曲最长音数


def x2g(x):
    return int(np.clip(round((x + HALF) / GRID), 0, NB - 1))


def g2x(g):
    return g * GRID - HALF


def load_song(path):
    d = json.load(open(path))
    ns = [n for n in d['notes'] if n.get('x_abs') is not None]
    ns.sort(key=lambda n: n['t'])
    return ns


def feats(ns):
    """(n, 11) 节奏特征，照 placement3 的 feats 但用秒/类型"""
    n = len(ns)
    t = np.array([z['t'] for z in ns], np.float32)
    ty = np.array([int(z.get('type', 1)) for z in ns], np.int64)
    du = np.array([float(z.get('dur', 0) or 0) for z in ns], np.float32)
    prev = np.diff(t, prepend=t[0] - 2.)
    nxt = np.diff(t, append=t[-1] + 2.)
    f = np.zeros((n, DIM), np.float32)
    f[:, 0] = np.log(prev + .01)
    f[:, 1] = np.log(nxt + .01)
    f[np.arange(n), 2 + np.clip(ty - 1, 0, 3)] = 1
    f[:, 6] = np.log(np.maximum(du, 0) + .01) * (ty == 2)
    f[:, 7] = prev < .005
    f[:, 8] = nxt < .005
    ends = np.maximum.accumulate(t + du)
    f[:, 9] = (np.r_[0, ends[:-1]] > t + .03)
    lo = np.searchsorted(t, t - 1.)
    hi = np.searchsorted(t, t + 1.)
    f[:, 10] = np.log((hi - lo) / 2 + .5)
    # 段记忆：把过去 4 拍的走法（方向）当作输入，模型才能复用同一套手势
    if DIM > 11:
        x = np.array([float(z.get('x_abs', 0) or 0) for z in ns], np.float32)
        d = np.r_[0., np.sign(np.diff(x))]
        for j in range(4):
            f[:, 11 + j] = np.r_[np.zeros(j, np.float32), d[:len(d) - j]] if j else d
    return f


class ARNet(nn.Module):
    def __init__(self, h=192):
        super().__init__()
        self.prev = nn.Embedding(NB + 1, PEB)
        self.inp = nn.Linear(DIM + PEB, h)
        self.g = nn.GRU(h, h, 2, batch_first=True, dropout=.2)
        self.o = nn.Linear(h, NB)

    def forward(self, f, pb, st=None):
        o, st = self.g(torch.relu(self.inp(torch.cat([f, self.prev(pb)], -1))), st)
        return self.o(o), st


def build_batch(songs, idx, noise=0.0):
    """songs: [(f, y, pb)]；返回 padding 后的张量与 mask"""
    n = max(len(songs[i][0]) for i in idx)
    B = len(idx)
    F = np.zeros((B, n, DIM), np.float32)
    PB = np.full((B, n), NB, np.int64)
    Y = np.zeros((B, n), np.int64)
    M = np.zeros((B, n), np.float32)
    for j, i in enumerate(idx):
        f, y, pb = songs[i]
        t = len(f)
        F[j, :t] = f
        Y[j, :t] = y
        M[j, :t] = 1.0
        PB[j, :t] = pb
        if noise > 0:
            # 只做小幅抖动，不再把 prev 随机替换成任意格：
            # 随机替换会教模型“prev 不可信就回中间最稳”，推理时踩到 x=0 就有 39% 概率原地不动
            jitter = (np.random.rand(t) < noise)
            if jitter.any():
                PB[j, :t][jitter] = np.clip(PB[j, :t][jitter] + np.random.randint(-1, 2, int(jitter.sum())), 0, NB)
    return (torch.from_numpy(F), torch.from_numpy(PB), torch.from_numpy(Y), torch.from_numpy(M))


def load_data(paths, limit=0):
    songs = []
    for p in paths:
        try:
            ns = load_song(p)
        except Exception:
            continue
        if len(ns) < 8 or len(ns) > MAXLEN:
            continue
        f = feats(ns)
        y = np.array([x2g(z['x_abs']) for z in ns], np.int64)
        pb = np.full(len(ns), NB, np.int64)
        pb[1:] = y[:-1]          # teacher forcing：上一音真实位置
        songs.append((f, y, pb))
    return songs


def evaluate(model, songs, idx):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='none')
    tot_x = []; ok = 0; n = 0; acc = 0
    with torch.no_grad():
        for it in range(0, len(idx), 64):
            F, PB, Y, M = build_batch(songs, idx[it:it + 64], noise=0)
            lg, _ = model(F, PB)
            loss = (ce(lg.reshape(-1, NB), Y.reshape(-1)) * M.reshape(-1)).sum() / M.sum()
            n += 1; acc += float(loss)
            pred = lg.argmax(-1)
            mask = M > 0
            tot_x.append(g2x(pred[mask].numpy()))
            ok += int((pred[mask] == Y[mask]).sum()); 
            ncor = int(mask.sum())
            if not hasattr(evaluate, '_n'):
                evaluate._n = 0
            evaluate._n += ncor
    x = np.concatenate(tot_x)
    evaluate._n = 0
    return acc / max(1, n), x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pars', default='/home/user/Doubao/chats/38453392661469954/_run_real_traj/abs_full/parsed')
    ap.add_argument('--limit', type=int, default=1500)
    a = ap.parse_args()
    files = sorted(glob.glob(a.pars + '/*.json'))
    random.seed(0); random.shuffle(files)
    files = files[:a.limit] if a.limit > 0 else files
    print('曲目', len(files), flush=True)
    songs = load_data(files)
    print('可用', len(songs), '总音', sum(len(s[0]) for s in songs), flush=True)
    n = len(songs)
    vidx = list(range(int(n * 0.95), n))
    tridx = list(range(0, int(n * 0.95)))
    model = ARNet()
    if os.environ.get('RESUME') and os.path.exists(os.environ.get('CKPT_OUT', 'arnet.pt')):
        model.load_state_dict(torch.load(os.environ.get('CKPT_OUT', 'arnet.pt'), map_location='cpu'))
        print('热启动', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.03)
    GXT = torch.tensor([g2x(i) for i in range(NB)], dtype=torch.float32)
    BS = 16
    EPOCH = int(os.environ.get('EPOCH', 4))
    for ep in range(EPOCH):
        model.train(); random.shuffle(tridx)
        tot = 0.; nb = 0
        SS_EP = int(os.environ.get('SS_EP', '3'))
        SS_MAX = float(os.environ.get('SS_MAX', '1.0'))
        for it in range(0, len(tridx), BS):
            F, PB, Y, M = build_batch(songs, tridx[it:it + BS])
            if DROPDX > 0:
                # 随机抹掉段记忆维：逼它有时也能不看历史、自己起新走法
                keep = (torch.rand(F.shape[0], F.shape[1], 1) > DROPDX).float()
                F[:, :, 11:] = F[:, :, 11:] * keep
            rate = 0.0 if ep < SS_EP else min(SS_MAX, (ep - SS_EP + 1) / 3.0)
            if rate > 0:
                # 自采样：拿模型自己的采样结果当上一音，治 teacher forcing 后遗症
                with torch.no_grad():
                    lg1, _ = model(F, PB)
                    pr = torch.softmax(lg1, -1).reshape(-1, NB)
                    pred = torch.multinomial(pr, 1).reshape(PB.shape)
                    mix = torch.rand(PB.shape) < rate
                    PB = PB.clone()
                    PB[:, 1:] = torch.where(mix[:, 1:], pred[:, :-1], PB[:, 1:])
                    if DIM > 11:
                        # motif 维也换成自己生成的走势，否则训练看到真历史、推理自己乱走
                        for j in range(min(4, DIM - 11)):
                            if j == 0:
                                src = pred - torch.roll(pred, 1, 1)
                            else:
                                src = torch.roll(pred, j, 1) - torch.roll(pred, j + 1, 1)
                            src = src.sign().float()
                            F[:, :, 11 + j] = torch.where(mix, src, F[:, :, 11 + j])
            lg, _ = model(F, PB)
            if PRIORW > 0:
                # 位置先验：真谱每格的占比当 logit 偏置，模型不至于一路推到边上
                lp = torch.from_numpy(np.log(PRIORC / PRIORC.sum() + 1e-8)).float()
                lg = lg + PRIORW * lp[None, None, :]
            banpen = 0.; sbpen = 0.
            if GAPHI > 0:
                # 半叠禁带：跨拍不许落在距上一音 20..135 的区间（真谱这段只占 4%）
                # 不遮 logits，直接把禁带类上的概率压到 0，权重才会真的记住这条约束
                dprev = (GXT[None, None, :] - GXT[PB.clamp(max=NB - 1)].unsqueeze(-1)).abs()
                samebeat = (F[:, :, 7] > 0)
                ban = (dprev > GAPLO) & (dprev < GAPHI) & (~samebeat.unsqueeze(-1))
                ban[:, 0, :] = False
                prob = torch.softmax(lg, -1)
                banpen = (prob * ban.float()).sum(-1)
                banpen = (banpen * M).sum() / M.sum()
                # 同拍多押最小间距：同一拍的第二个音起，不许落在前一颗 180 以内
                sb = (dprev < SBAN) & samebeat.unsqueeze(-1)
                sb[:, 0, :] = False
                sbpen = (prob * sb.float()).sum(-1)
                sbpen = (sbpen * M).sum() / M.sum()

            M2 = M
            if STANDW > 0:
                # 真谱原地串 1638 个、平均 1.94 拍；模型学到“一直原地”就成了满屏纵连
                # 只降权“连续第 3 拍起仍原地”的真值样本，不碰 softmax，避免三项惩罚打架
                zg = int(round(HALF / GRID))
                y0 = (Y == zg)
                y1 = torch.zeros_like(y0); y1[:, 1:] = y0[:, :-1]
                y2 = torch.zeros_like(y0); y2[:, 2:] = y0[:, :-2]
                deep = y0 & y1 & y2
                w = torch.where(deep, torch.full_like(M, 1.0 - STANDW), torch.ones_like(M))
                M2 = M * w
            loss = (ce(lg.reshape(-1, NB), Y.reshape(-1)) * M2.reshape(-1)).sum() / M2.sum() + BANW * banpen + SBW * sbpen
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            if it % 200 == 0:
                print('  ep%d %d/%d loss %.4f' % (ep, it, len(tridx), tot / max(1, nb)), flush=True)
        print('ep%d 训练损失 %.4f' % (ep, tot / max(1, nb)), flush=True)
        model.eval()
        with torch.no_grad():
            vl = 0.; vn = 0; xs = []
            for it in range(0, len(vidx), 64):
                F, PB, Y, M = build_batch(songs, vidx[it:it + 64], noise=0)
                lg, _ = model(F, PB)
                loss = (ce(lg.reshape(-1, NB), Y.reshape(-1)) * M.reshape(-1)).sum() / M.sum()
                vl += float(loss); vn += 1
                pred = lg.argmax(-1); mask = M > 0
                xs.append(g2x(pred[mask].numpy()))
            x = np.concatenate(xs)
            print('  验证 音 %d |x|中位%.0f(真%.0f) <135占%.1f%%(真%.1f%%) >400占%.1f%%(真%.1f%%) loss %.4f' % (
                len(x), np.median(np.abs(x)), 253, np.mean(np.abs(x) < 135) * 100, 23.2,
                np.mean(np.abs(x) > 400) * 100, 21.9, vl / max(1, vn)), flush=True)
        torch.save(model.state_dict(), os.environ.get('CKPT_OUT', 'arnet.pt'))
        print('  已存', os.environ.get('CKPT_OUT', 'arnet.pt'), flush=True)


if __name__ == '__main__':
    main()
