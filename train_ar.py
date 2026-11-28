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
GAPLO = float(os.environ.get('GAPLO', '20.0'))   # 禁带下界
GAPHI = float(os.environ.get('GAPHI', '135.0'))  # 禁带上界
# 实测：note 视觉宽约 20 坐标（1280 宽下 20px）。0~20 是一根竖条，45 还像一列，89 才分成两列。
# 用户口径：要 270 以上，那就把禁带整个抬到 270。
SBAN = float(os.environ.get('SBAN', '180'))   # 同拍两音最小间距：note 宽 177，低于这个就叠死
SBW = float(os.environ.get('SBW', '3.0'))
DROPDX = float(os.environ.get('DROPDX', '0.3'))
DIM = int(os.environ.get('DIMF', '16'))  # 16=带过去4拍走势(段记忆)，11=老版
# 真谱 21 格占比（40 首统计）：中间格 15.7%，两端 4~6%，逗号格 1.7~2%
PRIORC = np.array([4.4,6.2,4.7,1.9,5.1,4.1,4.3,5.5,3.7,1.7,15.7,1.7,3.6,5.4,4.4,4.3,5.2,1.9,4.9,6.4,4.6], np.float32)
PRIORW = float(os.environ.get('PRIORW', '1.0'))
# 真谱跨拍单音的 |dx| 直方图（每档 45 坐标，40 首 19498 对）：
# 0~45 有 19.9%、90~135 有 5.9%、135~180 有 8.3%…… 这是唯一一条“奖励分布”的项
# 去掉 0~45 那档：它是本轮崩掉的元凶（模型把所有音塞进最大桶就能骗低 loss）
REALDIST = np.array([3.72,5.91,8.26,7.81,9.20,8.02,2.52,5.70,4.72,6.13,2.44,5.53,1.65,1.54,3.04,1.05,0.75,0.0,0.0,1.07], np.float32)
REALDIST = REALDIST / REALDIST.sum()
DISTW = float(os.environ.get('DISTW', '0.0'))
# 用户口径：要 270 以上。硬禁会撞墙不动，改成软奖励：落在 >= FARX 的格子直接减 loss
FARW = float(os.environ.get('FARW', '0.0'))
FARX = float(os.environ.get('FARX', '270.0'))
# 相对坐标约束：这一步必须迈 >= MINSTEP 个坐标（=6 格）。直接在 logit 上把迈不到的格子封掉，
# 不在 loss 里加罚。训练和推理用同一套约束，模型学的是「剩下的格子里挑哪个」
MINSTEP = float(os.environ.get('MINSTEP', '0.0'))
IDIAG = int(os.environ.get('IDIAG', '0'))
# 真谱不是随机跳：同方向平均连 3 拍、最长 22 拍。加方向连贯奖励，
# 让「延续上一拍的方向」直接减 loss。人味来自这个，不是来自间距本身
DIRW = float(os.environ.get('DIRW', '0.0'))  # 每步最小位移（0=不约束，270=六格)
HOLDW = float(os.environ.get('HOLDW', '0.0'))  # 长押占线惩罚
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


global songs_ns
songs_ns = {}


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
    # 长押占位：真谱 hold 占 11.6%，p95 1.2s、最长 42s；模型看不到“这条线被占着”，
    # 就会把后续音继续往那条线上排，视觉上塌成一根长条
    if DIM > 11:
        x = np.array([float(z.get('x_abs', z.get('x', 0)) or 0) for z in ns], np.float32)
        dtype = np.array([int(z.get('type', 1)) for z in ns], np.int64)
        duro = np.array([float(z.get('dur', 0) or 0) for z in ns], np.float32)
        live_x = np.zeros(n, np.float32)
        live_on = np.zeros(n, np.float32)
        live_rem = np.zeros(n, np.float32)
        live_sz = np.zeros(n, np.float32)
        # 用最短存活链：按时间顺序维护还活着的 hold，取最近一条
        act = []  # [(end_t, x, size)]
        for i in range(n):
            act = [a for a in act if a[0] > t[i]]
            # 没有活 hold 时填“上一音自己的位置”，不填 0：
            # 0 会同时表示“无 hold”和“hold 就在正中”，模型读成“中间被占”就一路挤死了（ar62 卡 456 音）
            live_x[i] = (x[i - 1] if i > 0 else 0.0) / HALF
            if act:
                live_x[i] = act[-1][1] / HALF
                live_on[i] = 1.0
                live_rem[i] = np.log1p(max(0.0, act[-1][0] - t[i]))
                live_sz[i] = act[-1][2]
            if dtype[i] == 2 and duro[i] > 0.5:
                act.append((t[i] + duro[i], x[i], float(ns[i].get('size', 1.0) or 1.0)))
        f[:, 11] = live_x
        f[:, 12] = live_on
        f[:, 13] = live_rem
        f[:, 14] = live_sz
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


def live_hold_x(ns):
    """每个音此刻活着的最长一条 hold 的 x；没有就 NaN"""
    t = np.array([z['t'] for z in ns], np.float32)
    x = np.array([float(z.get('x_abs', z.get('x', 0)) or 0) for z in ns], np.float32)
    ty = np.array([int(z.get('type', 1)) for z in ns], np.int64)
    du = np.array([float(z.get('dur', 0) or 0) for z in ns], np.float32)
    out = np.full(len(ns), np.nan, np.float32)
    act = []
    for i in range(len(ns)):
        act = [a for a in act if a[0] > t[i]]
        if act:
            out[i] = act[-1][1]
        if ty[i] == 2 and du[i] > 0.5:
            act.append((t[i] + du[i], x[i]))
    return out


def build_batch(songs, idx, noise=0.0):
    """songs: [(f, y, pb)]；返回 padding 后的张量与 mask"""
    n = max(len(songs[i][0]) for i in idx)
    B = len(idx)
    F = np.zeros((B, n, DIM), np.float32)
    PB = np.full((B, n), NB, np.int64)
    Y = np.zeros((B, n), np.int64)
    M = np.zeros((B, n), np.float32)
    HX = np.full((B, n), np.nan, np.float32)
    for j, i in enumerate(idx):
        f, y, pb = songs[i]
        t = len(f)
        F[j, :t] = f
        Y[j, :t] = y
        M[j, :t] = 1.0
        HX[j, :t] = live_hold_x(songs_ns[i])
        PB[j, :t] = pb
        if noise > 0:
            # 只做小幅抖动，不再把 prev 随机替换成任意格：
            # 随机替换会教模型“prev 不可信就回中间最稳”，推理时踩到 x=0 就有 39% 概率原地不动
            jitter = (np.random.rand(t) < noise)
            if jitter.any():
                PB[j, :t][jitter] = np.clip(PB[j, :t][jitter] + np.random.randint(-1, 2, int(jitter.sum())), 0, NB)
    return (torch.from_numpy(F), torch.from_numpy(PB), torch.from_numpy(Y), torch.from_numpy(M), torch.from_numpy(HX))


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
        songs_ns[len(songs) - 1] = ns
    return songs


def evaluate(model, songs, idx):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='none')
    tot_x = []; ok = 0; n = 0; acc = 0
    with torch.no_grad():
        for it in range(0, len(idx), 64):
            F, PB, Y, M, _ = build_batch(songs, idx[it:it + 64], noise=0)
            lg, _ = model(F, PB)
            if IDIAG:
                _d = (GXT[None, None, :] - GXT[PB.clamp(max=NB - 1)].unsqueeze(-1)).abs()
                _b = (_d < MINSTEP).gather(-1, Y.unsqueeze(-1)).squeeze(-1)
                _sb = (F[:, :, 7] > 0)
                if _b.sum() > 0:
                    print('DIAG 真值越界 %d/%d  avg dprev_true %.0f' % (int(_b.sum()), _b.numel(),
                          float(_d.gather(-1, Y.unsqueeze(-1)).squeeze(-1)[_b].mean())), flush=True)
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
            F, PB, Y, M, HXT = build_batch(songs, tridx[it:it + BS])
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
            if IDIAG:
                _d = (GXT[None, None, :] - GXT[PB.clamp(max=NB - 1)].unsqueeze(-1)).abs()
                _b = (_d < MINSTEP).gather(-1, Y.unsqueeze(-1)).squeeze(-1)
                _sb = (F[:, :, 7] > 0)
                if _b.sum() > 0:
                    print('DIAG 真值越界 %d/%d  avg dprev_true %.0f' % (int(_b.sum()), _b.numel(),
                          float(_d.gather(-1, Y.unsqueeze(-1)).squeeze(-1)[_b].mean())), flush=True)
            if PRIORW > 0:
                # 位置先验：真谱每格的占比当 logit 偏置，模型不至于一路推到边上
                lp = torch.from_numpy(np.log(PRIORC / PRIORC.sum() + 1e-8)).float()
                lg = lg + PRIORW * lp[None, None, :]
            banpen = 0.; sbpen = 0.
            dprev = (GXT[None, None, :] - GXT[PB.clamp(max=NB - 1)].unsqueeze(-1)).abs()
            samebeat = (F[:, :, 7] > 0)
            if MINSTEP > 0:
                short = (dprev < MINSTEP) & (~samebeat.unsqueeze(-1))
                # 必须用有限大负数：ce 带 label_smoothing=0.03，-1e9 会被摊到其余20格，
                # 均值拉到 -4.7e7，梯度直接污染，训练形同没学
                lg = lg.masked_fill(short, -30.0)
                # 真值本身落在禁区时不能算交叉熵（否则 ce 冲到 1e9，训练直接炸）
                bad = short.gather(-1, Y.unsqueeze(-1)).squeeze(-1)
                M = M * (~bad).float()
            prob = torch.softmax(lg, -1)
            if GAPHI > 0:
                # 半叠禁带：跨拍不许落在距上一音 20..135 的区间（真谱这段只占 4%）
                # 不遮 logits，直接把禁带类上的概率压到 0，权重才会真的记住这条约束
                ban = (dprev >= GAPLO) & (dprev < GAPHI) & (~samebeat.unsqueeze(-1))  # >= 而非 >：GAPLO=0 时把「原地」也关进禁带，否则不动永远是免罚的避风港
                ban[:, 0, :] = False
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
            holdpen = 0.
            if HOLDW > 0:
                hxl = HXT[:, :, None]
                hd = (GXT[None, None, :] - hxl).abs()
                hb = (hd < 177.0) & (~samebeat.unsqueeze(-1)) & torch.isfinite(hxl).expand_as(hd)
                hb[:, 0, :] = False
                holdpen = (prob * hb.float()).sum(-1)
                holdpen = (holdpen * M).sum() / M.sum()
            distpen = 0.
            if DISTW > 0 and GAPHI > 0:
                # 分布匹配：把这一批“跨拍单音”的间距档概率汇总，去对齐真谱直方图
                # 只在“确实离开上一音 >45”的音里对齐分布，0~45 那档不进统计
                dgb = (dprev / GRID).round().long().clamp(0, 20)
                q = torch.zeros(dgb.shape[0], dgb.shape[1], 21)
                q.scatter_add_(2, dgb, prob)
                keepm = ((dprev >= GRID) & (~samebeat).unsqueeze(-1)).float() * M.unsqueeze(-1)
                qm = (q[..., 1:] * keepm[..., 1:]).sum((0, 1))
                qm = qm / qm.sum().clamp(min=1e-8)
                tgt = torch.from_numpy(REALDIST).float()
                distpen = (qm - tgt).abs().sum()
            dirbon = 0.
            if DIRW > 0:
                sdx = GXT[None, None, :] - GXT[PB.clamp(max=NB - 1)].unsqueeze(-1)
                prev_dx = torch.zeros_like(sdx[:, :, 0])
                prev_dx[:, 1:] = (GXT[Y[:, :-1]] - GXT[PB[:, :-1].clamp(max=NB - 1)])
                same_dir = (torch.sign(sdx) * torch.sign(prev_dx).unsqueeze(-1)) > 0
                ok = same_dir & (~samebeat.unsqueeze(-1)) & (dprev >= MINSTEP if MINSTEP > 0 else dprev > 0)
                dirbon = (((prob * ok.float()).sum(-1)) * M).sum() / M.sum()
            farbon = 0.
            if FARW > 0:
                far = (dprev >= FARX) & (~samebeat.unsqueeze(-1))
                farbon = ((prob * far.float()).sum(-1) * M).sum() / M.sum()
            loss = (ce(lg.reshape(-1, NB), Y.reshape(-1)) * M2.reshape(-1)).sum() / M2.sum() + BANW * banpen + SBW * sbpen + HOLDW * holdpen + DISTW * distpen - FARW * farbon - DIRW * dirbon
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            if it % 200 == 0:
                print('  ep%d %d/%d loss %.4f' % (ep, it, len(tridx), tot / max(1, nb)), flush=True)
        print('ep%d 训练损失 %.4f' % (ep, tot / max(1, nb)), flush=True)
        model.eval()
        with torch.no_grad():
            vl = 0.; vn = 0; xs = []
            for it in range(0, len(vidx), 64):
                F, PB, Y, M, _ = build_batch(songs, vidx[it:it + 64], noise=0)
                lg, _ = model(F, PB)
                dprev = (GXT[None, None, :] - GXT[PB.clamp(max=NB - 1)].unsqueeze(-1)).abs()
                samebeat = (F[:, :, 7] > 0)
                if MINSTEP > 0:
                    short = (dprev < MINSTEP) & (~samebeat.unsqueeze(-1))
                    lg = lg.masked_fill(short, -30.0)
                    M = M * (~short.gather(-1, Y.unsqueeze(-1)).squeeze(-1)).float()
                if M.sum() > 0:
                    loss = (ce(lg.reshape(-1, NB), Y.reshape(-1)) * M.reshape(-1)).sum() / M.sum()
                    vl += float(loss); vn += 1
                pred = lg.argmax(-1); mask = M > 0
                if int(mask.sum()) > 0:
                    xs.append(g2x(pred[mask].numpy()))
            if not xs:
                print('  验证 全部样本被 MINSTEP 屏蔽，跳过', flush=True); continue
            x = np.concatenate(xs)
            print('  验证 音 %d |x|中位%.0f(真%.0f) <135占%.1f%%(真%.1f%%) >400占%.1f%%(真%.1f%%) loss %.4f' % (
                len(x), np.median(np.abs(x)), 253, np.mean(np.abs(x) < 135) * 100, 23.2,
                np.mean(np.abs(x) > 400) * 100, 21.9, vl / max(1, vn)), flush=True)
        torch.save(model.state_dict(), os.environ.get('CKPT_OUT', 'arnet.pt'))
        print('  已存', os.environ.get('CKPT_OUT', 'arnet.pt'), flush=True)


if __name__ == '__main__':
    main()
