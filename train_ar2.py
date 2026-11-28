#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐音自回归排键模型 v2（train_ar.py 的换代版，旧文件不动）。

和 train_ar.py 相比改了三件事，其余口径不变（2 层 GRU h=192、11 维节奏特征、文件顺序、95/5 切分）：
1. 27 格 × 50 铺满 ±675（BINS = linspace(-675, 675, 28)），边界音不再被 clip 到 ±450。
2. 显式段状态特征，全部因果：run 方向/连续拍数、连续原地数、最近 32/8 音的 27 格访问频率、
   长押占位，外加曲级 style embedding（nn.Embedding(64, 16)）。
   训练（含 scheduled sampling）和推理共用同一个 SegState 在线递推，只能看到第 t 颗音之前的位置。
3. 整曲为单位做 BPTT 分块，块间 hidden state 只 detach 不 reset；最后 20% 的 epoch 做
   scheduled sampling：每曲前缀 10~30% 让模型逐步吃自己采样的位置。

用法：
  python3 train_ar2.py --limit 1500            # 训练，权重存 arnet2_<tag>.pt
  EPOCH=0 python3 train_ar2.py --limit 200     # 只跑自检（熵基线 + 初始 CE + 因果性）
环境变量：EPOCH(5) BS(16) CHUNK(256) LR(2e-3) LS(0.03) CLIP(1.0) TAG CKPT_OUT RESUME SNAP
          SS_FRAC(0.2)   塑形罚沿用旧名：BANW(2.0) GAPLO(20) GAPHI(135) SBAN(180) SBW(3.0)
推理：m = load_ckpt(ARNet2(), 'arnet2_xx.pt'); bins, sid = free_run(m, notes, sid=None)
"""
import os
import glob
import time
import random
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 坐标 ----------
NB = 27            # 27 格 × 50 = 1350，正好铺满 ±675
HALF = 675.0
GRID = 50.0
BINS = np.linspace(-HALF, HALF, NB + 1)   # 格 g = [BINS[g], BINS[g+1])，中心 g2x(g) = -650 + 50g


def x2g(x):
    # 不能照搬旧版的 round((x + HALF) / GRID)：那要求 HALF/GRID 是整数（450/45=10 成立，675/50=13.5 不成立），
    # 否则格边界会压在 x=0 上（中列被劈成 (-50,0] / (0,50] 两格），最右格中心只到 625。这里按 BINS 边界取格。
    return int(np.clip(np.floor((x + HALF) / GRID), 0, NB - 1))


def g2x(g):
    return BINS[g] + GRID / 2


# ---------- 特征布局（FEATVER 存进权重，推理端据此核对）----------
FEATVER = 2
NRHY = 11                                   # 节奏特征：照搬 train_ar.feats 的前 11 维
IV32 = 3                                    # 最近 32 音访问频率 (NB 维) 起点
IV8 = IV32 + NB                             # 最近 8 音访问频率 (NB 维) 起点
HOLD0 = IV8 + NB                            # 长押占位 4 维起点
KSTATE = HOLD0 + 4                          # 段状态 61 维
DIM = NRHY + KSTATE                         # 72
PEB = 48          # prev 位置 embedding 维
NSTY = 64         # 曲级 style 个数
SDIM = 16         # style embedding 维

# ---------- 训练配置 ----------
EPOCH = int(os.environ.get('EPOCH', '5'))
BS = int(os.environ.get('BS', '16'))
CHUNK = int(os.environ.get('CHUNK', '256'))          # BPTT 块长（音数）
LR = float(os.environ.get('LR', '2e-3'))
LS = float(os.environ.get('LS', '0.03'))             # label smoothing（沿用旧版）
CLIP = float(os.environ.get('CLIP', '1.0'))
MAXLEN = int(os.environ.get('MAXLEN', '4096'))       # 单曲最长音数，超过的不练
SS_FRAC = float(os.environ.get('SS_FRAC', '0.2'))    # 最后这么多比例的 epoch 做 scheduled sampling
SS_LO, SS_HI = 0.1, 0.3                              # scheduled sampling 的前缀长度占整曲比例
# 塑形罚：沿用 train_ar 默认开着的那两条（跨拍半叠禁带、同拍最小间距），单位是坐标所以与格子数无关
BANW = float(os.environ.get('BANW', '2.0'))
GAPLO = float(os.environ.get('GAPLO', '20.0'))
GAPHI = float(os.environ.get('GAPHI', '135.0'))
SBAN = float(os.environ.get('SBAN', '180'))
SBW = float(os.environ.get('SBW', '3.0'))
TAG = os.environ.get('TAG') or time.strftime('%m%d_%H%M')
CKPT_OUT = os.environ.get('CKPT_OUT') or 'arnet2_%s.pt' % TAG

GXT = torch.tensor([g2x(i) for i in range(NB)], dtype=torch.float32)


# ---------- 数据 ----------
def load_song(path):
    d = json.load(open(path))
    ns = [n for n in d['notes'] if n.get('x_abs') is not None]
    ns.sort(key=lambda n: n['t'])
    return ns


def rhythm_feats(ns):
    """(n, 11) 节奏特征，原样照搬 train_ar.feats 的前 11 维。第 1/8 维看的是下一颗音的时间：
    节奏是推理时的给定输入、不是模型生成的量，所以不算泄漏。"""
    n = len(ns)
    t = np.array([z['t'] for z in ns], np.float32)
    ty = np.array([int(z.get('type', 1)) for z in ns], np.int64)
    du = np.array([float(z.get('dur', 0) or 0) for z in ns], np.float32)
    prev = np.diff(t, prepend=t[0] - 2.)
    nxt = np.diff(t, append=t[-1] + 2.)
    f = np.zeros((n, NRHY), np.float32)
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
    return f


def song_meta(ns):
    """tracker 推进时要用的、只跟节奏有关的量（全部是第 t 颗音自己的属性，不看后面）"""
    n = len(ns)
    t = np.array([z['t'] for z in ns], np.float32)
    ty = np.array([int(z.get('type', 1)) for z in ns], np.int64)
    du = np.array([float(z.get('dur', 0) or 0) for z in ns], np.float32)
    prev = np.diff(t, prepend=t[0] - 2.)
    return {
        't': t,
        'nb': np.r_[False, prev[1:] >= .005],        # 是否开了新的一拍（同拍第二颗起 False）
        'hold': (ty == 2) & (du > 0.5),
        'end': t + du,
        'size': np.array([float(z.get('size', 1.0) or 1.0) for z in ns], np.float32),
    }


class SegState:
    """「下一颗音」之前的段状态。feat(tc, out) 只读；push(...) 吃进一颗位置已定的音。
    训练（含 scheduled sampling）和推理共用这一个类：能 push 进去的只可能是已经喂给模型的音，天然不泄漏。"""

    def __init__(self):
        self.yp = None                      # 上一音格位（还没有音时 None）
        self.dir = 0                        # 当前 run 方向 -1/0/+1
        self.run = 0                        # 当前 run 已连续同向走了几拍
        self.stand = 0                      # 上一音所在格已连续落了几颗音
        self.h32 = deque(maxlen=32); self.c32 = np.zeros(NB, np.float32)
        self.h8 = deque(maxlen=8); self.c8 = np.zeros(NB, np.float32)
        self.act = []                       # 还活着的长押 [(结束时间, x, size)]

    def feat(self, tc, out):
        act = [a for a in self.act if a[0] > tc]
        self.act = act
        out[0] = self.dir
        out[1] = np.log1p(min(self.run, 16))
        out[2] = np.log1p(min(self.stand, 8))
        np.log1p(self.c32, out=out[IV32:IV32 + NB])
        np.log1p(self.c8, out=out[IV8:IV8 + NB])
        if act:
            out[HOLD0] = act[-1][1] / HALF
            out[HOLD0 + 1] = 1.0
            out[HOLD0 + 2] = np.log1p(max(0.0, act[-1][0] - tc))
            out[HOLD0 + 3] = act[-1][2]
        else:
            # 照 train_ar：没有活 hold 时填上一音位置而不是 0（0 会被读成“正中被占”）
            out[HOLD0] = (g2x(self.yp) if self.yp is not None else 0.0) / HALF
            out[HOLD0 + 1:HOLD0 + 4] = 0.0

    def push(self, y, tc, nb, hold, end, size):
        if self.yp is not None:
            if nb:                          # 跨拍的一步才算“走了一拍”，同拍多押不改变 run
                d = (y > self.yp) - (y < self.yp)
                if d == self.dir:
                    self.run += 1
                else:
                    self.dir = d; self.run = 1
            self.stand = self.stand + 1 if y == self.yp else 1
        else:
            self.stand = 1
        if len(self.h32) == 32:
            self.c32[self.h32[0]] -= 1.0
        self.h32.append(y); self.c32[y] += 1.0
        if len(self.h8) == 8:
            self.c8[self.h8[0]] -= 1.0
        self.h8.append(y); self.c8[y] += 1.0
        if hold:
            self.act.append((float(end), g2x(y), float(size)))
        self.yp = y


def state_feats(y, meta):
    """按一段已知的位置序列整段跑 tracker → (n, KSTATE)。"""
    out = np.zeros((len(y), KSTATE), np.float32)
    st = SegState()
    for i in range(len(y)):
        st.feat(meta['t'][i], out[i])
        st.push(int(y[i]), meta['t'][i], bool(meta['nb'][i]), bool(meta['hold'][i]), float(meta['end'][i]), float(meta['size'][i]))
    return out


def feats(ns):
    """(n, DIM)：11 维节奏特征 + 61 维段状态（按谱面真实 x 算的 teacher-forcing 版）。
    自回归推理时 x 是逐颗生成的，不能调这个函数，走 free_run()。"""
    x = np.array([float(z.get('x_abs', z.get('x', 0)) or 0) for z in ns], np.float32)
    return np.concatenate([rhythm_feats(ns), state_feats(np.array([x2g(v) for v in x], np.int64), song_meta(ns))], 1)


def make_song(ns, sid):
    x = np.array([float(z.get('x_abs', z.get('x', 0)) or 0) for z in ns], np.float32)
    y = np.array([x2g(v) for v in x], np.int64)
    return {'x': x, 'y': y, 'rhy': rhythm_feats(ns), 'meta': song_meta(ns), 'sid': int(sid)}


def load_data(paths, rng):
    songs = []
    for p in paths:
        try:
            ns = load_song(p)
        except Exception:
            continue
        if len(ns) < 8 or len(ns) > MAXLEN:
            continue
        s = make_song(ns, rng.integers(0, NSTY))     # 每曲随机分配一个 style id，固定种子所以可复现
        s['path'] = p
        songs.append(s)
    return songs


class Batch:
    """一小批整曲，padding 到批内最长。trk[b] 是第 b 首歌的 SegState，整曲跑完才作废。"""

    def __init__(self, songs, idx):
        self.ss = [songs[i] for i in idx]
        self.B = len(idx)
        self.lens = [len(s['y']) for s in self.ss]
        self.n = max(self.lens)
        B, n = self.B, self.n
        self.R = torch.zeros(B, n, NRHY)
        self.PB = torch.full((B, n), NB, dtype=torch.long)   # 首音填 NB 当 BOS
        self.Y = torch.zeros(B, n, dtype=torch.long)
        self.M = torch.zeros(B, n)
        for j, s in enumerate(self.ss):
            L = self.lens[j]
            self.R[j, :L] = torch.from_numpy(s['rhy'])
            self.PB[j, 1:L] = torch.from_numpy(s['y'][:-1])
            self.Y[j, :L] = torch.from_numpy(s['y'])
            self.M[j, :L] = 1.0
        self.sid = torch.tensor([s['sid'] for s in self.ss], dtype=torch.long)
        self.pbn = self.PB.numpy()
        self.yn = self.Y.numpy()
        self.trk = [SegState() for _ in range(B)]

    def bulk_state(self, c0, c1):
        """teacher-forcing 段特征：[c0,c1) 各音按真值序列推进 tracker → (B, c1-c0, KSTATE)"""
        S = np.zeros((self.B, c1 - c0, KSTATE), np.float32)
        for j, s in enumerate(self.ss):
            m = s['meta']; tr = self.trk[j]
            for t in range(c0, min(c1, self.lens[j])):
                tr.feat(m['t'][t], S[j, t - c0])
                tr.push(int(s['y'][t]), m['t'][t], m['nb'][t], m['hold'][t], m['end'][t], m['size'][t])
        return torch.from_numpy(S)


# ---------- 模型 ----------
class ARNet2(nn.Module):
    """结构同 train_ar.ARNet，输入多一路曲级 style 向量（整曲恒定，拼在每一步的输入上）。"""

    def __init__(self, h=192):
        super().__init__()
        self.prev = nn.Embedding(NB + 1, PEB)
        self.style = nn.Embedding(NSTY, SDIM)
        self.inp = nn.Linear(DIM + PEB + SDIM, h)
        self.g = nn.GRU(h, h, 2, batch_first=True, dropout=.2)
        self.o = nn.Linear(h, NB)

    def forward(self, f, pb, st=None, sid=None):
        """f (B,T,DIM)，pb (B,T) 上一音格位（首音 NB 当 BOS），st 为 GRU hidden state，sid (B,) 为 style id。
        sid=None 时 style 向量取 0。返回 (logits (B,T,NB), st)。"""
        sty = self.style(sid)[:, None, :] if sid is not None else f.new_zeros(f.shape[0], 1, SDIM)
        x = torch.cat([f, self.prev(pb), sty.expand(-1, f.shape[1], -1)], -1)
        o, st = self.g(torch.relu(self.inp(x)), st)
        return self.o(o), st


def save_ckpt(model, path):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    sd['NB'] = NB; sd['FEATVER'] = FEATVER       # meta：推理端据此识别格子数和特征布局
    torch.save(sd, path)


def load_ckpt(model, path):
    sd = torch.load(path, map_location='cpu')
    own = model.state_dict()
    if sd.get('NB') != NB or sd.get('FEATVER') != FEATVER or set(sd) - set(own) != {'NB', 'FEATVER'} or set(own) - set(sd):
        raise SystemExit('权重 %s 与当前代码不匹配：NB=%s FEATVER=%s（代码是 %d / %d）' % (
            path, sd.get('NB'), sd.get('FEATVER'), NB, FEATVER))
    model.load_state_dict({k: v for k, v in sd.items() if k in own})
    model.eval()
    return model


# ---------- loss ----------
def pos_loss(logits, pb, sb, Y, ls, shape):
    """逐音交叉熵 (B,T) + 逐音塑形罚 (B,T)。"""
    nll = F.cross_entropy(logits.transpose(1, 2), Y, reduction='none', label_smoothing=ls)
    if not shape or (BANW <= 0 and SBW <= 0):
        return nll, torch.zeros_like(nll)
    pen = torch.zeros_like(nll)
    dprev = (GXT[None, None, :] - GXT[pb.clamp(max=NB - 1)][:, :, None]).abs()
    ok = (pb != NB)[:, :, None]                 # BOS 位没有“上一音”，不罚
    sb = sb.reshape(sb.shape + (1,))   # (B,) 或 (B,T) 都补一维，对齐 (B,T,NB)
    prob = torch.softmax(logits, -1)
    if BANW > 0:
        # 跨拍不许落在距上一音 20..135 的半叠区间
        pen = pen + BANW * (prob * ((dprev >= GAPLO) & (dprev < GAPHI) & ~sb & ok)).sum(-1)
    if SBW > 0:
        # 同拍第二个音起，不许落在前一颗 180 以内
        pen = pen + SBW * (prob * ((dprev < SBAN) & sb & ok)).sum(-1)
    return nll, pen


def step_chunk(model, bt, t0, t1, st_h, cut, ls=0.0, shape=False, gen=None, temp=1.0):
    """[t0,t1) 逐步前向。cut[b] > t 的歌在 t 这步吃「自己上一步采样的位置」，其余吃真值 prev；
    段状态一律由 tracker 在线算（里面只有 t 之前的音）。cut 全取很大就是纯自回归生成。
    返回 (nll, pen, st_h, fed)；fed (B,t1-t0) 是每步实际吃进去的历史位置。"""
    B = bt.B
    ys = [NB] * B
    fed = torch.full((B, t1 - t0), NB, dtype=torch.long)
    L = []; P = []
    for t in range(t0, t1):
        S = np.zeros((B, KSTATE), np.float32)
        for j in range(B):
            if t < bt.lens[j]:
                bt.trk[j].feat(bt.ss[j]['meta']['t'][t], S[j])
        pb = torch.tensor([ys[j] if t < cut[j] else bt.pbn[j, t] for j in range(B)], dtype=torch.long)
        logits, st_h = model(torch.cat([bt.R[:, t], torch.from_numpy(S)], -1).unsqueeze(1), pb[:, None], st_h, bt.sid)
        l, p = pos_loss(logits, pb[:, None], bt.R[:, t, 7] > 0, bt.Y[:, t, None], ls, shape)
        L.append(l[:, 0]); P.append(p[:, 0])
        with torch.no_grad():
            samp = torch.multinomial(torch.softmax(logits[:, 0] / temp, -1), 1, generator=gen)[:, 0].tolist()
        for j in range(B):
            if t >= bt.lens[j]:
                continue
            m = bt.ss[j]['meta']
            y = int(samp[j]) if t < cut[j] else int(bt.yn[j, t])
            bt.trk[j].push(y, m['t'][t], m['nb'][t], m['hold'][t], m['end'][t], m['size'][t])
            ys[j] = y; fed[j, t - t0] = y
    return torch.stack(L, 1), torch.stack(P, 1), st_h, fed


def make_batches(songs, idx, bs, rng):
    """整曲成批：先打乱，再在池子（bs×8 首）内按长度排序切批，最后打乱批顺序 → padding 少，批内容仍随机"""
    idx = list(rng.permutation(idx))
    out = []
    for p in range(0, len(idx), bs * 8):
        pool = sorted(idx[p:p + bs * 8], key=lambda i: len(songs[i]['y']))
        out += [pool[q:q + bs] for q in range(0, len(pool), bs)]
    return [out[i] for i in rng.permutation(len(out))]


def train_batch(model, opt, bt, cut):
    """一首歌从头到尾算一个样本：按 CHUNK 分块，每块 backward + step，hidden state detach 后交给下一块。"""
    st = None
    tot = 0.0; ntok = 0.0
    c0 = 0
    while c0 < bt.n:
        c1 = min(bt.n, c0 + CHUNK)
        k = max(0, min(c1, max(cut)) - c0)        # 这块里还有歌在自采样的步数，这些只能逐步走
        L = []; P = []
        if k > 0:
            nll, pen, st, _ = step_chunk(model, bt, c0, c0 + k, st, cut, LS, True)
            L.append(nll); P.append(pen)
        if c1 - c0 > k:
            a = c0 + k
            logits, st = model(torch.cat([bt.R[:, a:c1], bt.bulk_state(a, c1)], -1), bt.PB[:, a:c1], st, bt.sid)
            nll, pen = pos_loss(logits, bt.PB[:, a:c1], bt.R[:, a:c1, 7] > 0, bt.Y[:, a:c1], LS, True)
            L.append(nll); P.append(pen)
        M = bt.M[:, c0:c1]
        if float(M.sum()) > 0:
            loss = ((torch.cat(L, 1) + torch.cat(P, 1)) * M).sum() / M.sum()
            opt.zero_grad(); loss.backward()
            if CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            tot += float(loss.detach()) * float(M.sum()); ntok += float(M.sum())
        st = st.detach()                           # 只 detach 不 reset：下一块接着这首歌的状态往下跑
        c0 = c1
    return tot / max(ntok, 1.0)


@torch.no_grad()
def evaluate(model, songs, idx, bs=32):
    """teacher-forcing 验证：整曲分块、块间 carry state（eval 模式下等价于整曲一次前向）。"""
    model.eval()
    order = sorted(idx, key=lambda i: len(songs[i]['y']))
    n = 0.0; s1 = 0.0; s2 = 0.0; pm = []; tm = []
    for it in range(0, len(order), bs):
        bt = Batch(songs, order[it:it + bs])
        st = None
        for c0 in range(0, bt.n, CHUNK):
            c1 = min(bt.n, c0 + CHUNK)
            logits, st = model(torch.cat([bt.R[:, c0:c1], bt.bulk_state(c0, c1)], -1), bt.PB[:, c0:c1], st, bt.sid)
            lg = logits.transpose(1, 2)
            M = bt.M[:, c0:c1]
            s1 += float((F.cross_entropy(lg, bt.Y[:, c0:c1], reduction='none') * M).sum())
            s2 += float((F.cross_entropy(lg, bt.Y[:, c0:c1], reduction='none', label_smoothing=LS) * M).sum())
            n += float(M.sum())
            m = M > 0
            pm.append(GXT[logits.argmax(-1)[m]].numpy()); tm.append(GXT[bt.Y[:, c0:c1][m]].numpy())
    p = np.concatenate(pm); q = np.concatenate(tm)
    return {'n': int(n), 'ce': s1 / max(n, 1.0), 'ce_ls': s2 / max(n, 1.0),
            'pmed': np.median(np.abs(p)), 'p135': np.mean(np.abs(p) < 135) * 100, 'p400': np.mean(np.abs(p) > 400) * 100,
            'tmed': np.median(np.abs(q)), 't135': np.mean(np.abs(q) < 135) * 100, 't400': np.mean(np.abs(q) > 400) * 100}


@torch.no_grad()
def freerun_ce(model, songs, idx, seed=0):
    """自回归 CE：让模型逐音吃自己的采样走完整曲，再量真值的 CE。这是 scheduled sampling 想改善的那个数。"""
    model.eval()
    bt = Batch(songs, list(idx))
    nll, _, _, _ = step_chunk(model, bt, 0, bt.n, None, [1 << 30] * bt.B, gen=torch.Generator().manual_seed(seed))
    return float((nll * bt.M).sum() / bt.M.sum())


@torch.no_grad()
def free_run(model, notes, sid=None, temp=1.0, seed=0):
    """推理入口。notes 是整曲节奏（只用 t/type/dur/size，不看 x），逐音采样的同时在线推进段状态。
    sid 指定曲级 style id；不给就随机取一个。返回 (每颗音的格位 list, 用到的 sid)。"""
    model.eval()
    if sid is None:
        sid = random.randrange(NSTY)
    bt = Batch([make_song(sorted(notes, key=lambda z: z['t']), sid)], [0])
    _, _, _, fed = step_chunk(model, bt, 0, bt.n, None, [1 << 30], gen=torch.Generator().manual_seed(seed), temp=temp)
    return [int(v) for v in fed[0, :bt.lens[0]]], int(sid)


# ---------- 自检 ----------
def selfcheck(songs, model):
    sub = songs[:200]
    xl = [s['x'] for s in sub]
    print('自检 %d 首 %d 音' % (len(sub), sum(len(x) for x in xl)), flush=True)
    # unigram / bigram 条件熵（nat）。新旧两套格子各算一遍：旧的那行用来对齐 21 格的基线
    old = [np.clip(np.round((x + 450) / 45), 0, 20).astype(np.int64) for x in xl]
    new = [s['y'] for s in sub]
    for name, yl, nb_ in (('21 格（旧口径，对齐用）', old, 21), ('27 格（本文件）', new, NB)):
        uni = np.bincount(np.concatenate(yl), minlength=nb_)
        a = np.concatenate([y[:-1] for y in yl]); b = np.concatenate([y[1:] for y in yl])
        big = np.bincount(a * nb_ + b, minlength=nb_ * nb_).reshape(nb_, nb_)
        p = big / np.maximum(big.sum(1, keepdims=True), 1)
        pu = uni / uni.sum()
        print('  %s  unigram %.3f  bigram %.3f' % (name, float(-(pu * np.log(pu)).sum()),
                                                   float((big * -np.log(np.maximum(p, 1e-12))).sum() / big.sum())), flush=True)
    x = np.concatenate(xl); inb = np.abs(x) <= HALF
    err = np.abs(g2x(new and np.concatenate(new)) [inb] - x[inb])
    print('  27 格量化误差 MAE %.1f 最大 %.1f；|x|>675 占 %.2f%%（clip 进两端格）' % (
        err.mean(), err.max(), np.mean(~inb) * 100), flush=True)
    # 因果性：feats(ns[:t+1])[t] 必须等于 feats(ns)[t]。第 1/8 维看下一颗音的时间（给定节奏），排除在外
    ns = load_song(sub[0]['path'])[:300]
    full = feats(ns)
    # 1/8 维是「与下一音是否同拍」，10 维是「前后 1 秒内音数」（密度），
    # 这三条都只依赖给定的节奏时间轴，推理时完全已知，不属于要隐藏的未来信息；
    # 真正要核的是段状态维（run/stand/访问频率/hold），它们只能由已 push 的音决定
    keep = [i for i in range(DIM) if i not in (1, 8, 10)]
    bad = max(float(np.abs(feats(ns[:t + 1])[t][keep] - full[t][keep]).max()) for t in range(len(ns)))
    print('  因果性 %d 音逐前缀比对：%s' % (len(ns), '通过' if bad == 0 else '不通过，最大偏差 %g' % bad), flush=True)
    if bad != 0:
        raise SystemExit('段状态特征泄漏：截断前缀后特征变了')
    r = evaluate(model, songs, list(range(len(sub))))
    print('  初始模型 CE %.4f（均匀分布 = log%d = %.4f）' % (r['ce'], NB, np.log(NB)), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pars', default='/home/user/Doubao/chats/38453392661469954/_run_real_traj/abs_full/parsed')
    ap.add_argument('--limit', type=int, default=1500)
    a = ap.parse_args()
    files = sorted(glob.glob(a.pars + '/*.json'))
    random.seed(0); random.shuffle(files)            # 与 train_ar 同一顺序 → 同一份 95/5 切分
    files = files[:a.limit] if a.limit > 0 else files
    print('曲目', len(files), flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    songs = load_data(files, rng)
    print('可用', len(songs), '总音', sum(len(s['y']) for s in songs), flush=True)
    model = ARNet2()
    selfcheck(songs, model)
    n = len(songs)
    vidx = list(range(int(n * 0.95), n))
    tridx = list(range(0, int(n * 0.95)))
    print('格子 %d×%g（%g..%g） DIM=%d FEATVER=%d  EPOCH=%d BS=%d CHUNK=%d LR=%g' % (
        NB, GRID, g2x(0), g2x(NB - 1), DIM, FEATVER, EPOCH, BS, CHUNK, LR), flush=True)
    if EPOCH <= 0:
        return
    if os.environ.get('RESUME') and os.path.exists(CKPT_OUT):
        load_ckpt(model, CKPT_OUT)
        print('热启动', CKPT_OUT, flush=True)
    ss_start = EPOCH - max(1, int(round(EPOCH * SS_FRAC))) if SS_FRAC > 0 else EPOCH
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    fr = vidx[::max(1, len(vidx) // 8)][:8]          # 自回归 CE 只抽 8 首，逐音跑太慢
    for ep in range(EPOCH):
        model.train()
        t0 = time.time()
        tot = 0.0; nb_ = 0
        for bi, bidx in enumerate(make_batches(songs, tridx, BS, rng)):
            bt = Batch(songs, bidx)
            # scheduled sampling：每曲前缀 10~30%（整曲为单位随机抽）吃模型自己的采样
            cut = [int(rng.uniform(SS_LO, SS_HI) * L) for L in bt.lens] if ep >= ss_start else [0] * bt.B
            tot += train_batch(model, opt, bt, cut); nb_ += 1
            if bi % 10 == 0:
                print('  ep%d %d/%d loss %.4f%s' % (ep, bi, len(tridx) // BS + 1, tot / nb_, '  [SS]' if ep >= ss_start else ''), flush=True)
        print('ep%d 训练损失 %.4f  %.0fs' % (ep, tot / max(nb_, 1), time.time() - t0), flush=True)
        r = evaluate(model, songs, vidx)
        print('  验证 音 %d  CE %.4f（带 LS %.4f，旧口径）  |x|中位 %.0f(真 %.0f)  <135 %.1f%%(真 %.1f%%)  >400 %.1f%%(真 %.1f%%)' % (
            r['n'], r['ce'], r['ce_ls'], r['pmed'], r['tmed'], r['p135'], r['t135'], r['p400'], r['t400']), flush=True)
        print('  自回归 %d 首 CE %.4f' % (len(fr), freerun_ce(model, songs, fr)), flush=True)
        save_ckpt(model, CKPT_OUT)
        if os.environ.get('SNAP'):
            save_ckpt(model, CKPT_OUT[:-3] + '_ep%d.pt' % ep)
        print('  已存', CKPT_OUT, flush=True)


if __name__ == '__main__':
    import json
    main()
