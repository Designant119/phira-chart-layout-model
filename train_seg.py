#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""段级位置模型：输入一段的节奏指纹，输出每拍落在哪一格。

题目设计要点：
- 输出 81 格分类，不是回归坐标（回归+MSE 的最优解永远是平均值，必塌中间）
- 推理按概率采样，不取 argmax，保住分布宽度
- 只学"单点走位"，多押展开交给规则
"""
import gzip
import json
import math
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

NB = 81
HALF = 675.0
MAXLEN = 64

torch.manual_seed(7)
np.random.seed(7)
random.seed(7)


def x2g(x):
    return int(np.clip(round((x + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1))


def g2x(g):
    return -HALF + g / (NB - 1) * 2 * HALF


def load_data(path, cap=0):
    """边读边特征化，原始记录用完就丢，控制内存"""
    data = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r['zero'] > 0.3:      # 多判定线的 x=0 假象，别学
                continue
            if r['beats'] < 6 or r['beats'] > MAXLEN:
                continue
            F, y = featurize(r)
            data.append((F, y, r['fid']))
            if cap and len(data) >= cap:
                break
    return data


def featurize(r):
    """每拍一个 9 维向量 + 每拍的位置标签"""
    chord = r['chord']; dflag = r['dflag']; iv = r['iv']; ans = r['ans']
    n = len(chord)
    pos = []
    k = 0
    for c in chord:
        seg = ans[k:k + c] if c > 0 else [0.0]
        pos.append(float(np.mean(seg)) if seg else 0.0)
        k += c
    F = np.zeros((n, 9), dtype=np.float32)
    for i in range(n):
        c = min(chord[i], 4)
        F[i, c - 1] = 1.0
        F[i, 4] = 1.0 if dflag[i] in (3, 34) else 0.0
        F[i, 5] = 1.0 if dflag[i] in (4, 34) else 0.0
        ivv = iv[i] if i < len(iv) else (iv[-1] if iv else 0.2)
        F[i, 6] = min(ivv / 0.5, 2.0)
        F[i, 7] = i / max(1, n - 1)
        F[i, 8] = c / 4.0
    y = np.array([x2g(p) for p in pos], dtype=np.int64)
    return F, y


class SegNet(nn.Module):
    def __init__(self, d_model=72, nhead=4, layers=2, ncol=NB):
        super().__init__()
        self.inp = nn.Linear(9, d_model)
        self.pos = nn.Parameter(torch.zeros(1, MAXLEN, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 128, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, ncol)

    def forward(self, x):
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        mask = x.sum(-1) < -1e8
        h = self.enc(h, src_key_padding_mask=None)
        return self.head(h)


def batchify(rs):
    F = [d[0] for d in rs]
    Y = [d[1] for d in rs]
    M = [np.ones(len(d[1]), dtype=np.float32) for d in rs]
    n = max(len(y) for y in Y)
    Fb = np.zeros((len(rs), n, 9), dtype=np.float32)
    Yb = np.zeros((len(rs), n), dtype=np.int64)
    Mb = np.zeros((len(rs), n), dtype=np.float32)
    for i, (f, y, m) in enumerate(zip(F, Y, M)):
        Fb[i, :len(y)] = f; Yb[i, :len(y)] = y; Mb[i, :len(y)] = m
    return (torch.from_numpy(Fb), torch.from_numpy(Yb), torch.from_numpy(Mb))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/user/Doubao/chats/38453392661469954/_run_real_traj/seg_index.jsonl.gz'
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    rows = load_data(path, cap)
    print('可用段 %d' % len(rows), flush=True)
    fids = sorted(set(r[2] for r in rows))
    val_fid = set(random.sample(fids, max(1, len(fids) // 20)))
    tr = [r for r in rows if r[2] not in val_fid]
    va = [r for r in rows if r[2] in val_fid]
    print('训练 %d 段 / 验证 %d 段（按曲目切分，不泄漏）' % (len(tr), len(va)))

    model = SegNet()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)

    BS = 128
    EPOCH = int(os.environ.get('EPOCH', 8))
    for ep in range(EPOCH):
        model.train()
        random.shuffle(tr)
        tot = 0.0
        for it in range(0, len(tr), BS):
            rs = tr[it:it + BS]
            Fb, Yb, Mb = batchify(rs)
            logits = model(Fb)
            loss = (ce(logits.reshape(-1, NB), Yb.reshape(-1)) * Mb.reshape(-1)).sum() / Mb.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
            if it % 2000 == 0:
                print('  ep%d %d/%d loss %.3f' % (ep, it, len(tr), tot / max(1, it / BS + 1)), flush=True)
        print('ep%d 训练损失 %.3f' % (ep, tot / max(1, len(tr) / BS)))
        evaluate(model, va)
        torch.save(model.state_dict(), 'segnet.pt')
        print('  已存 segnet.pt', flush=True)


@torch.no_grad()
def evaluate(model, va):
    """不看准确率，看手感：分布宽度/换向率/步长/中间带"""
    model.eval()
    allpred, allgt = [], []
    for it in range(0, len(va), 256):
        rs = va[it:it + 256]
        Fb, Yb, Mb = batchify(rs)
        lg = model(Fb)
        for i, r in enumerate(rs):
            n = len(r[1])
            p = lg[i, :n]
            prob = torch.softmax(p, dim=-1)
            # 采样而不是 argmax
            samp = torch.multinomial(prob, 1).squeeze(-1)
            allpred.append(samp.numpy())
            allgt.append(Yb[i, :n].numpy())
    ps = np.concatenate(allpred)
    gs = np.concatenate(allgt)
    xs = np.array([g2x(g) for g in ps])
    gx = np.array([g2x(g) for g in gs])
    d = np.diff(xs); d = d[np.abs(d) > 1e-6]
    alt = float((np.sign(d[:-1]) * np.sign(d[1:]) < 0).mean()) if len(d) >= 2 else 0.5
    print('  验证 预测|x|中位%.0f(真%.0f) 中间带135内%.1f%%(真%.1f%%) 步长中位%.0f(真%.0f) 换向%.2f(真%.2f)' % (
        np.median(np.abs(xs)), np.median(np.abs(gx)),
        100 * (np.abs(xs) < 135).mean(), 100 * (np.abs(gx) < 135).mean(),
        np.median(np.abs(d)) if len(d) else 0,
        np.median(np.abs(np.diff(gx))) if len(gx) > 1 else 0,
        alt, float((np.sign(np.diff(gx)[:-1]) * np.sign(np.diff(gx)[1:]) < 0).mean()) if len(gx) > 3 else 0.5))


if __name__ == '__main__':
    main()
