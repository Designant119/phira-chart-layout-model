#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""段级位置模型 v2：一拍输出多颗（最多 4 颗），每颗各自一格。

跟 v1 的区别：
1. 标签不再是"这一拍 x 的平均"，改成这拍每颗的真实 x（按位置排序）各给一个 81 格标签
2. 输出头是 4 组 81 类，颗数不足 4 的那几组不计 loss
3. padding 真正传进 encoder 的 key padding mask，不再拿 0 向量去污染注意力
4. 输入多一维 mask 通道，第 10 维标"这拍是不是真的存在"
"""
import gzip
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

NB = 81
HALF = 675.0
MAXLEN = 64
MAXK = 4
DIM = 11


def x2g(x):
    return int(np.clip(round((x + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1))


def g2x(g):
    return -HALF + g / (NB - 1) * 2 * HALF


def featurize(r):
    """(T,10) 特征 + (T,4) 标签 + (T,4) 标签有效位"""
    chord = r['chord']; dflag = r['dflag']; iv = r['iv']; per = r['per']
    n = len(chord)
    F = np.zeros((n, DIM), dtype=np.float32)
    Y = np.zeros((n, MAXK), dtype=np.int64)
    M = np.zeros((n, MAXK), dtype=np.float32)
    for i in range(n):
        c = min(max(chord[i], 1), 4)
        F[i, c - 1] = 1.0
        F[i, 4] = 1.0 if dflag[i] in (3, 34) else 0.0
        F[i, 5] = 1.0 if dflag[i] in (4, 34) else 0.0
        ivv = iv[i] if i < len(iv) else (iv[-1] if iv else 0.2)
        F[i, 6] = min(ivv / 0.5, 2.0)
        F[i, 7] = i / max(1, n - 1)
        F[i, 8] = c / 4.0
        F[i, 9] = 1.0
        pv = r['prev'][i] if ('prev' in r and i < len(r['prev'])) else None
        F[i, 10] = 0.0 if pv is None else float(pv) / HALF
        xs = per[i][:MAXK] if i < len(per) else []
        for j, x in enumerate(xs):
            Y[i, j] = x2g(x)
            M[i, j] = 1.0
    return F, Y, M


class SegNet2(nn.Module):
    def __init__(self, d_model=96, nhead=4, layers=3, ncol=NB, maxk=MAXK):
        super().__init__()
        self.inp = nn.Linear(DIM, d_model)
        self.pos = nn.Parameter(torch.zeros(1, MAXLEN, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 192, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, maxk * ncol)

    def forward(self, x, alive=None):
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        if alive is not None:
            h = self.enc(h, src_key_padding_mask=(alive < 0.5))
        else:
            h = self.enc(h)
        out = self.head(h).view(x.shape[0], x.shape[1], MAXK, NB)
        return out


def batchify(rs):
    n = max(len(r[1]) for r in rs)
    Fb = np.zeros((len(rs), n, DIM), dtype=np.float32)
    Yb = np.zeros((len(rs), n, MAXK), dtype=np.int64)
    Mb = np.zeros((len(rs), n, MAXK), dtype=np.float32)
    for i, r in enumerate(rs):
        f, y, m = r[0], r[1], r[2]
        t = len(y)
        Fb[i, :t] = f; Yb[i, :t] = y; Mb[i, :t] = m
    return (torch.from_numpy(Fb), torch.from_numpy(Yb), torch.from_numpy(Mb))


def load_data(path, cap=0):
    data = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r['beats'] < 6 or r['beats'] > MAXLEN:
                continue
            F, Y, M = featurize(r)
            data.append((F, Y, M, r['fid']))
            if cap and len(data) >= cap:
                break
    return data


def evaluate(model, va, tag=''):
    model.eval()
    preds, gts = [], []
    seg_lens = []
    with torch.no_grad():
        for it in range(0, len(va), 256):
            rs = va[it:it + 256]
            Fb, Yb, Mb = batchify(rs)
            lg = model(Fb, Fb[:, :, 9])
            for i, r in enumerate(rs):
                t = len(r[1])
                prob = torch.softmax(lg[i, :t], dim=-1)
                samp = torch.multinomial(prob.reshape(-1, NB), 1).reshape(t, MAXK)
                for bi in range(t):
                    kk = int(Mb[i, bi].sum())
                    if kk == 0:
                        continue
                    for j in range(kk):
                        preds.append(float(g2x(int(samp[bi, j]))))
                        gts.append(float(g2x(int(Yb[i, bi, j]))))
                seg_lens.append(t)
    p = np.array(preds); g = np.array(gts)
    print('%s 颗数 %d  预测|x|中位%.0f(真%.0f)  <135占%.1f%%(真%.1f%%)  >400占%.1f%%(真%.1f%%)' % (
        tag, len(p), np.median(np.abs(p)), np.median(np.abs(g)),
        100 * (np.abs(p) < 135).mean(), 100 * (np.abs(g) < 135).mean(),
        100 * (np.abs(p) > 400).mean(), 100 * (np.abs(g) > 400).mean()))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'seg_index_abs.jsonl.gz'
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rows = load_data(path, cap)
    print('可用段 %d' % len(rows), flush=True)
    fids = sorted(set(r[3] for r in rows))
    val_fid = set(random.sample(fids, max(1, len(fids) // 20)))
    tr = [r for r in rows if r[3] not in val_fid]
    va = [r for r in rows if r[3] in val_fid]
    print('训练 %d 段 / 验证 %d 段（按曲目切分）' % (len(tr), len(va)))

    model = SegNet2()
    _ck = os.environ.get('CKPT_OUT', 'segnet2.pt')
    if os.environ.get('RESUME') and os.path.exists(_ck):
        model.load_state_dict(torch.load(_ck, map_location='cpu'))
        print('  从 *_ck 热启动', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)
    BS = 128
    EPOCH = int(os.environ.get('EPOCH', 8))
    for ep in range(EPOCH):
        model.train()
        random.shuffle(tr)
        tot = 0.0; nb = 0
        for it in range(0, len(tr), BS):
            rs = tr[it:it + BS]
            Fb, Yb, Mb = batchify(rs)
            # prev 维必须加噪声：训练时它是真值，模型会走"直接复制上一拍"的捷径，
            # 推理时喂的是自己上一拍预测，误差一滚就整段塌成一列（实测同列率冲到 62%）
            pv = Fb[:, :, 10]
            keep = (torch.rand_like(pv) > 0.3).float()
            Fb[:, :, 10] = (pv + torch.randn_like(pv) * 0.04) * keep
            lg = model(Fb, Fb[:, :, 9])
            loss = (ce(lg.reshape(-1, NB), Yb.reshape(-1)) * Mb.reshape(-1)).sum() / Mb.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            if it % 4000 == 0:
                print('  ep%d %d/%d loss %.4f' % (ep, it, len(tr), tot / max(1, nb)), flush=True)
        print('ep%d 训练损失 %.4f' % (ep, tot / max(1, nb)), flush=True)
        evaluate(model, va, '  验证')
        torch.save(model.state_dict(), os.environ.get('CKPT_OUT', 'segnet2.pt'))
        print('  已存 segnet2.pt', flush=True)


if __name__ == '__main__':
    main()
