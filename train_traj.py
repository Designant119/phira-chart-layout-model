#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer 轨迹回归训练：输入 (N,128,feat) → 输出 128 个 x（确定性，非采样）。
损失 = MSE(谱师x) + λ_smooth·平滑(相邻|Δx|) + λ_bound·边界(越界惩罚)。
用法：python3 train_traj.py --data <dir> --ckpt <out> [--epochs 40] [--lr 3e-4] [--feat 8] [--resume]
"""
import json, os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
W = 128
NB = 15


class TrajTransformer(nn.Module):
    def __init__(self, feat=8, d_model=128, nhead=4, layers=3):
        super().__init__()
        self.proj = nn.Linear(feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, W, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model * 4, dropout=0.1,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x) + self.pos
        h = self.enc(h)
        return self.head(h).squeeze(-1)  # (N,128)


def load_npz(d, key='train'):
    a = np.load(os.path.join(BASE, d, f'{key}.npz'))
    return torch.from_numpy(a['X']), torch.from_numpy(a['y']), a['song_ids']


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--feat", type=int, default=8)
    ap.add_argument("--lam_smooth", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", type=str, default=None,
                    help="只加载 model.state_dict() 作为初始权重，start_ep=0/best_val=inf/wait=0，不读 ckpt_state.json")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    Xtr, ytr, _ = load_npz(args.data, 'train')
    Xva, yva, _ = load_npz(args.data, 'val')
    if Xtr.shape[2] != args.feat:
        raise SystemExit(f"数据特征 {Xtr.shape[2]} != --feat {args.feat}")
    print(f"train {Xtr.shape[0]} 窗口 / val {Xva.shape[0]} 窗口 / feat={args.feat}", flush=True)

    model = TrajTransformer(feat=args.feat)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    L1 = nn.L1Loss()

    ckdir = os.path.join(BASE, args.ckpt)
    os.makedirs(ckdir, exist_ok=True)
    start_ep, best_val, best_ep, wait = 0, float('inf'), -1, 0
    if args.init_from:
        ip = args.init_from if os.path.isabs(args.init_from) else os.path.join(BASE, args.init_from)
        ck = torch.load(ip, map_location='cpu')
        sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        model.load_state_dict(sd)
        print(f"[init-from] {ip} → start_ep=0 best_val=inf wait=0", flush=True)
    elif args.resume and os.path.exists(os.path.join(ckdir, 'ckpt_state.json')):
        st = json.load(open(os.path.join(ckdir, 'ckpt_state.json')))
        model.load_state_dict(torch.load(os.path.join(ckdir, 'model.pth'), map_location='cpu')['model'])
        start_ep, best_val, best_ep, wait = st['epoch'], st['best_val'], st['best_epoch'], st['wait']
        print(f"[resume] ep{start_ep} best={best_val:.5f}@{best_ep}", flush=True)

    def run(X, y, grad):
        model.train(grad)
        n = X.shape[0]
        idx = np.random.permutation(n) if grad else np.arange(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, args.batch):
            ids = idx[i:i + args.batch]
            xb, yb = X[ids], y[ids]
            if grad:
                opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            # 平滑：相邻音符位移惩罚（越界大跳惩罚，允许谱师级 300px 内）
            dx = (pred[:, 1:] - pred[:, :-1]).abs()
            target = torch.full_like(dx, 0.4)  # 0.4*675=270px=谱师中位
            smooth = torch.clamp(dx - target, min=0).mean()
            # 边界惩罚
            bound = ((pred.abs() - 1.05).clamp(min=0) ** 2).mean()
            total = loss + args.lam_smooth * smooth + 0.3 * bound
            if grad:
                total.backward()
                opt.step()
            tot += total.item() * len(ids)
            cnt += len(ids)
        return tot / cnt

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        trl = run(Xtr, ytr, True)
        val = run(Xva, yva, False)
        tag = ''
        if val < best_val:
            best_val, best_ep, wait = val, ep + 1, 0
            torch.save({'model': model.state_dict()}, os.path.join(ckdir, 'model.pth'))
            tag = ' *'
        else:
            wait += 1
        json.dump({'epoch': ep + 1, 'best_val': best_val, 'best_epoch': best_ep, 'wait': wait,
                   'lr': args.lr, 'feat': args.feat},
                  open(os.path.join(ckdir, 'ckpt_state.json'), 'w'))
        print(f"[ep {ep+1}/{args.epochs}] train {trl:.4f} val {val:.4f} best {best_val:.4f}@{best_ep} wait{wait}/{args.patience} ({time.time()-t0:.0f}s){tag}", flush=True)
        if wait >= args.patience:
            print("早停")
            break
    print(f"完成: best valCE {best_val:.4f}@{best_ep} → {ckdir}/model.pth")


if __name__ == '__main__':
    train()
