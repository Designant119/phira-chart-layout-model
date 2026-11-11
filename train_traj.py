#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer 轨迹回归训练：输入 (N,128,feat) → 输出 128 个 x（确定性，非采样）。
损失 = MSE(谱师x) + λ_smooth·平滑(相邻|Δx|超阈值惩罚) + λ_bound·边界(越界惩罚)。
选模/早停用 val 纯 MSE（正则项只参与训练，不参与模型选择）。
--data/--ckpt/--init-from 相对路径相对 cwd（绝对原样），与 build_traj_npz --out 口径一致。
用法：python3 train_traj.py --data <dir> --ckpt <out> [--epochs 40] [--lr 3e-4] [--feat 8] [--resume]
"""
import json, os, time, argparse
import numpy as np
import torch
import torch.nn as nn

from traj_common import W, artifact_path

SMOOTH_TARGET = 0.4  # 相邻位移惩罚阈值（0.4*675=270px≈谱师中位）
LAM_BOUND = 0.3      # 边界惩罚权重默认（--lam-bound 可覆盖）


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

    def forward(self, x, padding_mask=None):
        """padding_mask: (N,S) bool，True=padding 位（推理补零窗口用；训练全窗不传）。
        不改变 state_dict 结构，兼容旧 ckpt。"""
        h = self.proj(x) + self.pos
        h = self.enc(h, src_key_padding_mask=padding_mask)
        return self.head(h).squeeze(-1)  # (N,128)


def load_npz(d, key='train'):
    a = np.load(os.path.join(d, f'{key}.npz'))
    return torch.from_numpy(a['X']), torch.from_numpy(a['y']), a['song_ids']


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--feat", type=int, default=8)
    ap.add_argument("--lam_smooth", type=float, default=0.5)
    ap.add_argument("--lam_bound", type=float, default=LAM_BOUND)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", type=str, default=None,
                    help="只加载 model.state_dict() 作为初始权重，start_ep=0/best_val=inf/wait=0，不读 ckpt_state.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datadir = artifact_path(args.data)
    ckdir = artifact_path(args.ckpt)
    Xtr, ytr, _ = load_npz(datadir, 'train')
    Xva, yva, _ = load_npz(datadir, 'val')
    if Xtr.shape[2] != args.feat:
        raise SystemExit(f"数据特征 {Xtr.shape[2]} != --feat {args.feat}")
    if Xtr.shape[1] != W:
        raise SystemExit(f"数据窗长 {Xtr.shape[1]} != W {W}")
    print(f"train {Xtr.shape[0]} 窗口 / val {Xva.shape[0]} 窗口 / feat={args.feat} / seed={args.seed}", flush=True)

    model = TrajTransformer(feat=args.feat)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    os.makedirs(ckdir, exist_ok=True)
    start_ep, best_val, best_ep, wait = 0, float('inf'), -1, 0
    if args.init_from:
        ip = artifact_path(args.init_from)
        ck = torch.load(ip, map_location='cpu')
        sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        model.load_state_dict(sd)
        print(f"[init-from] {ip} → start_ep=0 best_val=inf wait=0", flush=True)
    elif args.resume and os.path.exists(os.path.join(ckdir, 'model.pth')) \
            and os.path.exists(os.path.join(ckdir, 'ckpt_state.json')):
        st = json.load(open(os.path.join(ckdir, 'ckpt_state.json')))
        ck = torch.load(os.path.join(ckdir, 'model.pth'), map_location='cpu')
        model.load_state_dict(ck['model'])
        if 'opt' in ck:
            opt.load_state_dict(ck['opt'])  # 恢复 AdamW 动量，避免续训损失跳变
        start_ep = st['epoch']
        # 旧 ckpt_state 的 best_val 是含正则的总损失（口径不同），重置按 MSE 重选
        if st.get('metric') == 'val_mse':
            best_val, best_ep, wait = st['best_val'], st['best_epoch'], st['wait']
        else:
            print("[resume] 旧 state 指标口径为总损失，best_val 重置为 inf", flush=True)
        print(f"[resume] ep{start_ep} best={best_val:.5f}@{best_ep}", flush=True)

    def run(X, y, grad):
        """返回 (avg_total, avg_mse)。avg_mse 纯 MSE 用于选模；total 含正则仅参考打印。"""
        model.train(grad)
        n = X.shape[0]
        idx = np.random.permutation(n) if grad else np.arange(n)
        s_tot = s_mse = 0.0
        cnt = 0
        for i in range(0, n, args.batch):
            ids = idx[i:i + args.batch]
            xb, yb = X[ids], y[ids]
            if grad:
                opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            # 平滑：相邻音符位移惩罚（越界大跳惩罚，允许谱师级 300px 内）
            dx = (pred[:, 1:] - pred[:, :-1]).abs()
            target = torch.full_like(dx, SMOOTH_TARGET)
            smooth = torch.clamp(dx - target, min=0).mean()
            # 边界惩罚
            bound = ((pred.abs() - 1.05).clamp(min=0) ** 2).mean()
            total = loss + args.lam_smooth * smooth + args.lam_bound * bound
            if grad:
                total.backward()
                opt.step()
            bs = len(ids)
            s_tot += total.item() * bs
            s_mse += loss.item() * bs
            cnt += bs
        return s_tot / cnt, s_mse / cnt

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        trl, trm = run(Xtr, ytr, True)
        vt, vm = run(Xva, yva, False)
        tag = ''
        if vm < best_val:
            best_val, best_ep, wait = vm, ep + 1, 0
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict()},
                       os.path.join(ckdir, 'model.pth'))
            tag = ' *'
        else:
            wait += 1
        json.dump({'epoch': ep + 1, 'best_val': best_val, 'best_epoch': best_ep, 'wait': wait,
                   'lr': args.lr, 'feat': args.feat, 'metric': 'val_mse',
                   'lam_smooth': args.lam_smooth, 'lam_bound': args.lam_bound},
                  open(os.path.join(ckdir, 'ckpt_state.json'), 'w'))
        print(f"[ep {ep+1}/{args.epochs}] train {trl:.4f}(mse {trm:.4f}) "
              f"val {vt:.4f}(mse {vm:.4f}) bestMSE {best_val:.4f}@{best_ep} "
              f"wait{wait}/{args.patience} ({time.time()-t0:.0f}s){tag}", flush=True)
        if wait >= args.patience:
            print("早停")
            break
    print(f"完成: best valMSE {best_val:.4f}@{best_ep} → {ckdir}/model.pth")


if __name__ == '__main__':
    train()
