#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""微调实验：加载 v5_low 权重 → 用 6thPecJam 的谱面微调 → 对比 val CE 是否下降。
验证"模型学没学到该谱师习惯"：val = 6thPecJam 未训过的 16 首。
"""
import json, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
from train_layout_v5_low import GRUCls, to_bins, NB

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
DATA = os.path.join(BASE, "train_data_ft6th")
CKPT0 = os.path.join(BASE, "layout_ckpt_v5_low")   # 初始权重（v5_low 全量）
CKPT = os.path.join(BASE, "layout_ckpt_ft6th")     # 微调输出
os.makedirs(CKPT, exist_ok=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, split):
        d = np.load(os.path.join(DATA, f"{split}.npz"))
        self.X = torch.from_numpy(d["X"])
        self.y = torch.from_numpy(to_bins(d["y"]).astype(np.int64))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def evaluate(model, val):
    model.eval(); tot, cnt = 0.0, 0
    with torch.no_grad():
        for Xb, yb in val:
            o = model(Xb)
            tot += nn.CrossEntropyLoss()(o.reshape(-1, NB), yb.reshape(-1)).item() * Xb.shape[0]
            cnt += Xb.shape[0]
    model.train(); return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)

    tr = DS("train"); va = DS("val")
    trl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True)
    val = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False)

    model = GRUCls()
    ck0 = torch.load(os.path.join(CKPT0, "model.pth"), map_location="cpu")
    model.load_state_dict(ck0["model"])
    print(f"[载入 v5_low 权重: best_val={ck0['best_val']:.5f}@{ck0['best_epoch']} epoch={ck0['epoch']}")

    # 基线：原 v5_low 在 6thPecJam 留出歌上的 CE
    base_ce = evaluate(model, val)
    print(f"[基线] 原 v5_low 在 6thPecJam-val 上 CE = {base_ce:.5f}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    best_val = float("inf"); best_epoch = -1; wait = 0
    for ep in range(args.epochs):
        t0 = time.time(); model.train(); run, cnt = 0.0, 0
        for Xb, yb in trl:
            opt.zero_grad()
            o = model(Xb)
            loss = lossf(o.reshape(-1, NB), yb.reshape(-1))
            loss.backward(); opt.step()
            run += loss.item() * Xb.shape[0]; cnt += Xb.shape[0]
        v = evaluate(model, val); el = time.time() - t0
        if v < best_val - 1e-5:
            best_val, best_epoch, wait = v, ep, 0
        else:
            wait += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best_val": best_val, "best_epoch": best_epoch, "wait": wait},
                   os.path.join(CKPT, "model.pth"))
        print(f"[ep {ep+1}/{args.epochs}] ft-trainCE {run/cnt:.5f} ft-valCE {v:.5f} best {best_val:.5f}@{best_epoch} ({el:.0f}s)", flush=True)
        if wait >= args.patience:
            print("[early stop]", flush=True); break
    print(f"基线 {base_ce:.5f} → 微调后 best {best_val:.5f}@{best_epoch}  (Δ={base_ce-best_val:+.5f})")
    json.dump({"base_ce": base_ce, "best_val": best_val, "best_epoch": best_epoch,
               "charter": "6thPecJam", "n_train_win": len(tr), "n_val_win": len(va),
               "lr": args.lr, "epochs": args.epochs}, open(os.path.join(CKPT, "ft_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
