#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4：分布建模——dx 分桶分类（学习交替/分布模式，而非回归均值）。
复用 train_data_v3（分线窗口），标签 dx → 15 桶。
推理：softmax 期望（E[dx]）累积生成 x。
"""
import json, os, time, argparse
import numpy as np
import torch
import torch.nn as nn

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
DATA = os.path.join(BASE, "train_data_v3")
CKPT = os.path.join(BASE, "layout_ckpt_v4")
os.makedirs(CKPT, exist_ok=True)

NB = 15          # 桶数
BIN_LO, BIN_HI = -1.2, 1.2
BIN_EDGES = np.linspace(BIN_LO, BIN_HI, NB + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2


def to_bins(y):
    """y (N,W) dx 值 → (N,W) 桶索引"""
    idx = np.clip(np.digitize(y, BIN_EDGES) - 1, 0, NB - 1)
    return idx


class GRUCls(nn.Module):
    def __init__(self, feat=7, hidden=64, layers=1, nb=NB):
        super().__init__()
        self.gru = nn.GRU(feat, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, nb)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h)  # (B, W, NB)


class DS(torch.utils.data.Dataset):
    def __init__(self, split):
        d = np.load(os.path.join(DATA, f"{split}.npz"))
        self.X = torch.from_numpy(d["X"])
        self.y = torch.from_numpy(to_bins(d["y"]).astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def save_state(st):
    tmp = os.path.join(CKPT, "ckpt_state.json.tmp")
    json.dump(st, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(CKPT, "ckpt_state.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    st = {}
    if args.resume and os.path.exists(os.path.join(CKPT, "ckpt_state.json")):
        st = json.load(open(os.path.join(CKPT, "ckpt_state.json")))
    model = GRUCls()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    start_epoch = 0
    best_val = float("inf"); best_epoch = -1; wait = 0
    if args.resume and os.path.exists(os.path.join(CKPT, "model.pth")):
        ck = torch.load(os.path.join(CKPT, "model.pth"), map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]; best_val = ck["best_val"]; best_epoch = ck["best_epoch"]; wait = ck["wait"]
        print(f"[resume] epoch={start_epoch} best={best_val:.4f}@{best_epoch}", flush=True)
    tr = DS("train"); va = DS("val")
    trl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True)
    val = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False)

    def evaluate():
        model.eval(); tot, cnt = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val:
                o = model(Xb)
                tot += lossf(o.reshape(-1, NB), yb.reshape(-1)).item() * Xb.shape[0]
                cnt += Xb.shape[0]
        model.train(); return tot / cnt

    for ep in range(start_epoch, args.epochs):
        t0 = time.time(); model.train(); run, cnt = 0.0, 0
        for Xb, yb in trl:
            opt.zero_grad()
            o = model(Xb)
            loss = lossf(o.reshape(-1, NB), yb.reshape(-1))
            loss.backward(); opt.step()
            run += loss.item() * Xb.shape[0]; cnt += Xb.shape[0]
        v = evaluate(); el = time.time() - t0
        if v < best_val - 1e-5:
            best_val, best_epoch, wait = v, ep, 0
        else:
            wait += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best_val": best_val, "best_epoch": best_epoch, "wait": wait},
                   os.path.join(CKPT, "model.pth"))
        save_state({"epoch": ep + 1, "total_epochs": args.epochs, "best_val": round(float(best_val), 5),
                    "best_epoch": best_epoch, "wait": wait, "finished": False,
                    "train_ce": round(float(run / cnt), 5), "val_ce": round(float(v), 5)})
        print(f"[ep {ep+1}/{args.epochs}] ce {run/cnt:.5f} / {v:.5f} best {best_val:.5f}@{best_epoch} wait {wait} ({el:.0f}s)", flush=True)
        if wait >= args.patience:
            print(f"[early stop]", flush=True); break
    save_state({"epoch": ep + 1, "total_epochs": args.epochs, "best_val": round(float(best_val), 5),
                "best_epoch": best_epoch, "wait": wait, "finished": True})
    print(f"DONE best={best_val:.5f}@{best_epoch}", flush=True)


if __name__ == "__main__":
    main()
