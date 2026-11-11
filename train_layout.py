#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排键布局模型训练：输入窗口 128 音符 × 6 特征 → 输出每音符全局 x（归一化）。
支持 mlp/gru、resume、checkpoint（ckpt_state.json 供守护读取）。
用法：
  python3 train_layout.py --model gru --epochs 30
  python3 train_layout.py --model gru --resume   # 续训
"""
import json, os, time, argparse, math
import numpy as np
import torch
import torch.nn as nn

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
DATA = os.path.join(BASE, "train_data")
CKPT = os.path.join(BASE, "layout_ckpt")
os.makedirs(CKPT, exist_ok=True)


class MLP(nn.Module):
    def __init__(self, feat=6, hidden=256, W=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat * W, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, W),
        )

    def forward(self, x):
        B = x.shape[0]
        return self.fc(x.reshape(B, -1))


class GRU(nn.Module):
    def __init__(self, feat=6, hidden=64, layers=1, W=128):
        super().__init__()
        self.gru = nn.GRU(feat, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        B = x.shape[0]
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)  # (B, W)


class LayoutDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        d = np.load(os.path.join(DATA, f"{split}.npz"))
        self.X = torch.from_numpy(d["X"])
        self.y = torch.from_numpy(d["y"])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_state():
    p = os.path.join(CKPT, "ckpt_state.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {}


def save_state(st):
    p = os.path.join(CKPT, "ckpt_state.json")
    tmp = p + ".tmp"
    json.dump(st, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "gru"], default="gru")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-epoch", type=int, default=0, help="调试：只跑前 N epoch")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dev = "cpu"
    st = load_state()
    # resume 时继承原模型类型
    model_name = args.model
    if args.resume and st.get("model"):
        model_name = st["model"]
    model = GRU(feat=6, hidden=args.hidden, layers=args.layers) if model_name == "gru" else MLP(feat=6, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.MSELoss()

    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    if args.resume and os.path.exists(os.path.join(CKPT, "model.pth")):
        ck = torch.load(os.path.join(CKPT, "model.pth"), map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0)
        best_val = ck.get("best_val", float("inf"))
        best_epoch = ck.get("best_epoch", -1)
        wait = ck.get("wait", 0)
        print(f"[resume] epoch={start_epoch} best_val={best_val:.4f}@{best_epoch} wait={wait}", flush=True)

    train_ds = LayoutDataset("train")
    val_ds = LayoutDataset("val")
    print(f"train {len(train_ds)} val {len(val_ds)}", flush=True)
    if args.smoke:
        train_ds.X = train_ds.X[:64]
        train_ds.y = train_ds.y[:64]
        val_ds.X = val_ds.X[:32]
        val_ds.y = val_ds.y[:32]

    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    def evaluate():
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for Xb, yb in va_loader:
                o = model(Xb)
                tot += lossf(o, yb).item() * Xb.shape[0]
                cnt += Xb.shape[0]
        model.train()
        return tot / cnt

    print(f"=== 开始训练 {model_name} epochs={args.epochs} batch={args.batch} lr={args.lr} ===", flush=True)
    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        run, cnt = 0.0, 0
        for Xb, yb in tr_loader:
            opt.zero_grad()
            o = model(Xb)
            loss = lossf(o, yb)
            loss.backward()
            opt.step()
            run += loss.item() * Xb.shape[0]
            cnt += Xb.shape[0]
        trl = run / cnt
        val = evaluate()
        el = time.time() - t0
        improved = val < best_val - 1e-5
        if improved:
            best_val, best_epoch = val, ep
            wait = 0
        else:
            wait += 1
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": ep + 1,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "wait": wait,
        }, os.path.join(CKPT, "model.pth"))
        save_state({
            "model": model_name, "epoch": ep + 1, "total_epochs": args.epochs,
            "best_val": round(float(best_val), 5), "best_epoch": best_epoch,
            "wait": wait, "finished": False, "train_loss": round(float(trl), 5),
            "val_loss": round(float(val), 5), "time_s": round(el, 1),
        })
        print(f"[ep {ep+1}/{args.epochs}] train {trl:.5f} val {val:.5f} best {best_val:.5f}@{best_epoch} wait {wait} ({el:.0f}s)", flush=True)
        if args.max_epoch and ep + 1 >= args.max_epoch:
            break
        if wait >= args.patience:
            print(f"[early stop] best {best_val:.5f}@{best_epoch}", flush=True)
            break

    save_state({
        "model": model_name, "epoch": ep + 1, "total_epochs": args.epochs,
        "best_val": round(float(best_val), 5), "best_epoch": best_epoch,
        "wait": wait, "finished": True, "val_loss": round(float(val), 5),
    })
    print(f"DONE best_val={best_val:.5f} best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
