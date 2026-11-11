#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chord_x+level_norm(9维) 排键模型训练：从 v5_low(7维) 迁移权重。
- 迁移：GRU 输入权重前7列拷旧、第8列(chord_x)与第9列(level_norm)都置0；head/bias 直接拷
- 每 epoch 保存 model.pth + ckpt_state.json（resume 用）
- 用法：python3 train_chord.py --data <dir> --ckpt <out_dir> [--epochs 30] [--lr 1e-4] [--feat 9] [--resume]
"""
import json, os, time, argparse
import numpy as np
import torch, torch.nn as nn
import torch.utils.data

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
CKPT0 = os.path.join(BASE, "layout_ckpt_v5_low")
W = 128
NB = 15  # Δx 分桶数（--nb 覆盖；桶宽 = 2*810/NB px）


def make_bins(nb):
    edges = np.linspace(-1.2, 1.2, nb + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def to_bins(y, nb=None):
    nb = nb or NB
    edges, _ = make_bins(nb)
    idx = np.clip(np.digitize(y, edges) - 1, 0, nb - 1)
    return idx


class GRUCls9(nn.Module):
    def __init__(self, feat=9, hidden=64, layers=1, nb=NB):
        super().__init__()
        self.nb = nb
        self.gru = nn.GRU(feat, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, nb)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h)


def migrate_from_v5(model, feat):
    """把 7 维 v5_low 权重迁移到 feat 维模型（第8列 chord_x 与第9列 level 都置0，从0学）"""
    ck0 = torch.load(os.path.join(CKPT0, "model.pth"), map_location="cpu")
    s0 = ck0["model"]
    sd = model.state_dict()
    for k in sd:
        if k not in s0:
            print(f"  [新权重] {k} 置零")
            sd[k].zero_()
            continue
        if sd[k].shape != s0[k].shape:
            # 只处理输入权重扩展：GRU 输入 7->feat
            if k == "gru.weight_ih_l0" and s0[k].shape[1] == 7 and sd[k].shape[1] == feat:
                print(f"  [扩展] {k}: {s0[k].shape} -> {sd[k].shape}（前7列拷贝、第8/9列0）")
                sd[k][:, :7] = s0[k]
                sd[k][:, 7:] = 0.0
            else:
                print(f"  [形状不符] {k}: {s0[k].shape} vs {sd[k].shape} 跳过")
            continue
        sd[k] = s0[k]
    print(f"  迁移完成（v5_low best_val={ck0['best_val']:.5f}@{ck0['best_epoch']}）")
    return ck0


class DS(torch.utils.data.Dataset):
    def __init__(self, npz, nb=NB):
        d = np.load(npz)
        self.X = torch.from_numpy(d["X"])
        self.y = torch.from_numpy(to_bins(d["y"], nb).astype(np.int64))

    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def evaluate(model, loader, nb=NB):
    model.eval(); tot, cnt = 0.0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            o = model(Xb)
            tot += nn.CrossEntropyLoss()(o.reshape(-1, nb), yb.reshape(-1)).item() * Xb.shape[0]
            cnt += Xb.shape[0]
    model.train(); return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--feat", type=int, default=9, help="输入特征维数（9=chord_x+level_norm）")
    ap.add_argument("--nb", type=int, default=15, help="Δx 分桶数（15=108px/桶，36=45px/桶）")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    global NB
    NB = args.nb
    torch.manual_seed(42); np.random.seed(42)
    model = GRUCls9(feat=args.feat, nb=args.nb)
    st_path = os.path.join(args.ckpt, "ckpt_state.json")
    start_ep = 0
    if args.resume and os.path.exists(st_path):
        st = json.load(open(st_path))
        model.load_state_dict(torch.load(os.path.join(args.ckpt, "model.pth"), map_location="cpu")["model"])
        start_ep = st.get("epoch", 0)
        print(f"[resume] 从 epoch {start_ep} 续训")
    else:
        migrate_from_v5(model, args.feat)

    tr = DS(os.path.join(args.data, "train.npz"), args.nb)
    va = DS(os.path.join(args.data, "val.npz"), args.nb)
    # 校验数据维度与 feat 一致
    if tr.X.shape[2] != args.feat:
        raise SystemExit(f"[维度不符] 数据 X.shape[1]={tr.X.shape[2]} 与 --feat {args.feat} 不一致")
    print(f"特征维 feat={args.feat} nb={args.nb}（数据 X.shape={tuple(tr.X.shape)}）")
    trl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True)
    val = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False)
    print(f"train {len(tr)} 窗口 / val {len(va)} 窗口")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    best_val, best_epoch, wait = float("inf"), -1, 0

    os.makedirs(args.ckpt, exist_ok=True)
    for ep in range(start_ep, args.epochs):
        t0 = time.time(); model.train(); run, cnt = 0.0, 0
        for Xb, yb in trl:
            opt.zero_grad()
            o = model(Xb)
            loss = lossf(o.reshape(-1, args.nb), yb.reshape(-1))
            loss.backward(); opt.step()
            run += loss.item() * Xb.shape[0]; cnt += Xb.shape[0]
        v = evaluate(model, val, args.nb)
        if v < best_val - 1e-5:
            best_val, best_epoch, wait = v, ep, 0
        else:
            wait += 1
        # 每 epoch 落盘（防进程清理丢进度）
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best_val": best_val, "best_epoch": best_epoch, "wait": wait},
                   os.path.join(args.ckpt, "model.pth"))
        json.dump({"epoch": ep + 1, "best_val": best_val, "best_epoch": best_epoch, "wait": wait,
                   "lr": args.lr, "feat": args.feat},
                  open(st_path, "w"), ensure_ascii=False, indent=1)
        print(f"[ep {ep+1}/{args.epochs}] trainCE {run/cnt:.5f} valCE {v:.5f} best {best_val:.5f}@{best_epoch} wait{wait}/{args.patience} ({time.time()-t0:.0f}s)", flush=True)
        if wait >= args.patience:
            print("[early stop]", flush=True)
            break
    print(f"完成: best valCE {best_val:.5f}@{best_epoch} → {args.ckpt}/model.pth")


if __name__ == "__main__":
    main()
