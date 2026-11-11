#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nb36 底座 + 6thPecJam 谱面微调（9维 feat=9, nb=36）。
- 底座: layout_ckpt_chord9_nb36/model.pth (GRUCls9 feat=9 nb=36)
- 数据: train_data_ft6th9/{train,val}.npz（复用旧 ft6th split 划分）
- 先算底座在 ft-val 上的基线 CE，再全参微调，对比 Δ
"""
import json, os, time, argparse
import numpy as np
import torch, torch.nn as nn
from train_chord import GRUCls9, to_bins

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
DATA = os.path.join(BASE, "train_data_ft6th9")
CKPT0 = os.path.join(BASE, "layout_ckpt_chord9_nb36")   # 底座
CKPT = os.path.join(BASE, "layout_ckpt_ft9nb36")         # 微调输出
NB = 36


class DS(torch.utils.data.Dataset):
    def __init__(self, split):
        d = np.load(os.path.join(DATA, f"{split}.npz"))
        self.X = torch.from_numpy(d["X"])
        self.y = torch.from_numpy(to_bins(d["y"], NB).astype(np.int64))
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
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    os.makedirs(CKPT, exist_ok=True)

    model = GRUCls9(feat=9, nb=NB)
    st_path = os.path.join(CKPT, "ckpt_state.json")
    start_ep = 0
    base_ce = None
    if args.resume and os.path.exists(st_path):
        st = json.load(open(st_path))
        model.load_state_dict(torch.load(os.path.join(CKPT, "model.pth"), map_location="cpu")["model"])
        start_ep = st.get("epoch", 0)
        base_ce = st.get("base_ce")
        print(f"[resume] 从 epoch {start_ep} 续训")
    else:
        ck0 = torch.load(os.path.join(CKPT0, "model.pth"), map_location="cpu")
        model.load_state_dict(ck0["model"])
        print(f"[载入 nb36 底座: best_val={ck0['best_val']:.5f}@{ck0['best_epoch']} epoch={ck0['epoch']} head={ck0['model']['head.weight'].shape[0]}类]")

    tr = DS("train"); va = DS("val")
    print(f"train {len(tr)} 窗口 / val {len(va)} 窗口（feat={tr.X.shape[2]} nb={NB}）")
    trl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True)
    val = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False)

    if base_ce is None:
        base_ce = evaluate(model, val)
    print(f"[基线] nb36 底座在 6thPecJam-val 上 CE = {base_ce:.5f}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    best_val = float("inf"); best_epoch = -1; wait = 0
    for ep in range(start_ep, args.epochs):
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
                    "epoch": ep + 1, "best_val": best_val, "best_epoch": best_epoch, "wait": wait,
                    "base_ce": base_ce},
                   os.path.join(CKPT, "model.pth"))
        json.dump({"epoch": ep + 1, "best_val": best_val, "best_epoch": best_epoch, "wait": wait,
                   "base_ce": base_ce, "lr": args.lr, "feat": 9, "nb": NB},
                  open(st_path, "w"), ensure_ascii=False, indent=1)
        print(f"[ep {ep+1}/{args.epochs}] ft-trainCE {run/cnt:.5f} ft-valCE {v:.5f} best {best_val:.5f}@{best_epoch} ({el:.0f}s)", flush=True)
        if wait >= args.patience:
            print("[early stop]", flush=True); break
    print(f"基线 {base_ce:.5f} -> 微调后 best {best_val:.5f}@{best_epoch}  (Δ={base_ce-best_val:+.5f})")


if __name__ == "__main__":
    main()
