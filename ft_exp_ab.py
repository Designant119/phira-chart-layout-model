#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 微调实验对比：
A=混数据(60%原+40%ft)完整微调  B=纯ft冻结GRU只训head
统一评估: ft-val CE（学没学到6thPecJam习惯）+ orig-val CE（普通歌有没有遗忘）
"""
import json, os, time
import numpy as np
import torch, torch.nn as nn
from train_layout_v5_low import GRUCls, to_bins, NB

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
FT = os.path.join(BASE, "train_data_ft6th")     # 6thPecJam 数据
ORIG = os.path.join(BASE, "train_data_v4_low")  # 原全量数据
CKPT0 = os.path.join(BASE, "layout_ckpt_v5_low")


class DS(torch.utils.data.Dataset):
    def __init__(self, npz):
        d = np.load(npz)
        self.X = torch.from_numpy(d["X"])
        self.y = torch.from_numpy(to_bins(d["y"]).astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


def evaluate(model, loader):
    model.eval(); tot, cnt = 0.0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            o = model(Xb)
            tot += nn.CrossEntropyLoss()(o.reshape(-1, NB), yb.reshape(-1)).item() * Xb.shape[0]
            cnt += Xb.shape[0]
    model.train(); return tot / cnt


def make_mixA(seed=42, n_orig=1875):
    """60%原 + 40%ft：原数据随机抽 n_orig 窗口 + ft 全部 1250 窗口 = 3125"""
    rng = np.random.RandomState(seed)
    od = np.load(os.path.join(ORIG, "train.npz"))
    idx = rng.choice(od["X"].shape[0], n_orig, replace=False)
    fd = np.load(os.path.join(FT, "train.npz"))
    X = np.concatenate([od["X"][idx], fd["X"]], axis=0)
    y = np.concatenate([od["y"][idx], fd["y"]], axis=0)
    np.savez(os.path.join(BASE, "train_data_mixA.npz"), X=X, y=y)
    print(f"[mixA] 原{n_orig} + ft{fd['X'].shape[0]} = {X.shape[0]} 窗口 (ft占比{fd['X'].shape[0]/X.shape[0]*100:.0f}%)", flush=True)


def train(tag, Xnpz, lr, epochs, freeze_gru=False, patience=8, batch=64):
    torch.manual_seed(42); np.random.seed(42)
    CKPT = os.path.join(BASE, f"layout_ckpt_ft{tag}")
    os.makedirs(CKPT, exist_ok=True)
    model = GRUCls()
    ck0 = torch.load(os.path.join(CKPT0, "model.pth"), map_location="cpu")
    model.load_state_dict(ck0["model"])
    if freeze_gru:
        for p in model.gru.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lossf = nn.CrossEntropyLoss()
    tr = DS(Xnpz); trl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True)
    fval = torch.utils.data.DataLoader(DS(os.path.join(FT, "val.npz")), batch_size=batch, shuffle=False)
    ova = torch.utils.data.DataLoader(DS(os.path.join(ORIG, "val.npz")), batch_size=batch, shuffle=False)
    # 原val抽样2000窗口（固定seed，减少评估耗时）
    odv = np.load(os.path.join(ORIG, "val.npz"))
    rng = np.random.RandomState(7)
    oidx = rng.choice(odv["X"].shape[0], min(2000, odv["X"].shape[0]), replace=False)
    import tempfile
    tmp = os.path.join(BASE, "orig_val_sample.npz")
    np.savez(tmp, X=odv["X"][oidx], y=odv["y"][oidx])
    ova_s = torch.utils.data.DataLoader(DS(tmp), batch_size=batch, shuffle=False)
    print(f"[{tag}] 载入v5_low，freeze_gru={freeze_gru} lr={lr} 数据={Xnpz}", flush=True)
    best_f, best_o, wait = float("inf"), float("inf"), 0
    for ep in range(epochs):
        t0 = time.time(); model.train(); run, cnt = 0.0, 0
        for Xb, yb in trl:
            opt.zero_grad()
            o = model(Xb)
            loss = lossf(o.reshape(-1, NB), yb.reshape(-1))
            loss.backward(); opt.step()
            run += loss.item() * Xb.shape[0]; cnt += Xb.shape[0]
        vf = evaluate(model, fval); vo = evaluate(model, ova_s)
        if vf < best_f - 1e-5: best_f, wait = vf, 0
        else: wait += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1,
                    "best_f": best_f, "wait": wait}, os.path.join(CKPT, "model.pth"))
        print(f"[{tag} ep{ep+1}/{epochs}] trainCE {run/cnt:.5f} | ft-val {vf:.5f} (best {best_f:.5f}) | orig-val(s2000) {vo:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if wait >= patience:
            print(f"[{tag} early stop]", flush=True); break
    json.dump({"tag": tag, "freeze_gru": freeze_gru, "lr": lr, "epochs": ep + 1,
               "best_ft_val": best_f}, open(os.path.join(CKPT, "report.json"), "w"), indent=1)
    return best_f


if __name__ == "__main__":
    make_mixA()
    bestA = train("A", os.path.join(BASE, "train_data_mixA.npz"), lr=1e-4, epochs=25)
    bestB = train("B", os.path.join(FT, "train.npz"), lr=3e-4, epochs=25, freeze_gru=True)
    print(f"\n=== 汇总 ===")
    print(f"A(混数据): best ft-val CE = {bestA:.5f}")
    print(f"B(冻结GRU): best ft-val CE = {bestB:.5f}")
    print(f"对照: 原v5_low 在ft-val = 1.79232（基线）；纯ft全参微调(上次实验) = 1.78068")
