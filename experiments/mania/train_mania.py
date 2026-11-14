#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mania 4K 列分类：因果 Transformer，逐音预测列号。"""
import os, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
W = 128; STRIDE = 64


def windows(ids, X, y, keys):
    out = []
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            n = i - start
            for s in range(0, max(1, n - W + 1), STRIDE):
                e = min(n, s + W)
                if e - s < 8:
                    continue
                xw = np.zeros((W, X.shape[1]), np.float32); yw = np.zeros(W, np.int64)
                xw[:e - s] = X[start + s:start + e]
                yw[:e - s] = y[start + s:start + e]
                out.append((xw, yw, e - s))
            start = i
    return out


class Net(nn.Module):
    def __init__(self, ind, keys, d=128, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(ind, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, keys)
        self.register_buffer("mask", torch.triu(torch.ones(W, W), 1).bool())

    def forward(self, x):
        h = self.inp(x)
        h = self.enc(h, mask=self.mask)
        return self.out(h)


def run(args):
    d = np.load(os.path.join(ROOT, "mania_4k.npz"))
    sets = {}
    for sp in ("train", "val", "test"):
        sets[sp] = windows(d[f"{sp}_ids"], d[f"{sp}_X"], d[f"{sp}_y"], 4)
    print({k: len(v) for k, v in sets.items()}, flush=True)
    model = Net(d["train_X"].shape[1], 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0; nb = 0; acc = 0
        order = np.random.permutation(len(sets["train"]))
        for bi, i in enumerate(order):
            x, y, vlen = sets["train"][i]
            xb = torch.from_numpy(x[None]); yb = torch.from_numpy(y[None])
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits[0, :vlen], yb[0, :vlen])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
            acc += float((logits[0, :vlen].argmax(-1) == yb[0, :vlen]).float().mean())
            if bi % 100 == 0 and bi:
                print(f"  {bi}/{len(order)} loss {tot/nb:.4f} acc {acc/nb:.3f}", flush=True)
        model.eval()
        def ev(sp):
            a = 0; n = 0
            with torch.no_grad():
                for x, y, vlen in sets[sp]:
                    logits = model(torch.from_numpy(x[None]))
                    a += int((logits[0, :vlen].argmax(-1) == torch.from_numpy(y[:vlen])).sum()); n += vlen
            return a / n
        va = ev("val"); te = ev("test")
        print(f"[ep {ep+1}] loss {tot/nb:.4f} val acc {va:.4f} test acc {te:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if va > best:
            best = va; torch.save({"model": model.state_dict(), "ind": d["train_X"].shape[1]},
                                  os.path.join(ROOT, "ckpt_mania4k.pt"))
    print("best val", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=12)
    run(ap.parse_args())
