#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scheduled sampling 微调：训练时把 prev/streak/手位换成模型自己的预测。"""
import os, time, argparse
import numpy as np, torch, torch.nn as nn
from train_4k_full import Net, windows

ROOT = os.path.dirname(os.path.abspath(__file__))


def onehot(v):
    o = np.zeros(5, np.float32); o[int(np.clip(v + 1, 0, 4))] = 1.0
    return o


def make_ss(X, logits, vlen):
    Y = X.copy(); pred = logits.argmax(-1)
    run = 0; lastL = lastR = -1
    for i in range(vlen):
        p1 = pred[i - 1] if i > 0 else -1
        p2 = pred[i - 2] if i > 1 else -1
        Y[i, 5] = min(run / 10.0, 1.0)
        Y[i, 6:11] = onehot(p1); Y[i, 11:16] = onehot(p2)
        Y[i, 16] = lastL / 3.0; Y[i, 17] = lastR / 3.0
        p = int(pred[i])
        if i == 0: run = 1
        elif p == int(pred[i - 1]): run += 1
        else: run = 1
        if p < 2: lastL = p
        else: lastR = p
    Y[vlen:] = 0
    return Y


def run(args):
    d = np.load(os.path.join(ROOT, "mania_4k_full.npz"))
    sets = {sp: windows(d[f"{sp}_ids"], d[f"{sp}_X"], d[f"{sp}_y"]) for sp in ("train", "val", "test")}
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Net(18, 4, ck["d"], ck["layers"], ck["heads"]); model.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = nb = 0
        order = np.random.permutation(len(sets["train"]))
        for bi in range(0, len(order), 16):
            idx = order[bi:bi + 16]
            xs = np.stack([sets["train"][i][0] for i in idx])
            ys = np.stack([sets["train"][i][1] for i in idx])
            vl = [sets["train"][i][2] for i in idx]
            xb = torch.from_numpy(xs)
            logits = model(xb)
            loss = sum(nn.functional.cross_entropy(logits[j, :v], torch.from_numpy(ys[j, :v])) for j, v in enumerate(vl)) / len(idx)
            # 第二遍：用自己的预测替换历史
            ss = np.stack([make_ss(xs[j], logits[j], vl[j]) for j in range(len(idx))])
            logits2 = model(torch.from_numpy(ss))
            loss2 = sum(nn.functional.cross_entropy(logits2[j, :v], torch.from_numpy(ys[j, :v])) for j, v in enumerate(vl)) / len(idx)
            (loss + loss2).backward(); opt.step(); opt.zero_grad()
            tot += float(loss); nb += 1
            if bi % 100 == 0 and bi:
                print(f"  {bi}/{len(order)} loss {tot/nb:.4f}", flush=True)
        model.eval()
        def ev(sp):
            a = n = 0
            with torch.no_grad():
                for bi in range(0, len(sets[sp]), 16):
                    idx = range(bi, min(bi + 16, len(sets[sp])))
                    xs = np.stack([sets[sp][i][0] for i in idx]); ys = [sets[sp][i][1] for i in idx]
                    vl = [sets[sp][i][2] for i in idx]; logits = model(torch.from_numpy(xs))
                    for j, v in enumerate(vl):
                        a += int((logits[j, :v].argmax(-1) == torch.from_numpy(ys[j][:v])).sum()); n += v
            return a / n
        va = ev("val"); te = ev("test")
        print(f"[ss ep {ep+1}] loss {tot/nb:.4f} val {va:.4f} test {te:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if va > best:
            best = va
            torch.save({"model": model.state_dict(), "ind": 18, "d": ck["d"], "layers": ck["layers"], "heads": ck["heads"]},
                       os.path.join(ROOT, "ckpt_mania4k_full_ss.pt"))
    print("best", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ckpt", type=str, default=os.path.join(ROOT, "ckpt_mania4k_full.pt"))
    run(ap.parse_args())
