#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量 4K mania 列分类训练（因果 Transformer，逐音窗口）。"""
import os, time, argparse
import numpy as np, torch, torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
W = 128; STRIDE = 64


def windows(ids, X, y):
    out = []
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            n = i - start
            for s in range(0, max(1, n - W + 1), STRIDE):
                e = min(n, s + W)
                if e - s < 8: continue
                xw = np.zeros((W, X.shape[1]), np.float32); yw = np.zeros(W, np.int64)
                xw[:e - s] = X[start + s:start + e]; yw[:e - s] = y[start + s:start + e]
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
        return self.out(self.enc(self.inp(x), mask=self.mask))


def run(args):
    d = np.load(os.path.join(ROOT, "mania_4k_full.npz"))
    sets = {sp: windows(d[f"{sp}_ids"], d[f"{sp}_X"], d[f"{sp}_y"]) for sp in ("train", "val", "test")}
    print({k: len(v) for k, v in sets.items()}, flush=True)
    model = Net(d["train_X"].shape[1], 4, args.d, args.layers, args.heads)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = nb = acc = 0
        order = np.random.permutation(len(sets["train"]))
        bs = 16
        for bi in range(0, len(order), bs):
            idx = order[bi:bi + bs]
            xs = np.stack([sets["train"][i][0] for i in idx])
            ys = np.stack([sets["train"][i][1] for i in idx])
            vl = [sets["train"][i][2] for i in idx]
            logits = model(torch.from_numpy(xs))
            loss = sum(nn.functional.cross_entropy(logits[j, :v], torch.from_numpy(ys[j, :v])) for j, v in enumerate(vl)) / len(idx)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
            acc += sum(float((logits[j, :v].argmax(-1) == torch.from_numpy(ys[j, :v])).float().mean()) for j, v in enumerate(vl)) / len(idx)
            if bi % 100 == 0 and bi:
                print(f"  {bi}/{len(order)} loss {tot/nb:.4f} acc {acc/nb:.3f}", flush=True)
        sched.step()
        model.eval()
        def ev(sp):
            a = n = 0
            with torch.no_grad():
                for bi in range(0, len(sets[sp]), 16):
                    idx = range(bi, min(bi + 16, len(sets[sp])))
                    xs = np.stack([sets[sp][i][0] for i in idx])
                    ys = [sets[sp][i][1] for i in idx]
                    vl = [sets[sp][i][2] for i in idx]
                    logits = model(torch.from_numpy(xs))
                    for j, v in enumerate(vl):
                        a += int((logits[j, :v].argmax(-1) == torch.from_numpy(ys[j][:v])).sum()); n += v
            return a / n
        va = ev("val"); te = ev("test")
        print(f"[ep {ep+1}] loss {tot/nb:.4f} val {va:.4f} test {te:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if va > best:
            best = va
            torch.save({"model": model.state_dict(), "ind": d["train_X"].shape[1], "d": args.d,
                        "layers": args.layers, "heads": args.heads},
                       os.path.join(ROOT, "ckpt_mania4k_full.pt"))
    print("best val", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    run(ap.parse_args())
