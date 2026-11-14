#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4K 列模型逐音自回归生成列号（和弦互斥），输出 gen_<name>.json。

用法：
  python3 gen_mania.py --osu /path/chart.osu --ckpt /path/ckpt_mania4k.pt --out gen.json
"""
import os, json, re, argparse
import numpy as np, torch
from train_mania import Net

W = 128


def parse(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    keys = int(re.search(r"CircleSize:\s*(\d+)", txt).group(1))
    body = txt.split("[HitObjects]", 1)[-1]
    notes = []
    for line in body.splitlines():
        p = line.strip().split(",")
        if len(p) < 5:
            continue
        x, t, typ = float(p[0]), int(p[2]), int(p[3])
        col = min(max(int(np.floor(x * keys / 512.0)), 0), keys - 1)
        hold = bool(typ & 128)
        end = int(p[5]) if hold and len(p) > 5 and p[5].strip().isdigit() else t
        notes.append([t / 1000.0, col, 2 if hold else 1, max(0.0, (end - t) / 1000.0)])
    notes.sort(key=lambda n: (n[0], n[1]))
    return keys, notes


def static_feats(notes):
    n = len(notes)
    t = np.array([q[0] for q in notes])
    typ = np.array([q[2] for q in notes])
    dur = np.array([q[3] for q in notes])
    dt = np.zeros(n); dt[1:] = np.clip(np.log1p(t[1:] - t[:-1]) / 3.0, 0, 1)
    chord = np.zeros(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        chord[i:j + 1] = min(j - i + 1, 4) / 4.0
        i = j + 1
    dens = np.zeros(n); j0 = 0
    for i in range(n):
        while t[i] - t[j0] > 0.5:
            j0 += 1
        j1 = i
        while j1 + 1 < n and t[j1 + 1] - t[i] <= 0.5:
            j1 += 1
        dens[i] = min((j1 - j0 + 1) / 20.0, 1.0)
    return t, dt, chord, dens, typ, dur


def onehot(v, k=4):
    oh = np.zeros(k + 1, np.float32)
    oh[int(np.clip(v + 1, 0, k))] = 1.0
    return oh


def make_vec(i, st, prev1, prev2, streak, lastL, lastR, win):
    """填第 i 音特征；win 是已填好的历史行缓存，只补静态列。"""
    dt, chord, dens, typ, dur = st
    win[i, 0] = dt[i]; win[i, 1] = chord[i]; win[i, 2] = dens[i]
    win[i, 3] = float(typ[i] == 2); win[i, 4] = min(dur[i] / 2.0, 1.0)
    win[i, 5] = min(streak / 10.0, 1.0)
    win[i, 6:11] = onehot(prev1); win[i, 11:16] = onehot(prev2)
    win[i, 16] = lastL / 3.0; win[i, 17] = lastR / 3.0


def load_trans(path):
    d = np.load(path)
    cnt = np.ones((4, 4), np.float64)
    for sp in ("train",):
        y = d[f"{sp}_y"]; ids = d[f"{sp}_ids"]; X = d[f"{sp}_X"]
        for i in range(1, len(y)):
            if ids[i] == ids[i - 1] and X[i, 0] > 0:
                cnt[y[i - 1], y[i]] += 1
    return cnt / cnt.sum(0, keepdims=True)


def gen(osu, ckpt, out_path, beta=0.0, trans=None, jack_bonus=0.0):
    keys, notes = parse(osu)
    if keys != 4:
        raise SystemExit(f"只支持 4K，当前 CircleSize={keys}")
    sf = static_feats(notes); t = sf[0]; st = sf[1:]
    n = len(notes)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Net(18, 4)
    model.load_state_dict(ck["model"]); model.eval()
    P = load_trans(trans) if (beta and trans) else None

    buf = np.zeros((n, 18), np.float32)
    pred = np.zeros(n, np.int64); prev1 = prev2 = -1; streak = 0; lastL = lastR = -1
    i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        logits = []
        for k in range(i, j + 1):
            make_vec(k, st, prev1, prev2, streak, lastL, lastR, buf)
            s = max(0, k - W + 1)
            L = k - s + 1
            win = np.zeros((W, 18), np.float32)
            win[:L] = buf[s:k + 1]
            with torch.no_grad():
                logits.append(model(torch.from_numpy(win[None]))[0, L - 1].numpy())
        used = set()
        for k in range(i, j + 1):
            sc = logits[k - i].copy()
            if P is not None and prev1 >= 0:
                sc += beta * np.log(P[:, prev1] + 1e-6)
            if prev1 >= 0:
                sc[prev1] += jack_bonus
            for u in used:
                sc[u] = -1e9
            c = int(sc.argmax()); used.add(c); pred[k] = c
            p1 = prev1
            if p1 < 0:
                streak = 1
            elif c == p1:
                streak += 1
            else:
                streak = 1
            prev2 = prev1; prev1 = c
            if c < 2:
                lastL = c
            else:
                lastR = c
        i = j + 1

    true = np.array([q[1] for q in notes])
    out = [{"t": float(notes[k][0]), "lane": int(pred[k]), "type": int(notes[k][2]),
            "dur": float(notes[k][3]), "true": int(true[k])} for k in range(n)]
    json.dump({"fid": os.path.splitext(os.path.basename(osu))[0], "acc_ar": float((pred == true).mean()),
               "notes": out}, open(out_path, "w"), ensure_ascii=False)
    print(out_path, "notes", n, "AR acc", round(float((pred == true).mean()), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.0, help="换列先验权重；0=不启用")
    ap.add_argument("--trans", type=str, default=None, help="mania npz，用于估计 4x4 换列先验")
    ap.add_argument("--jack-bonus", type=float, default=0.0, help="同列 jack 的 logit 加成")
    a = ap.parse_args()
    gen(a.osu, a.ckpt, a.out, a.beta, a.trans, a.jack_bonus)
