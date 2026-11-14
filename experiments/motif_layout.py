#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pattern 检索式排键：从训练谱抽 12 音 motif（节奏上下文 + 真实 Δx 序列），
测试谱逐段查近邻，用两只手的移动代价在候选里选最省力的一段贴上去。"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree

REPO = "/home/user/Doubao/chats/38453392661469954/phira-chart-layout-model"
RUN = "/home/user/Doubao/chats/38453392661469954/_run_real_traj"
BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
sys.path.insert(0, REPO)
from traj_common import KEEP_TYPES, LEVEL_NORM_DEFAULT, X_SCALE, X_CLIP

L = 24          # motif 长度（音符数）
STRIDE = 2
K = 12          # 候选数


def load_line(fid):
    d = json.load(open(os.path.join(BASE, "parsed", f"{fid}.json")))
    notes = [n for n in d["notes"] if int(n.get("type", 1)) in KEEP_TYPES]
    notes.sort(key=lambda n: (n["t"], n.get("line_id", -1)))
    return notes


def lvl_of(fid):
    try:
        return float(json.load(open(os.path.join(BASE, "level_cache.json")).get(fid, LEVEL_NORM_DEFAULT)))
    except Exception:
        return LEVEL_NORM_DEFAULT


def note_feats(notes):
    """每个音符的节奏上下文特征（不用任何 x）：dt/phase/ty/dur/chord/density。"""
    n = len(notes)
    ts = np.array([q["t"] for q in notes], dtype=np.float64)
    ty = np.array([q.get("type", 1) for q in notes], dtype=np.float32)
    dur = np.array([min(q.get("dur", 0.0) or 0.0, 2.0) for q in notes], dtype=np.float32)
    dt = np.zeros(n, np.float32)
    if n > 1:
        dt[1:] = np.clip(np.log1p(ts[1:] - ts[:-1]) / 3.0, 0, 1)
    # 同刻音数
    from collections import Counter
    c = Counter(round(t, 3) for t in ts)
    chord = np.array([min(c[round(t, 3)], 4) / 4.0 for t in ts], dtype=np.float32)
    # 局部密度：±0.5s 内音符数 / 20
    dens = np.zeros(n, np.float32)
    j0 = 0
    for i in range(n):
        while ts[i] - ts[j0] > 0.5:
            j0 += 1
        j1 = i
        while j1 + 1 < n and ts[j1 + 1] - ts[i] <= 0.5:
            j1 += 1
        dens[i] = min((j1 - j0 + 1) / 20.0, 1.0)
    # phase 用 bpm 段
    return np.stack([dt, chord, ty, dur, dens], axis=1), ts


def motif_matrix(feat, i):
    """取 [i, i+L) 的上下文，尾部不足用最后一条重复。"""
    idx = np.clip(np.arange(i, i + L), 0, len(feat) - 1)
    return feat[idx].reshape(-1)


def build_bank(fids):
    ctx, dx, x0, src = [], [], [], []
    for fid in fids:
        notes = load_line(fid)
        if len(notes) < L + 1:
            continue
        feat, ts = note_feats(notes)
        xs = np.array([q["x"] for q in notes], dtype=np.float32)
        for i in range(0, len(notes) - L, STRIDE):
            ctx.append(motif_matrix(feat, i))
            seg = xs[i:i + L]
            dx.append(np.diff(seg) / X_SCALE)
            x0.append(seg[0] / X_SCALE)
            src.append(fid)
    C = np.stack(ctx)
    D = np.stack(dx)
    X0 = np.array(x0, np.float32)
    mu, sd = C.mean(0), C.std(0) + 1e-6
    tree = cKDTree((C - mu) / sd)
    return tree, mu, sd, D, X0, len(C)


def hand_cost(path_norm):
    """两只手从 ±0.55 出发，逐音就近分配，返回总移动距离（归一化单位）。"""
    lp, rp = -0.55, 0.55
    cost = 0.0
    for x in path_norm:
        if abs(x - lp) <= abs(x - rp):
            cost += abs(x - lp); lp = x
        else:
            cost += abs(x - rp); rp = x
    return cost


def predict(fid, tree, mu, sd, bank_dx, bank_x0, pass1=None, k=K):
    notes = load_line(fid)
    feat, ts = note_feats(notes)
    n = len(notes)
    pred = np.zeros(n, np.float32)
    cur = 0.0
    i = 0
    while i < n:
        q = motif_matrix(feat, i)
        d, idx = tree.query((q - mu) / sd, k=k)
        best, best_score = None, 1e18
        for dd, j in zip(d, idx):
            # 候选自己的绝对轨迹（真实 x 形状），平移接上当前位置，限制越界
            abs_path = (bank_x0[j] + np.concatenate([[0.0], np.cumsum(bank_dx[j])])) * X_SCALE
            path = abs_path[:n - i]
            shift = cur - path[0]
            path = path + shift
            path[0] = cur
            bound = max(0.0, float(np.abs(path).max()) / (X_CLIP * X_SCALE) - 1.0)
            cost = hand_cost(np.clip(path / X_SCALE, -X_CLIP, X_CLIP))
            score = 0.45 * dd + 2.0 * bound + 0.10 * cost + 0.10 * abs(shift) / X_SCALE
            if score < best_score:
                best_score, best = score, path
        m = len(best)
        pred[i:i + m] = np.clip(best, -X_CLIP * X_SCALE, X_CLIP * X_SCALE)
        cur = pred[i + m - 1]
        i += m
    out = [{"t": float(q["t"]), "type": int(q["type"]), "x": float(pred[j]),
            "line_id": int(q.get("line_id", -1)), "dur": float(q.get("dur", 0.0) or 0.0),
            "src": "motif"} for j, q in enumerate(notes)]
    return {"fid": fid, "mode": "motif", "n_notes": n, "n_model": n, "n_fallback": 0, "notes": out}


if __name__ == "__main__":
    split = json.load(open(os.path.join(RUN, "npz_prevx", "split.json")))
    print("bank...", flush=True)
    tree, mu, sd, bdx, bx0, nb = build_bank(split["train"])
    print("motifs", nb, flush=True)
    tag = sys.argv[1] if len(sys.argv) > 1 else "motif"
    outdir = f"/tmp/pred_{tag}"
    os.makedirs(outdir, exist_ok=True)
    for i, fid in enumerate(split["test"]):
        p = predict(fid, tree, mu, sd, bdx, bx0)
        json.dump(p, open(os.path.join(outdir, f"{fid}.json"), "w"))
        if (i + 1) % 6 == 0:
            print(i + 1, flush=True)
