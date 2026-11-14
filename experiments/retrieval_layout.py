#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索式排键基线：用训练谱的逐音符特征建 KDTree，测试谱每个音符查 k 近邻，
x 取近邻真实 x 的加权平均（两遍：第二遍用第一遍预测算 chord_x）。"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree

REPO = "/home/user/Doubao/chats/38453392661469954/phira-chart-layout-model"
RUN = "/home/user/Doubao/chats/38453392661469954/_run_real_traj"
BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
sys.path.insert(0, REPO)
from traj_common import (line_features, chord_x_map, KEEP_TYPES, LEVEL_NORM_DEFAULT,
                         X_SCALE, X_CLIP, W, STRIDE)

DIMS = ["dt", "phase", "ty", "durs", "chord_x", "level"]


def feat_matrix(feat):
    n = len(feat["dt"])
    cols = []
    for k in DIMS:
        v = np.asarray(feat[k], dtype=np.float32)
        cols.append(v if v.ndim == 1 else np.full(n, float(v), dtype=np.float32))
    return np.stack(cols, axis=1)
K = 10


def load_line(fid):
    d = json.load(open(os.path.join(BASE, "parsed", f"{fid}.json")))
    notes = [n for n in d["notes"] if int(n.get("type", 1)) in KEEP_TYPES]
    bpm = json.load(open(os.path.join(BASE, "bpm", f"{fid}.json")))
    return notes, bpm["segs"]


def lvl_of(fid):
    try:
        return float(json.load(open(os.path.join(BASE, "level_cache.json")).get(fid, LEVEL_NORM_DEFAULT)))
    except Exception:
        return LEVEL_NORM_DEFAULT


def build_bank(fids):
    xs, ys, fids_bank = [], [], []
    for fid in fids:
        notes, bpm = load_line(fid)
        if not notes:
            continue
        feat = line_features(notes, bpm, lvl_of(fid))
        m = feat_matrix(feat)
        xs.append(m)
        ys.append(feat["xnorm"])
        fids_bank += [fid] * len(m)
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    tree = cKDTree((X - mu) / sd)
    return tree, mu, sd, y, len(y)


def predict(fid, tree, mu, sd, yb, k=K, pass1=None):
    notes, bpm = load_line(fid)
    cmap = {}
    if pass1 is not None:
        cmap = chord_x_map(pass1)
    feat = line_features(notes, bpm, lvl_of(fid), cmap)
    Q = feat_matrix(feat)
    d, idx = tree.query((Q - mu) / sd, k=k)
    w = 1.0 / (d + 1e-3)
    w /= w.sum(1, keepdims=True)
    x = np.clip((w * yb[idx]).sum(1) * X_SCALE, -X_CLIP, X_CLIP)
    out = []
    for i, n in enumerate(notes):
        out.append({"t": float(n["t"]), "type": int(n["type"]), "x": float(x[i]),
                    "line_id": int(n.get("line_id", -1)), "dur": float(n.get("dur", 0.0) or 0.0),
                    "src": "retrieval"})
    return {"fid": fid, "mode": "retrieval", "n_notes": len(out), "n_model": len(out),
            "n_fallback": 0, "notes": out}


if __name__ == "__main__":
    split = json.load(open(os.path.join(RUN, "npz_prevx", "split.json")))
    train_fids = split["train"]
    print("bank 构建中:", len(train_fids), "谱", flush=True)
    tree, mu, sd, y_bank, n = build_bank(train_fids)
    print("bank", n, "音符", flush=True)
    tag = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    outdir = f"/tmp/pred_{tag}"
    os.makedirs(outdir, exist_ok=True)
    for i, fid in enumerate(split["test"]):
        p1 = predict(fid, tree, mu, sd, y_bank, pass1=None)
        p2 = predict(fid, tree, mu, sd, y_bank, pass1=p1["notes"])
        json.dump(p2, open(os.path.join(outdir, f"{fid}.json"), "w"))
        if (i + 1) % 6 == 0:
            print(i + 1, flush=True)
