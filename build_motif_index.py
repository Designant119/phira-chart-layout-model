#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phira motif 索引：每条判定线切 12 音 / stride 4，节奏向量与 lane pattern 分开存。
用法：python3 build_motif_index.py --roots <dir1,dir2> --out <dir>
"""
import os, glob, json, argparse
import numpy as np
from collections import defaultdict

L, STRIDE = 12, 4
X_SCALE = 675.0


def line_motifs(notes, lid, fid):
    ns = [n for n in notes if int(n.get("line_id", 0)) == lid and int(n.get("type", 1)) in (1, 2, 3, 4)]
    ns.sort(key=lambda n: n["t"])
    n = len(ns)
    if n < L:
        return []
    t = np.array([q["t"] for q in ns], dtype=np.float64)
    x = np.clip(np.array([q["x"] for q in ns], dtype=np.float32) / X_SCALE, -1.2, 1.2)
    ty = np.array([int(q.get("type", 1)) for q in ns], dtype=np.int64)
    dur = np.array([min(float(q.get("dur", 0)), 2.0) for q in ns], dtype=np.float32)
    dt = np.zeros(n, np.float32); dt[1:] = np.clip(np.log1p(t[1:] - t[:-1]) / 3.0, 0, 1)
    chord = np.zeros(n, np.float32); i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(t[j + 1] - t[i]) < 1e-4: j += 1
        chord[i:j + 1] = min(j - i + 1, 4) / 4.0; i = j + 1
    dens = np.zeros(n, np.float32)
    for i in range(n):
        lo = np.searchsorted(t, t[i] - 0.5, "left")
        hi = np.searchsorted(t, t[i] + 0.5, "right")
        dens[i] = min((hi - lo) / 20.0, 1.0)
    out = []
    for s in range(0, n - L + 1, STRIDE):
        e = s + L
        # 节奏 query：dt / 同刻和弦 / 密度 / 类型 one-hot / Hold 时长
        q = np.concatenate([
            dt[s:e][:, None], chord[s:e][:, None], dens[s:e][:, None],
            np.eye(4, dtype=np.float32)[ty[s:e] - 1],
            np.where(ty[s:e] == 2, np.clip(dur[s:e] / 2.0, 0, 1), 0)[:, None],
        ], axis=1).astype(np.float32)
        xa = x[s:e]
        out.append((q.reshape(-1), xa, (xa - xa[0]).astype(np.float32), ty[s:e], t[s:e], fid, lid, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True, help="逗号分隔，每个含 parsed/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    roots = [r.strip() for r in a.roots.split(",") if r.strip()]
    fids = set()
    for r in roots:
        fids |= {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(r, "parsed", "*.json"))}
    fids = sorted(fids)
    all_m = []
    chart_count = 0
    for fi, fid in enumerate(fids):
        p = None
        for r in roots:
            q = os.path.join(r, "parsed", f"{fid}.json")
            if os.path.exists(q): p = q; break
        if not p: continue
        try:
            notes = json.load(open(p)).get("notes", [])
        except Exception:
            continue
        if not notes: continue
        chart_count += 1
        for lid in sorted({int(n.get("line_id", 0)) for n in notes}):
            all_m += line_motifs(notes, lid, fid)
        if (fi + 1) % 30 == 0:
            print(f"{fi+1}/{len(fids)} charts, motifs {len(all_m)}", flush=True)
    if not all_m:
        raise SystemExit("没有 motif")
    Q = np.stack([m[0] for m in all_m])
    X = np.stack([m[1] for m in all_m])
    R = np.stack([m[2] for m in all_m])
    T = np.stack([m[3] for m in all_m])
    meta = {"fid": [m[5] for m in all_m], "line": np.array([m[6] for m in all_m]),
            "start": np.array([m[7] for m in all_m]), "t0": np.array([m[4][0] for m in all_m])}
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "motifs.npz"), Q=Q, X=X, R=R, T=T, **meta)
    stats = {
        "charts": chart_count, "motifs": len(all_m), "L": L, "stride": STRIDE,
        "q_dim": Q.shape[1],
        "jack_rate": float(np.mean(np.abs(np.diff(X, axis=1)) < 1e-4)),
        "hold_rate": float(np.mean(T == 2)), "drag_rate": float(np.mean(T == 3)),
        "flick_rate": float(np.mean(T == 4)),
    }
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print("saved", a.out, stats, flush=True)


if __name__ == "__main__":
    main()
