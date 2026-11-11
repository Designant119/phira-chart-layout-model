#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""带双押上下文(chord_x)+难度(level_norm)的排键数据构建。
特征从 8 维 → 9 维：dt, phase, density, pos, ty, durs, prev_x, chord_x, level_norm
chord_x = 同刻(±0.03s)其他线音符 x 均值，归一化 x/675 clip；无双押=0。
level_norm = 谱面难度 level/25.0（0~1），该歌所有窗口的常量第9维；无 level 歌=0 兜底。
标签不变（线内 Δx）。按歌划分 train/val/test。
用法：python3 build_chord_npz.py --out <dir> [--fids <json或txt>]
"""
import json, os, sys, glob, argparse
import numpy as np
from collections import defaultdict

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
W, STRIDE = 128, 64
X_SCALE, X_CLIP = 675.0, 1.2
LEVEL_DIV = 25.0
FEAT_DIM = 9


def beat_at(t, segs):
    """音符 t(秒) -> beat。segs=[[start_beat,bpm],...]（首段从0补起）"""
    n = len(segs)
    if n == 0:
        return 0.0
    start_secs = [0.0] * n
    for i in range(1, n):
        b0, bpm0 = segs[i - 1]
        b1 = segs[i][0]
        start_secs[i] = start_secs[i - 1] + (b1 - b0) * 60.0 / bpm0
    b_start, bpm = segs[0]
    idx = n - 1
    for i in range(n):
        if start_secs[i] > t:
            idx = i - 1
            break
    b_start, bpm = segs[idx]
    return b_start + (t - start_secs[idx]) * bpm / 60.0


def build_song(fid, notes, bpm, level_norm=0.0):
    """按线构建窗口，特征 9 维（含 chord_x 第8、level_norm 第9）。返回窗口列表 [(X128x9, y128)]"""
    # 全部音符按 t 建索引：同刻(±0.03s) 其他线音符 -> chord_x
    all_by_t = defaultdict(list)
    for n in notes:
        all_by_t[round(n["t"], 3)].append(n)
    chord_x_of = {}
    for t, ns in all_by_t.items():
        xs = [n["x"] for n in ns]
        if len(xs) >= 2:  # 双押/多押时刻
            m = float(np.mean(xs))
            for n in ns:
                chord_x_of[id(n)] = m

    lines = {}
    for n in notes:
        lid = n.get("line_id", 0)
        lines.setdefault(lid, []).append(n)

    windows = []
    for lid, lnotes in lines.items():
        if len(lnotes) < 4:
            continue
        lnotes = sorted(lnotes, key=lambda n: n["t"])
        ts = np.array([n["t"] for n in lnotes], dtype=np.float64)
        xs = np.array([n["x"] for n in lnotes], dtype=np.float32)
        tys = np.array([n.get("type", 1) for n in lnotes], dtype=np.float32)
        durs = np.array([min(n.get("dur", 0.0), 2.0) for n in lnotes], dtype=np.float32)

        dt = np.zeros(len(lnotes), dtype=np.float32)
        dt[1:] = np.clip(np.log1p(ts[1:] - ts[:-1]) / 3.0, 0, 1)

        if bpm is not None:
            beats = np.array([beat_at(float(t), bpm) for t in ts])
            phase = ((np.floor(beats * 4.0).astype(np.int64) % 16) / 15.0).astype(np.float32)
        else:
            phase = np.zeros(len(lnotes), dtype=np.float32)

        ty = (tys - 1.0) / 3.0
        pos = np.arange(len(lnotes), dtype=np.float32) % W / float(W - 1)

        xnorm = np.clip(xs / X_SCALE, -X_CLIP, X_CLIP).astype(np.float32)
        dx = np.zeros(len(lnotes), dtype=np.float32)
        dx[0] = xnorm[0]
        dx[1:] = np.clip(xnorm[1:] - xnorm[:-1], -X_CLIP, X_CLIP)
        prev_x = np.zeros(len(lnotes), dtype=np.float32)
        prev_x[1:] = xnorm[:-1]

        # chord_x：同刻其他线音符 x 均值（无双押=0）
        chord_x = np.zeros(len(lnotes), dtype=np.float32)
        for i, n in enumerate(lnotes):
            m = chord_x_of.get(id(n))
            if m is not None:
                chord_x[i] = float(np.clip(m / X_SCALE, -X_CLIP, X_CLIP))

        nw = max(0, (len(lnotes) - W) // STRIDE + 1)
        for i in range(nw):
            s = i * STRIDE
            e = s + W
            seg_t = ts[s:e]
            span = seg_t[-1] - seg_t[0]
            density = np.clip(float(W) / max(span, 1e-6) / 20.0, 0, 1)
            X = np.stack([
                dt[s:e], phase[s:e], np.full(W, density, dtype=np.float32),
                pos[s:e], ty[s:e], durs[s:e], prev_x[s:e], chord_x[s:e],
                np.full(W, level_norm, dtype=np.float32),
            ], axis=1)
            windows.append((X.astype(np.float32), dx[s:e].astype(np.float32)))
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fids", type=str, default=None, help="fid 列表文件(json数组或每行一个)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--split", type=str, default="80/10/10", help="train/val/test 百分比")
    args = ap.parse_args()

    if args.fids:
        if args.fids.endswith(".json"):
            obj = json.load(open(args.fids))
            fids = obj.get("fids_parsed") if isinstance(obj, dict) else obj
        else:
            fids = [l.strip() for l in open(args.fids) if l.strip()]
    else:
        fids = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(BASE, "parsed", "*.json"))]
    fids = sorted(set(fids))
    print(f"fid 总数: {len(fids)}")

    # 加载 level_cache {fid: level_float}
    lvl_cache = {}
    lvl_path = os.path.join(BASE, "level_cache.json")
    if os.path.exists(lvl_path):
        lvl_cache = json.load(open(lvl_path))
    print(f"level_cache 载入: {len(lvl_cache)} fid")

    songs = []
    stats = {"no_bpm": 0, "empty": 0, "no_notes": 0, "used": 0, "no_level": 0}
    for fid in fids:
        bpm_path = os.path.join(BASE, "bpm", f"{fid}.json")
        if not os.path.exists(bpm_path):
            stats["no_bpm"] += 1
            continue
        bpm = json.load(open(bpm_path))["segs"]
        try:
            d = json.load(open(os.path.join(BASE, "parsed", f"{fid}.json")))
        except Exception:
            stats["empty"] += 1
            continue
        notes = d.get("notes", [])
        if not notes:
            stats["no_notes"] += 1
            continue
        # 第9维：level_norm = level/25.0；无 level 歌=0 兜底
        lv = lvl_cache.get(fid)
        if lv is None:
            level_norm = 0.0
            stats["no_level"] += 1
        else:
            level_norm = float(lv) / LEVEL_DIV
        w = build_song(fid, notes, bpm, level_norm)
        if not w:
            continue
        songs.append((fid, w))
        stats["used"] += 1
    print("构建统计:", stats)

    rng = np.random.RandomState(123)
    rng.shuffle(songs)
    tr_s, va_s, te_s = [int(x) for x in args.split.split("/")]
    n = len(songs)
    n_tr = int(n * tr_s / 100)
    n_va = int(n * va_s / 100)
    splits = {"train": songs[:n_tr], "val": songs[n_tr:n_tr + n_va], "test": songs[n_tr + n_va:]}
    fid_split = {k: [s[0] for s in v] for k, v in splits.items()}
    os.makedirs(args.out, exist_ok=True)
    json.dump(fid_split, open(os.path.join(args.out, "split.json"), "w"), ensure_ascii=False, indent=1)

    for split, items in splits.items():
        total_w = sum(len(w) for _, w in items)
        X = np.zeros((total_w, W, FEAT_DIM), dtype=np.float32)
        y = np.zeros((total_w, W), dtype=np.float32)
        sid = []
        idx = 0
        for fid, wins in items:
            for wx, wy in wins:
                X[idx] = wx
                y[idx] = wy
                idx += 1
            sid.extend([fid] * len(wins))
        np.savez(os.path.join(args.out, f"{split}.npz"), X=X, y=y, song_ids=np.array(sid))
        print(f"{split}: 歌{len(items)} 窗口{total_w} → {os.path.join(args.out, split)}.npz")


if __name__ == "__main__":
    main()
