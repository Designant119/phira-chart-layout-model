#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建排键训练集：
窗口 128 音符、stride 32；特征 6 维；标签 x/675 clip[-2,2]；按歌 8:1:1 划分。
产物：train_data/train.npz val.npz test.npz + split.json + stats.json
"""
import json, os, glob, math
import numpy as np

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
OUT = os.path.join(BASE, "train_data")
os.makedirs(OUT, exist_ok=True)

W = 128          # 窗口
STRIDE = 32      # 滑动步长
SEED = 42
FEAT_NAMES = ["dt", "grid_phase", "density", "pos", "type", "dur"]
# 标签归一化
X_SCALE = 675.0
X_CLIP = 2.0


def beat_at(t, segs):
    """音符 t(秒) -> beat。segs=[[start_beat,bpm],...]（已排序、首段从0或补0）
    beat→秒 逐段积分：sec(b) = ∫ db / (bpm/60)"""
    if t <= 0:
        return 0.0
    # 构造分段秒边界
    n = len(segs)
    start_secs = [0.0] * n
    for i in range(1, n):
        b0, bpm0 = segs[i - 1]
        b1 = segs[i][0]
        start_secs[i] = start_secs[i - 1] + (b1 - b0) * 60.0 / bpm0
    # 定位 t 所在段
    idx = n - 1
    for i in range(n - 1, -1, -1):
        if t >= start_secs[i]:
            idx = i
            break
    b_start, bpm = segs[idx]
    return b_start + (t - start_secs[idx]) * bpm / 60.0


def build_song(fid, notes, bpm):
    """音符序列 -> (X 窗口序列, y)"""
    if len(notes) < 4:
        return None
    notes = sorted(notes, key=lambda n: n["t"])
    ts = np.array([n["t"] for n in notes], dtype=np.float64)
    xs = np.array([n["x"] for n in notes], dtype=np.float32)
    tys = np.array([n.get("type", 1) for n in notes], dtype=np.float32)
    durs = np.array([min(n.get("dur", 0.0), 2.0) for n in notes], dtype=np.float32)

    # Δt（log1p 压缩）
    dt = np.zeros(len(notes), dtype=np.float32)
    dt[1:] = np.clip(np.log1p(ts[1:] - ts[:-1]) / 3.0, 0, 1)

    # 16 分网格相位（每小节 16 格，0~15 → /15）
    if bpm is not None:
        beats = np.array([beat_at(float(t), bpm) for t in ts])
        phase = ((np.floor(beats * 4.0).astype(np.int64) % 16) / 15.0).astype(np.float32)
    else:
        phase = np.zeros(len(notes), dtype=np.float32)

    # type 归一化
    ty = (tys - 1.0) / 3.0

    # 窗口级密度：窗口时间跨度
    pos = np.arange(len(notes), dtype=np.float32) % W / float(W - 1)

    # 标签
    y = np.clip(xs / X_SCALE, -X_CLIP, X_CLIP).astype(np.float32)

    windows = []
    nw = max(0, (len(notes) - W) // STRIDE + 1)
    for i in range(nw):
        s = i * STRIDE
        e = s + W
        seg_t = ts[s:e]
        span = seg_t[-1] - seg_t[0]
        density = np.clip(float(W) / max(span, 1e-6) / 20.0, 0, 1)  # 20音符/秒 → 1
        X = np.stack([
            dt[s:e],
            phase[s:e],
            np.full(W, density, dtype=np.float32),
            pos[s:e],
            ty[s:e],
            durs[s:e],
        ], axis=1)
        windows.append((X, y[s:e].copy()))
    return windows


def main():
    rng = np.random.RandomState(SEED)
    parsed_files = sorted(glob.glob(os.path.join(BASE, "parsed", "*.json")))
    songs = []  # (fid, X_list, y_list)
    stats = {"no_bpm": 0, "empty": 0, "too_short": 0, "used": 0}
    for pf in parsed_files:
        fid = os.path.basename(pf)[:-5]
        bpm_path = os.path.join(BASE, "bpm", f"{fid}.json")
        if not os.path.exists(bpm_path):
            stats["no_bpm"] += 1
            continue
        bpm = json.load(open(bpm_path))["segs"]
        try:
            d = json.load(open(pf))
            notes = d.get("notes", [])
        except Exception:
            stats["empty"] += 1
            continue
        if not notes:
            stats["empty"] += 1
            continue
        w = build_song(fid, notes, bpm)
        if w is None:
            stats["too_short"] += 1
            continue
        songs.append((fid, w))
        stats["used"] += 1
    print("歌曲筛选:", stats, flush=True)

    # 按歌划分 8:1:1
    rng.shuffle(songs)
    n = len(songs)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    splits = {
        "train": songs[:n_tr],
        "val": songs[n_tr:n_tr + n_va],
        "test": songs[n_tr + n_va:],
    }
    fid_split = {k: [s[0] for s in v] for k, v in splits.items()}
    json.dump(fid_split, open(os.path.join(OUT, "split.json"), "w"), ensure_ascii=False, indent=1)

    for split, items in splits.items():
        # 预分配
        total_w = sum(len(w) for _, w in items)
        print(f"{split}: 歌 {len(items)} 窗口 {total_w}", flush=True)
        X = np.zeros((total_w, W, 6), dtype=np.float32)
        y = np.zeros((total_w, W), dtype=np.float32)
        idx = 0
        for fid, w in items:
            for wx, wy in w:
                X[idx] = wx
                y[idx] = wy
                idx += 1
        np.savez_compressed(
            os.path.join(OUT, f"{split}.npz"),
            X=X, y=y,
            song_ids=np.array([s[0] for s in items]),
        )
        print(f"{split} 已写盘: {X.shape} {X.nbytes/1e9:.2f}GB", flush=True)

    stats["windows"] = {k: sum(len(w) for _, w in v) for k, v in splits.items()}
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print("全部完成", stats, flush=True)


if __name__ == "__main__":
    main()
