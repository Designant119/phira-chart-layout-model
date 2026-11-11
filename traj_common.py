#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排键轨迹主线共享模块：口径常量、BPM/拍换算、单线特征、窗口切片、双押间距后处理。

build_traj_npz（训练数据）与 infer_traj（推理）必须共用本模块的特征代码，
杜绝 train/serve 偏移（dt/pos/density/类型过滤/难度兜底口径只允许一处定义）。
"""
import os
from collections import defaultdict
import numpy as np

# --- 口径常量（训练/推理唯一出处，勿在别处复制）---
W, STRIDE = 128, 64
X_SCALE, X_CLIP = 675.0, 1.2
LEVEL_DIV = 25.0
FEAT_DIM = 8
KEEP_TYPES = (1, 2)          # 只排 Tap/Hold
LEVEL_NORM_DEFAULT = 0.0     # 缺难度兜底：训练=推理统一（改这里必须重建 npz 并重训）


def seg_start_secs(segs):
    """BPM 段 [[start_beat, bpm], ...] -> 各段起始秒（累积）。"""
    n = len(segs)
    start_secs = [0.0] * n
    for i in range(1, n):
        b0, bpm0 = segs[i - 1]
        b1 = segs[i][0]
        start_secs[i] = start_secs[i - 1] + (b1 - b0) * 60.0 / bpm0
    return start_secs


def beats_array(ts, segs):
    """时间(秒)数组 -> beat 数组。start_secs 只算一次，O(len·log n)。
    t<0 锁定第 0 段 BPM 反向外推（前奏倒数 phase 连续；旧版此处会因负索引
    取到最后一段，属 bug，已修——含负时间谱需重建 npz 才与旧数据可比）。"""
    ts = np.asarray(ts, dtype=np.float64)
    if len(segs) == 0:
        return np.zeros(len(ts))
    ss = np.array(seg_start_secs(segs))
    idx = np.clip(np.searchsorted(ss, ts, side="right") - 1, 0, len(segs) - 1)
    b_start = np.array([s[0] for s in segs], dtype=np.float64)[idx]
    bpm = np.array([s[1] for s in segs], dtype=np.float64)[idx]
    return b_start + (ts - ss[idx]) * bpm / 60.0


def chord_x_map(notes):
    """同刻（round(t,3) 且 >=2 个）音符 x 均值 -> {t3: xnorm}。
    训练传谱师真值音符，推理传第一遍预测音符；调用方负责先过滤成 KEEP_TYPES。"""
    by_t = defaultdict(list)
    for n in notes:
        by_t[round(n["t"], 3)].append(float(n["x"]))
    out = {}
    for t3, xs in by_t.items():
        if len(xs) >= 2:
            out[t3] = float(np.clip(np.mean(xs) / X_SCALE, -X_CLIP, X_CLIP))
    return out


def line_features(lnotes, bpm, level_norm, chord_map=None):
    """整条线（已按 t 排序）-> 每音符特征 dict。按整线计算，窗口切片交给 make_window，
    保证 dt（窗口首音=真实间隔）与 pos（线内索引 %W 回卷）训练/推理同口径。
    返回 keys: ts, dt, phase, pos, ty, durs, chord_x, level, xnorm(标签)。"""
    N = len(lnotes)
    ts = np.array([nt["t"] for nt in lnotes], dtype=np.float64)
    tys = np.array([nt.get("type", 1) for nt in lnotes], dtype=np.float32)
    durs = np.array([min(nt.get("dur", 0.0), 2.0) for nt in lnotes], dtype=np.float32)

    dt = np.zeros(N, dtype=np.float32)
    if N > 1:
        dt[1:] = np.clip(np.log1p(ts[1:] - ts[:-1]) / 3.0, 0, 1)

    if bpm:
        beats = beats_array(ts, bpm)
        phase = ((np.floor(beats * 4.0).astype(np.int64) % 16) / 15.0).astype(np.float32)
    else:
        phase = np.zeros(N, dtype=np.float32)

    pos = (np.arange(N, dtype=np.float32) % W) / float(W - 1)
    ty = (tys - 1.0) / 3.0

    chord_x = np.zeros(N, dtype=np.float32)
    if chord_map:
        for i, nt in enumerate(lnotes):
            m = chord_map.get(round(nt["t"], 3))
            if m is not None:
                chord_x[i] = m

    xs = np.array([nt["x"] for nt in lnotes], dtype=np.float32)
    return {
        "ts": ts, "dt": dt, "phase": phase, "pos": pos, "ty": ty,
        "durs": durs, "chord_x": chord_x, "level": np.float32(level_norm),
        "xnorm": np.clip(xs / X_SCALE, -X_CLIP, X_CLIP).astype(np.float32),
    }


def window_starts(n, w=W, stride=STRIDE):
    """推理滑窗 [(s, e)]，保证覆盖 0..n-1 每个音符（尾部不足一个 stride 时补尾窗）。
    注意：训练数据构建仍用对齐全窗（build_traj_npz），两者刻意不同——训练窗口
    必须与历史 npz 同口径保证 ckpt 可比，推理则要求每个音符都有预测。"""
    if n <= 0:
        return []
    if n <= w:
        return [(0, n)]
    starts = list(range(0, n - w + 1, stride))
    if starts[-1] + w < n:
        starts.append(n - w)
    return [(s, min(s + w, n)) for s in starts]


def make_window(feat, s, e, w=W):
    """整线特征切 [s, e) 窗口 -> (X (w,FEAT_DIM) 不足补零, vlen)。
    要求 e-s <= w（window_starts 保证）；density = vlen/span/20，
    全窗 vlen==W 时与训练公式 W/span/20 完全一致。"""
    vlen = e - s
    assert 0 < vlen <= w, f"窗口长度 {vlen} 越界，须在 (0, {w}]（检查 window_starts 调用）"
    span = max(float(feat["ts"][e - 1] - feat["ts"][s]), 1e-6)
    density = np.clip(float(vlen) / span / 20.0, 0, 1)
    seg = np.stack([
        feat["dt"][s:e], feat["phase"][s:e],
        np.full(vlen, density, dtype=np.float32),
        feat["pos"][s:e], feat["ty"][s:e], feat["durs"][s:e],
        feat["chord_x"][s:e],
        np.full(vlen, feat["level"], dtype=np.float32),
    ], axis=1).astype(np.float32)
    if vlen < w:
        seg = np.concatenate([seg, np.zeros((w - vlen, FEAT_DIM), dtype=np.float32)], axis=0)
    return seg, vlen


def enforce_chord_gaps(out_notes, min_gap=200.0, xlim=675.0):
    """跨线同刻多音符：强制相邻间距 >= min_gap（贪心双向推）。
    注意：同刻音符数 > 2*xlim/min_gap+1 时（如 8 音 @200px）物理放不下，
    最终 clip 会静默压回边界、间距略小于 min_gap。"""
    grp = defaultdict(list)
    for n in out_notes:
        grp[round(n['t'], 3)].append(n)
    for t, ns in grp.items():
        if len(ns) < 2:
            continue
        ns.sort(key=lambda n: n['x'])
        xs = [n['x'] for n in ns]
        for i in range(1, len(xs)):          # 从左到右推
            if xs[i] - xs[i - 1] < min_gap:
                xs[i] = xs[i - 1] + min_gap
        if xs[-1] > xlim:                     # 溢出：锚定右边界后从右到左反向推
            # 左推后相邻间距已全部 >= min_gap，若不先把 xs[-1] 锚到 xlim，
            # 反向条件恒为假（死代码），clip 会把溢出组压叠在同一像素上
            xs[-1] = xlim
            for i in range(len(xs) - 2, -1, -1):
                if xs[i + 1] - xs[i] < min_gap:
                    xs[i] = xs[i + 1] - min_gap
        for n, x in zip(ns, xs):
            n['x'] = float(np.clip(x, -xlim, xlim))
    return out_notes


def artifact_path(p):
    """产物路径（npz 目录 / ckpt 目录 / 输出 json）：相对路径相对 cwd，绝对路径原样。
    数据根（parsed/ bpm/ level_cache.json）仍由各脚本 BASE 常量管理，两者别混。"""
    return os.path.abspath(os.path.expanduser(p))
