#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轨迹回归数据构建：标签 = 窗口内音符 x（直接回归），不再是 dx 位移。
输入特征 8 维（去掉 prev_x，推理无需真值）：
dt, phase, density, pos, ty, durs, chord_x, level_norm
标签 y = xnorm (x/675，clip ±1.2)，形状 (N,128)。
特征构建与推理共用 traj_common（line_features/make_window），单处定义防口径漂移。
用法：python3 build_traj_npz.py --out <dir> [--fids <json>]
"""
import json, os, glob, argparse
import numpy as np
from collections import defaultdict

from traj_common import (W, STRIDE, X_CLIP, LEVEL_DIV, FEAT_DIM, KEEP_TYPES,
                         LEVEL_NORM_DEFAULT, chord_x_map, line_features, make_window,
                         artifact_path)

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"


def build_song(notes, bpm, level_norm=0.0):
    """返回窗口列表 [(X128x8, y128)]，y = xnorm。
    只用 Tap(1)/Hold(2)，排除 Drag(3)/Flick(4)；窗口为对齐全窗
    （与历史 npz/ckpt 同口径，保证可比性；推理侧的尾窗全覆盖见 traj_common.window_starts）。"""
    notes = [n for n in notes if int(n.get("type", 1)) in KEEP_TYPES]
    cmap = chord_x_map(notes)

    lines = defaultdict(list)
    for n in notes:
        lines.setdefault(n.get("line_id", 0), []).append(n)

    windows = []
    for lid, lnotes in lines.items():
        if len(lnotes) < 4:
            continue
        lnotes = sorted(lnotes, key=lambda n: n["t"])
        feat = line_features(lnotes, bpm, level_norm, cmap)
        yline = feat["xnorm"]
        for i in range(max(0, (len(lnotes) - W) // STRIDE + 1)):
            s = i * STRIDE
            X, _ = make_window(feat, s, s + W)
            windows.append((X, yline[s:s + W].astype(np.float32)))
    return windows


def find_fid_file(fid, roots, subdir):
    """在多个数据根里找 {root}/{subdir}/{fid}.json，返回路径或 None。"""
    for r in roots:
        p = os.path.join(r, subdir, f"{fid}.json")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fids", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--split", type=str, default="80/10/10")
    ap.add_argument("--roots", type=str, default=BASE,
                    help="逗号分隔的数据根列表（每个含 parsed/ 和 bpm/）")
    ap.add_argument("--level-extra", type=str, default=None)
    ap.add_argument("--aug-blank", type=float, default=0.0,
                    help="空白增强概率：train 窗口以该概率追加 y-=y[0] 的平移副本（val/test 不动）")
    args = ap.parse_args()

    outdir = artifact_path(args.out)
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    if args.fids:
        if args.fids.endswith(".json"):
            obj = json.load(open(args.fids))
            fids = obj.get("fids_parsed") if isinstance(obj, dict) else obj
        else:
            fids = [l.strip() for l in open(args.fids) if l.strip()]
    else:
        fids = set()
        for r in roots:
            fids |= set(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(r, "parsed", "*.json")))
    fids = sorted(set(fids))
    print(f"fid 总数: {len(fids)} | roots: {roots}")

    lvl_cache = {}
    lvl_path = os.path.join(BASE, "level_cache.json")
    if os.path.exists(lvl_path):
        lvl_cache = json.load(open(lvl_path))
    if args.level_extra and os.path.exists(args.level_extra):
        lvl_cache.update(json.load(open(args.level_extra)))
    print(f"level_cache 载入: {len(lvl_cache)} fid")

    songs = []
    stats = {"no_bpm": 0, "empty": 0, "no_notes": 0, "used": 0, "no_level": 0}
    for fid in fids:
        bpm_p = find_fid_file(fid, roots, "bpm")
        if not bpm_p:
            stats["no_bpm"] += 1
            continue
        bpm = json.load(open(bpm_p))["segs"]
        parsed_p = find_fid_file(fid, roots, "parsed")
        if parsed_p is None:
            stats["empty"] += 1
            continue
        try:
            d = json.load(open(parsed_p))
        except Exception:
            stats["empty"] += 1
            continue
        notes = d.get("notes", [])
        if not notes:
            stats["no_notes"] += 1
            continue
        lv = lvl_cache.get(fid)
        if lv is None:
            level_norm = LEVEL_NORM_DEFAULT
            stats["no_level"] += 1
        else:
            level_norm = float(lv) / LEVEL_DIV
        w = build_song(notes, bpm, level_norm)
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

    # --- 空白增强：仅对 train split 追加平移副本（y 从 0 开始），val/test 绝对不动 ---
    n_aug = 0
    if args.aug_blank > 0:
        aug_rng = np.random.RandomState(12345)  # 独立种子，保证可复现且不影响 shuffle
        aug_items = []
        for fid, wins in splits["train"]:
            new_wins = list(wins)
            for wx, wy in wins:
                if aug_rng.rand() < args.aug_blank:
                    wy2 = np.clip(wy - wy[0], -X_CLIP, X_CLIP).astype(np.float32)
                    wx2 = wx.copy()
                    # chord_x 是绝对位置特征，必须同步平移，否则 X/y 语义不一致；
                    # 但 0 是"无和弦"哨兵（line_features 初始化为 0），不能参与平移，
                    # 否则 0-wy[0] 会凭空造出幻影和弦位置，污染增强窗口
                    nz = wx2[:, 6] != 0
                    wx2[nz, 6] = np.clip(wx2[nz, 6] - wy[0], -X_CLIP, X_CLIP).astype(np.float32)
                    new_wins.append((wx2, wy2))
                    n_aug += 1
            aug_items.append((fid, new_wins))
        splits["train"] = aug_items
        print(f"空白增强: p={args.aug_blank} 追加 {n_aug} 个 train 窗口")

    os.makedirs(outdir, exist_ok=True)
    json.dump(fid_split, open(os.path.join(outdir, "split.json"), "w"), ensure_ascii=False, indent=1)

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
        np.savez(os.path.join(outdir, f"{split}.npz"), X=X, y=y, song_ids=np.array(sid))
        print(f"{split}: 歌{len(items)} 窗口{total_w} → {os.path.join(outdir, split)}.npz")


if __name__ == "__main__":
    main()
