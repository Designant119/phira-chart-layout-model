#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9维(+chord_x+level_norm) 6thPecJam 微调数据构建。
严格复用 train_data_ft6th/split.json 的 train(62)/val(16) 划分（不重新随机），
逐 fid 构建 9 维窗口，输出 train_data_ft6th9/{train,val}.npz + split.json 副本。
"""
import json, os, glob
import numpy as np
# 复用全量构建的核心逻辑（不改 build_chord_npz.py 行为）
from build_chord_npz import build_song, BASE, W, LEVEL_DIV

OUT = os.path.join(BASE, "train_data_ft6th9")
SPLIT_SRC = os.path.join(BASE, "train_data_ft6th", "split.json")


def main():
    lvl_cache = json.load(open(os.path.join(BASE, "level_cache.json")))
    split = json.load(open(SPLIT_SRC))
    os.makedirs(OUT, exist_ok=True)
    json.dump(split, open(os.path.join(OUT, "split.json"), "w"), ensure_ascii=False, indent=1)

    for sp in ("train", "val"):
        fids = split[sp]  # 保持原顺序与重复项，复刻旧划分
        songs = []
        n_no_bpm, n_no_notes, n_no_level = 0, 0, 0
        for fid in fids:
            bpm_path = os.path.join(BASE, "bpm", f"{fid}.json")
            if not os.path.exists(bpm_path):
                n_no_bpm += 1
                continue
            bpm = json.load(open(bpm_path))["segs"]
            try:
                d = json.load(open(os.path.join(BASE, "parsed", f"{fid}.json")))
            except Exception:
                n_no_notes += 1
                continue
            notes = d.get("notes", [])
            if not notes:
                n_no_notes += 1
                continue
            lv = lvl_cache.get(fid)
            if lv is None:
                level_norm = 0.0
                n_no_level += 1
            else:
                level_norm = float(lv) / LEVEL_DIV
            w = build_song(fid, notes, bpm, level_norm)
            if w:
                songs.append(w)
        total_w = sum(len(w) for w in songs)
        X = np.zeros((total_w, W, 9), dtype=np.float32)
        y = np.zeros((total_w, W), dtype=np.float32)
        idx = 0
        for wins in songs:
            for wx, wy in wins:
                X[idx] = wx; y[idx] = wy; idx += 1
        np.savez(os.path.join(OUT, f"{sp}.npz"), X=X, y=y)
        print(f"{sp}: 条目{len(fids)} 窗口{total_w} | no_bpm{n_no_bpm} no_notes{n_no_notes} no_level{n_no_level} "
              f"-> X{X.shape}")
    # 校验 val 含 15710/15372
    v = split["val"]
    print("15710 in val:", "15710" in v, "| 15372 in val:", "15372" in v)


if __name__ == "__main__":
    main()
