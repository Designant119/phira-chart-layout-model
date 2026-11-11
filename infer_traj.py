#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer 轨迹回归推理：滑窗(128,stride64)重叠平均 → 输出每音符 x。
两遍：第一遍 chord_x=0 粗排 → 第二遍用同刻粗排 x 均值精排。
用法：python3 infer_traj.py <fid> [--ckpt ckpt_traj_full] [--pass1] [--out xxx.json]
"""
import json, os, sys, argparse
import numpy as np
import torch
from collections import defaultdict
from build_traj_npz import build_song, beat_at
from train_traj import TrajTransformer
from infer_layout_ckpt import enforce_chord_gaps

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
W, STRIDE = 128, 64
X_SCALE = 675.0


def get_level(fid):
    cache_p = os.path.join(BASE, "level_cache.json")
    if os.path.exists(cache_p):
        c = json.load(open(cache_p))
        if fid in c:
            return float(c[fid])
    return 15.0  # 兜底中等难度


def infer_notes(fid, model, chord_map, level_norm):
    d = json.load(open(os.path.join(BASE, "parsed", f"{fid}.json")))
    bpm = json.load(open(os.path.join(BASE, "bpm", f"{fid}.json")))["segs"]
    # 构建特征（chord_x 用 chord_map 提供，同刻其他音符 x 均值）
    lines = defaultdict(list)
    for n in d["notes"]:
        lines[n["line_id"]].append(n)
    out = []
    with torch.no_grad():
        for lid in sorted(lines.keys()):
            lnotes = sorted(lines[lid], key=lambda n: n["t"])
            if len(lnotes) < 4:
                for n in lnotes:
                    out.append({"t": n["t"], "type": n.get("type", 1), "x": n["x"],
                                "line_id": lid, "dur": n.get("dur", 0.0)})
                continue
            # 用 build_song 拿 X（chord_x 来自原谱真值——推理时换成 chord_map）
            wins = build_song(fid, lnotes, bpm, level_norm)
            if not wins:
                for n in lnotes:
                    out.append({"t": n["t"], "type": n.get("type", 1), "x": n["x"],
                                "line_id": lid, "dur": n.get("dur", 0.0)})
                continue
            # build_song 只对 >W 的线有窗口；对每条线手动滑窗
            pred = {}
            for i in range(max(0, len(lnotes) - W) // STRIDE + 1):
                s = i * STRIDE
                e = min(s + W, len(lnotes))
                seg_notes = lnotes[s:e]
                if len(seg_notes) < 8:
                    break
                # 组装 8 维特征
                ts = np.array([n["t"] for n in seg_notes], dtype=np.float64)
                xs0 = np.array([n["x"] for n in seg_notes], dtype=np.float32)
                tys = np.array([n.get("type", 1) for n in seg_notes], dtype=np.float32)
                durs = np.array([min(n.get("dur", 0.0), 2.0) for n in seg_notes], dtype=np.float32)
                dt = np.zeros(len(seg_notes), dtype=np.float32)
                dt[1:] = np.clip(np.log1p(ts[1:] - ts[:-1]) / 3.0, 0, 1)
                beats = np.array([beat_at(float(t), bpm) for t in ts])
                phase = ((np.floor(beats * 4.0).astype(np.int64) % 16) / 15.0).astype(np.float32)
                ty = (tys - 1.0) / 3.0
                pos = np.arange(len(seg_notes), dtype=np.float32) % W / float(W - 1)
                span = ts[-1] - ts[0]
                density = np.clip(float(len(seg_notes)) / max(span, 1e-6) / 20.0, 0, 1)
                chord_x = np.zeros(len(seg_notes), dtype=np.float32)
                for j, n in enumerate(seg_notes):
                    m = chord_map.get(round(n["t"], 3))
                    if m is not None:
                        chord_x[j] = float(np.clip(m / X_SCALE, -1.2, 1.2))
                X = np.stack([dt, phase, np.full(len(seg_notes), density, np.float32),
                              pos, ty, durs, chord_x, np.full(len(seg_notes), level_norm, np.float32)], axis=1)
                # pad 到 128
                if len(seg_notes) < W:
                    pad = np.zeros((W - len(seg_notes), 8), np.float32)
                    X = np.concatenate([X, pad], axis=0)
                Xt = torch.from_numpy(X.astype(np.float32)).unsqueeze(0)
                yp = model(Xt)[0].numpy()[:len(seg_notes)] * X_SCALE
                for j, n in enumerate(seg_notes):
                    k = (lid, round(n["t"], 3))
                    if k in pred:
                        pred[k] = (pred[k][0] + yp[j], pred[k][1] + 1)
                    else:
                        pred[k] = (yp[j], 1)
            for n in lnotes:
                k = (lid, round(n["t"], 3))
                if k in pred:
                    v, c = pred[k]
                    out.append({"t": n["t"], "type": n.get("type", 1), "x": float(v / c),
                                "line_id": lid, "dur": n.get("dur", 0.0)})
                else:
                    out.append({"t": n["t"], "type": n.get("type", 1), "x": n["x"],
                                "line_id": lid, "dur": n.get("dur", 0.0)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fid", type=str)
    ap.add_argument("--ckpt", type=str, default="ckpt_traj_full")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--pass1", action="store_true", help="只跑第一遍(chord_x=0)")
    args = ap.parse_args()

    ck = torch.load(os.path.join(BASE, args.ckpt, "model.pth"), map_location="cpu")
    feat = ck["model"]["proj.weight"].shape[1]
    model = TrajTransformer(feat=feat)
    model.load_state_dict(ck["model"])
    model.eval()
    lvl = get_level(args.fid) / 25.0

    # 第一遍：chord_x=0
    pass1 = infer_notes(args.fid, model, {}, lvl)
    if args.pass1:
        notes = pass1
    else:
        by_t = defaultdict(list)
        for n in pass1:
            by_t[round(n["t"], 3)].append(n["x"])
        chord_map = {t: float(np.mean(xs)) for t, xs in by_t.items() if len(xs) >= 2}
        notes = infer_notes(args.fid, model, chord_map, lvl)
    notes = [n for n in notes if n.get("type", 1) in {1, 2}]
    notes = enforce_chord_gaps(notes)
    result = {"fid": args.fid, "mode": "traj", "ckpt": args.ckpt,
              "two_pass": not args.pass1, "n_notes": len(notes), "notes": notes}
    if args.out:
        json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"已写入 {args.out}: {len(notes)} 音符")
    else:
        print(json.dumps(result, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
