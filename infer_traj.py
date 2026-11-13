#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer 轨迹回归推理：9 维（含 prev_x）逐音自回归；8 维旧 ckpt 走滑窗平均。
两遍：第一遍 chord_x=0 粗排 → 第二遍用同刻粗排 x 均值精排。
特征与训练共用 traj_common（整线计算后切窗），修复旧版三处偏移：
- 窗口起止：旧版 (len-128)%64 个尾部音符永远不被窗口覆盖、静默沿用原谱 x；
  现在 window_starts 保证全覆盖（尾部不足一个 stride 时补尾窗）。
- 短线 gate：旧版 4~127 音的线被 build_song 空结果整体拦截回退原谱 x；
  现在只拦 <4 音的线，其余走补零+mask 窗口。
- 特征口径：dt/pos 旧版按窗口重算（训练按整线算后切片），Drag/Flick 旧版混进
  序列参与特征；现在与训练逐位一致。
9 维 ckpt 的推理是自回归：每个音只被预测一次（窗口末位），prev_x 用上一个音的预测值，
因此不再做重叠窗口平均（因果模型下平均会把已学的位置信号搅浑）。
每个音符输出带 "src":"model"|"fallback"（fallback=线内<4 音沿用原谱 x），
评估端可据此区分模型真实覆盖，不再被 MAE≈0 的回退音符稀释。
用法：python3 infer_traj.py <fid> [--ckpt ckpt_traj_full] [--pass1] [--out xxx.json]
"""
import json, os, argparse
from collections import defaultdict
import numpy as np
import torch

from traj_common import (W, X_SCALE, X_CLIP, KEEP_TYPES, LEVEL_DIV, LEVEL_NORM_DEFAULT,
                         FEAT_DIM, PREV_X_COL, line_features, window_starts, make_window,
                         enforce_chord_gaps, artifact_path)
from train_traj import TrajTransformer

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"


def get_level_norm(fid):
    """难度归一化；缺失兜底 LEVEL_NORM_DEFAULT（0.0），与训练构建同口径。"""
    cache_p = os.path.join(BASE, "level_cache.json")
    if os.path.exists(cache_p):
        c = json.load(open(cache_p))
        if fid in c:
            return float(c[fid]) / LEVEL_DIV
    return LEVEL_NORM_DEFAULT


def infer_notes(fid, model, chord_map, level_norm, feat_dim=None):
    """一遍推理。chord_map: {t3: xnorm}（第一遍={}，第二遍=第一遍预测的同刻均值）。
    feat_dim>=9（带 prev_x）：逐音自回归，每个音只被预测一次，窗口末尾位取输出；
    feat_dim==8（旧 ckpt）：走旧的重叠窗口平均路径。"""
    fd = FEAT_DIM if feat_dim is None else int(feat_dim)
    d = json.load(open(os.path.join(BASE, "parsed", f"{fid}.json")))
    bpm_p = os.path.join(BASE, "bpm", f"{fid}.json")
    bpm = json.load(open(bpm_p))["segs"] if os.path.exists(bpm_p) else None
    # 先过滤类型再建特征（与训练一致：Drag/Flick 不进序列，也不参与 chord 语义）
    notes = [n for n in d["notes"] if int(n.get("type", 1)) in KEEP_TYPES]
    lines = defaultdict(list)
    for n in notes:
        lines[n.get("line_id", 0)].append(n)

    out = []
    with torch.no_grad():
        for lid in sorted(lines.keys()):
            lnotes = sorted(lines[lid], key=lambda n: n["t"])

            def emit(nt, x, src):
                # 训练标签 clip 到 ±1.2(=±810px)，模型回归值乘 675 后可能越界；
                # enforce_chord_gaps 只裁同刻>=2 的音，单音必须在这里截到合法域 ±675
                out.append({"t": nt["t"], "type": nt.get("type", 1),
                            "x": float(np.clip(x, -X_SCALE, X_SCALE)),
                            "line_id": lid, "dur": nt.get("dur", 0.0), "src": src})

            if len(lnotes) < 4:  # 训练时这类线不进数据集，模型没见过，沿用原谱
                for nt in lnotes:
                    emit(nt, nt["x"], "fallback")
                continue
            feat = line_features(lnotes, bpm, level_norm, chord_map)

            if fd >= 9:
                # --- 自回归：第 i 音用窗口 [i-127, i]，prev_x 填已预测值 ---
                n = len(lnotes)
                preds_norm = np.zeros(n, dtype=np.float32)  # 归一化 ((-1.2,1.2))，与训练 prev_x 同口径
                causal = True
                for i in range(n):
                    s = max(0, i - W + 1)
                    X, vlen = make_window(feat, s, i + 1, feat_dim=fd)
                    prev = np.zeros(vlen, dtype=np.float32)
                    for j in range(s, i + 1):
                        if j > 0:
                            prev[j - s] = preds_norm[j - 1]
                    X[:vlen, PREV_X_COL] = prev
                    Xt = torch.from_numpy(X).unsqueeze(0)
                    pad_mask = None
                    if vlen < W:
                        pad_mask = torch.zeros(1, W, dtype=torch.bool)
                        pad_mask[0, vlen:] = True
                    y = model(Xt, padding_mask=pad_mask, causal=causal)[0, vlen - 1].item()
                    preds_norm[i] = float(np.clip(y, -X_CLIP, X_CLIP))
                for i, nt in enumerate(lnotes):
                    emit(nt, preds_norm[i] * X_SCALE, "model")
                continue

            # --- 旧路径（feat_dim==8，双向模型 + 重叠窗口平均）---
            pred = {}
            for s, e in window_starts(len(lnotes)):
                X, vlen = make_window(feat, s, e, feat_dim=fd)
                Xt = torch.from_numpy(X).unsqueeze(0)
                pad_mask = None
                if vlen < W:  # 补零位屏蔽注意力（训练全窗无 padding，不能让模型看假音符）
                    pad_mask = torch.zeros(1, W, dtype=torch.bool)
                    pad_mask[0, vlen:] = True
                yp = model(Xt, padding_mask=pad_mask)[0].numpy()[:vlen] * X_SCALE
                for j in range(vlen):
                    t3 = round(lnotes[s + j]["t"], 3)
                    if t3 in pred:
                        pred[t3][0] += float(yp[j])
                        pred[t3][1] += 1
                    else:
                        pred[t3] = [float(yp[j]), 1]
            for nt in lnotes:
                p = pred.get(round(nt["t"], 3))
                if p:  # window_starts 已全覆盖，此处 fallback 仅防御
                    emit(nt, p[0] / p[1], "model")
                else:
                    emit(nt, nt["x"], "fallback")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fid", type=str)
    ap.add_argument("--ckpt", type=str, default="ckpt_traj_full",
                    help="ckpt 目录（相对 cwd 或绝对路径）")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--pass1", action="store_true", help="只跑第一遍(chord_x=0)")
    args = ap.parse_args()

    ckpt_dir = artifact_path(args.ckpt)
    ck = torch.load(os.path.join(ckpt_dir, "model.pth"), map_location="cpu")
    feat = ck["model"]["proj.weight"].shape[1]
    model = TrajTransformer(feat=feat)
    model.load_state_dict(ck["model"])
    model.eval()
    lvl = get_level_norm(args.fid)

    # 第一遍：chord_x=0
    pass1 = infer_notes(args.fid, model, {}, lvl, feat_dim=feat)
    if args.pass1:
        notes = pass1
    else:
        # 第二遍 chord_map：第一遍预测里同刻(>=2)音符的 x 均值（已过滤为 Tap/Hold，与训练 chord 同语义）
        by_t = defaultdict(list)
        for n in pass1:
            by_t[round(n["t"], 3)].append(n["x"])
        chord_map = {t: float(np.clip(np.mean(xs) / X_SCALE, -X_CLIP, X_CLIP))
                     for t, xs in by_t.items() if len(xs) >= 2}
        notes = infer_notes(args.fid, model, chord_map, lvl, feat_dim=feat)
    notes = enforce_chord_gaps(notes)
    n_fb = sum(1 for n in notes if n.get("src") == "fallback")
    result = {"fid": args.fid, "mode": "traj", "ckpt": args.ckpt,
              "two_pass": not args.pass1, "n_notes": len(notes),
              "n_model": len(notes) - n_fb, "n_fallback": n_fb, "notes": notes}
    if args.out:
        out_p = artifact_path(args.out)
        json.dump(result, open(out_p, "w"), ensure_ascii=False, indent=1)
        print(f"已写入 {out_p}: {len(notes)} 音符（模型 {len(notes) - n_fb} / fallback {n_fb}）")
    else:
        print(json.dumps(result, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
