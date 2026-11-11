#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排键模型推理：输入一首歌的 fid → 输出每个音符的具体 x 坐标（可直接用于制谱器）。
用法: python3 infer_layout.py <fid> [--seed N] [--out out.json]
输出格式: {"fid":..., "mode":"sample", "notes":[{"t":秒,"type":1,"x":px,"line_id":...,"dur":...}, ...]}
x 范围约 -675~+675（中心0）；制谱器如需 0~1350 坐标，用 x_rpe = x + 675。
"""
import json, os, sys, argparse
import numpy as np
import torch
from train_layout_v4 import GRUCls, BIN_CENTERS, NB, BIN_LO, BIN_HI

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
X_SCALE = 675.0
X_CLIP = 2.0
W = 128
STRIDE = 32


def beat_at(t, segs):
    """t(秒) -> beat（与 build_train_npz_v3 一致）"""
    if t <= 0:
        return 0.0
    n = len(segs)
    start_secs = [0.0] * n
    for i in range(1, n):
        b0, bpm0 = segs[i - 1]
        b1 = segs[i][0]
        start_secs[i] = start_secs[i - 1] + (b1 - b0) * 60.0 / bpm0
    idx = n - 1
    for i in range(n - 1, -1, -1):
        if t >= start_secs[i]:
            idx = i
            break
    b_start, bpm = segs[idx]
    return b_start + (t - start_secs[idx]) * bpm / 60.0


def build_features(lnotes, bpm):
    """单条线音符 -> 特征矩阵 (N,7) + 线首索引。与训练特征完全一致。"""
    ts = np.array([n["t"] for n in lnotes], dtype=np.float64)
    xs = np.array([n["x"] for n in lnotes], dtype=np.float32)
    tys = np.array([n.get("type", 1) for n in lnotes], dtype=np.float32)
    durs = np.array([min(n.get("dur", 0.0), 2.0) for n in lnotes], dtype=np.float32)

    N = len(lnotes)
    dt = np.zeros(N, dtype=np.float32)
    dt[1:] = np.clip(np.log1p(ts[1:] - ts[:-1]) / 3.0, 0, 1)

    if bpm is not None:
        beats = np.array([beat_at(float(t), bpm) for t in ts])
        phase = ((np.floor(beats * 4.0).astype(np.int64) % 16) / 15.0).astype(np.float32)
    else:
        phase = np.zeros(N, dtype=np.float32)

    ty = (tys - 1.0) / 3.0
    pos = np.arange(N, dtype=np.float32) % W / float(W - 1)
    xnorm = np.clip(xs / X_SCALE, -X_CLIP, X_CLIP).astype(np.float32)
    prev_x = np.zeros(N, dtype=np.float32)
    prev_x[1:] = xnorm[:-1]

    span = ts[-1] - ts[0] if N > 1 else 1.0
    density = np.clip(float(N) / max(span, 1e-6) / 20.0, 0, 1)

    X = np.stack([dt, phase, np.full(N, density, np.float32), pos, ty, durs, prev_x], axis=1)
    return X.astype(np.float32)


def infer_line(model, Xw, rng, bound=0.98):
    """一条线特征 (N,7) -> 预测 x 序列（真自回归：逐音符前向 + prev_x 实时更新）。
    修复：旧版整窗口一次前向，窗口内 prev_x 特征全是初始值，模型误以为
    "前一个音符在 0 位置"，导致采样偏向小位移 → 输出大量纵连。
    现在每预测一个音符就把它作为下一个的 prev_x 特征，模型真正"看着前面排"。"""
    N = Xw.shape[0]
    pred_norm = np.zeros(N, dtype=np.float32)
    for i in range(N):
        # 构造以 i 结尾的窗口（长度≤W，与训练窗口语义一致）
        s = max(0, i - (W - 1))
        seg = Xw[s:i + 1].copy()
        # 用已预测的 x 更新窗口内每个音符的 prev_x 特征（第6列）
        for k in range(s, i + 1):
            if k > 0:
                seg[k - s, 6] = pred_norm[k - 1]
        # 窗口内位置特征（第3列）按当前窗口重算（训练时 %W/(W-1)）
        seg[:, 3] = (np.arange(len(seg)) % W) / float(W - 1)
        Xt = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = model(Xt)[0]
            probs = torch.softmax(logits[-1], dim=-1).numpy()  # 只取窗口最后位置
        cur = pred_norm[i - 1] if i > 0 else 0.0
        p = probs.copy()
        nxt = cur + BIN_CENTERS
        p[np.abs(nxt) > bound] = 0.0
        if p.sum() < 1e-9:  # 全被挡（极端），退回原始分布
            p = probs.copy()
        p /= p.sum()
        pred_norm[i] = cur + BIN_CENTERS[rng.choice(NB, p=p)]
    return pred_norm * X_SCALE

def enforce_chord_gaps(out_notes, min_gap=200.0, xlim=675.0):
    """跨线同刻多音符：强制相邻间距 >= min_gap（贪心双向推）。"""
    from collections import defaultdict
    grp = defaultdict(list)
    for n in out_notes:
        grp[round(n['t'], 3)].append(n)
    for t, ns in grp.items():
        if len(ns) < 2:
            continue
        ns.sort(key=lambda n: n['x'])
        xs = [n['x'] for n in ns]
        # 从左到右推
        for i in range(1, len(xs)):
            if xs[i] - xs[i-1] < min_gap:
                xs[i] = xs[i-1] + min_gap
        if xs[-1] > xlim:
            # 溢出：从右到左反向推
            for i in range(len(xs)-2, -1, -1):
                if xs[i+1] - xs[i] < min_gap:
                    xs[i] = xs[i+1] - min_gap
        for n, x in zip(ns, xs):
            n['x'] = float(np.clip(x, -xlim, xlim))
    return out_notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fid", type=str)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ck = torch.load(os.path.join(BASE, "layout_ckpt_v5_low", "model.pth"), map_location="cpu")
    model = GRUCls(); model.load_state_dict(ck["model"]); model.eval()
    rng = np.random.RandomState(args.seed)

    d = json.load(open(os.path.join(BASE, "parsed", f"{args.fid}.json")))
    bpm_path = os.path.join(BASE, "bpm", f"{args.fid}.json")
    bpm = json.load(open(bpm_path))["segs"] if os.path.exists(bpm_path) else None

    # 按线分组
    lines = {}
    for n in d["notes"]:
        lines.setdefault(n["line_id"], []).append(n)

    out_notes = []
    for lid in sorted(lines.keys()):
        lnotes = sorted(lines[lid], key=lambda n: n["t"])
        if len(lnotes) < 4:
            for n in lnotes:
                out_notes.append({"t": n["t"], "type": n.get("type", 1), "x": n["x"], "line_id": lid, "dur": n.get("dur", 0.0)})
            continue
        Xw = build_features(lnotes, bpm)
        pred = infer_line(model, Xw, rng)
        for j, n in enumerate(lnotes):
            out_notes.append({"t": n["t"], "type": n.get("type", 1),
                              "x": float(pred[j]), "line_id": lid, "dur": n.get("dur", 0.0)})

    out_notes.sort(key=lambda n: (n["line_id"], n["t"]))
    # 只保留 Tap(1) 和 Hold(2)（用户要求：不要 Flick/Drag）
    KEEP_TYPES = {1, 2}
    out_notes = [n for n in out_notes if n.get("type", 1) in KEEP_TYPES]
    out_notes = enforce_chord_gaps(out_notes)
    result = {"fid": args.fid, "mode": "sample", "seed": args.seed, "n_notes": len(out_notes),
              "x_scale_note": "x 中心0，范围约±675px；制谱器用 0~1350 则 x_rpe = x + 675", "notes": out_notes}
    if args.out:
        json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"已写入 {args.out}: {len(out_notes)} 个音符")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=1)[:2000])


if __name__ == "__main__":
    main()
