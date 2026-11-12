#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：不带 enforce_chord_gaps 后处理的原始推理评估。
对每个 fid 跑三组（seed 7）：
  a) v5_low 一遍（复刻 infer_layout_ckpt.main 的逐线推理，跳过后处理）
  b) chord 一遍 pass1（chord_x=0）
  c) chord 两遍（第二遍注入同刻粗排均值 chord_x）
均只留 type 1(tap)/2(hold)，坐标中心0直接与原谱比较。
用法: python3 eval_raw.py <fid>
"""
import json, os, sys
import numpy as np
from collections import defaultdict

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
KEEP = (1, 2)
SEED = 7


# ---------- 推理（无后处理） ----------
def infer_v5_low_raw(fid):
    import torch
    from train_layout_v4 import GRUCls
    from infer_layout_ckpt import build_features, infer_line
    ck = torch.load(os.path.join(BASE, "layout_ckpt_v5_low", "model.pth"), map_location="cpu")
    model = GRUCls(); model.load_state_dict(ck["model"]); model.eval()
    rng = np.random.RandomState(SEED)  # 与 infer_layout_ckpt.main 一致：单 rng 跨线共享
    d = json.load(open(os.path.join(BASE, "parsed", f"{fid}.json")))
    bpm_p = os.path.join(BASE, "bpm", f"{fid}.json")
    bpm = json.load(open(bpm_p))["segs"] if os.path.exists(bpm_p) else None
    lines = defaultdict(list)
    for n in d["notes"]:
        lines[n["line_id"]].append(n)
    out = []
    for lid in sorted(lines.keys()):
        lnotes = sorted(lines[lid], key=lambda n: n["t"])
        if len(lnotes) < 4:
            for n in lnotes:
                out.append({"t": n["t"], "type": n.get("type", 1), "x": n["x"],
                            "line_id": lid, "dur": n.get("dur", 0.0)})
            continue
        Xw = build_features(lnotes, bpm)
        ts = np.array([n["t"] for n in lnotes], dtype=np.float64)
        pred = infer_line(model, Xw, ts, rng)
        for j, n in enumerate(lnotes):
            out.append({"t": n["t"], "type": n.get("type", 1), "x": float(pred[j]),
                        "line_id": lid, "dur": n.get("dur", 0.0)})
    return [n for n in out if n.get("type", 1) in KEEP]


def infer_chord_raw(fid, two_pass):
    import torch
    from train_chord import GRUCls8
    from infer_chord import infer_song
    ck = torch.load(os.path.join(BASE, "layout_ckpt_chord_full", "model.pth"), map_location="cpu")
    model = GRUCls8(); model.load_state_dict(ck["model"]); model.eval()
    notes = infer_song(fid, model, SEED, two_pass=two_pass)
    return [n for n in notes if n.get("type", 1) in KEEP]


# ---------- 指标（与 chord_eval.py 一致） ----------
def by_line(notes):
    d = defaultdict(list)
    for n in notes:
        d[n["line_id"]].append(n)
    return {lid: sorted(lst, key=lambda n: (float(n["t"]), n.get("type", 1))) for lid, lst in d.items()}


def metrics(fid, pred):
    orig = [n for n in json.load(open(os.path.join(BASE, "parsed", f"{fid}.json")))["notes"]
            if n.get("type", 1) in KEEP]
    om, pm = by_line(orig), by_line(pred)
    pairs = []
    for lid in om:
        for o, p in zip(om[lid], pm.get(lid, [])):
            pairs.append((o, p))
    grp = defaultdict(list)
    for o, p in pairs:
        grp[round(float(o["t"]), 2)].append((o, p))
    chord_pairs, single_pairs = [], []
    spac = []
    for t, ps in grp.items():
        if len(ps) >= 2:
            chord_pairs.extend(ps)
            pxs = sorted(float(p["x"]) for o, p in ps)
            spac += [abs(pxs[i + 1] - pxs[i]) for i in range(len(pxs) - 1)]
        else:
            single_pairs.extend(ps)
    cmae = float(np.mean([abs(float(o["x"]) - float(p["x"])) for o, p in chord_pairs]))
    smae = float(np.mean([abs(float(o["x"]) - float(p["x"])) for o, p in single_pairs]))
    return {
        "chord_mae": cmae,
        "single_mae": smae,
        "spac_med": float(np.median(spac)) if spac else float("nan"),
        "spac_p25": float(np.percentile(spac, 25)) if spac else float("nan"),
        "spac_p75": float(np.percentile(spac, 75)) if spac else float("nan"),
        "n_chord": len(chord_pairs),
        "n_total": len(pairs),
    }


def main():
    fid = sys.argv[1]
    print(f"### fid={fid}  seed={SEED}  无 enforce_chord_gaps 后处理")
    rows = []
    for label, fn in [("v5_low一遍", lambda: infer_v5_low_raw(fid)),
                      ("chord一遍(pass1)", lambda: infer_chord_raw(fid, False)),
                      ("chord两遍", lambda: infer_chord_raw(fid, True))]:
        m = metrics(fid, fn())
        rows.append((label, m))
        print(f"{label:18s} 双押MAE={m['chord_mae']:7.1f}  单音MAE={m['single_mae']:7.1f}  "
              f"间距中位={m['spac_med']:7.1f}  (P25={m['spac_p25']:.0f}/P75={m['spac_p75']:.0f})  "
              f"双押音符={m['n_chord']}/{m['n_total']}")


if __name__ == "__main__":
    main()
