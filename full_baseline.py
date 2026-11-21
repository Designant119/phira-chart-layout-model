#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量谱面排键基准统计：扫所有 parsed 谱，逐首算步长/换向/同列率/多押宽度/密度，
输出每首指标 jsonl + 整体与按密度分层的分位数汇总。"""
import json, glob, os, sys
import numpy as np
from collections import defaultdict, Counter

PARS = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/phira_all/parsed"
OUT_JSONL = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/baseline_per_song.jsonl"
OUT_SUM = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/baseline_summary.json"

NB = 81
HALF = 675.0


def song_stats(notes):
    notes = sorted(notes, key=lambda n: n['t'])
    g = defaultdict(list)
    for n in notes:
        g[round(n['t'], 3)].append(n['x'])
    order = sorted(g)
    groups = [g[t] for t in order]

    singles = [v[0] for v in groups if len(v) == 1]
    xs = np.array([x for v in groups for x in v], dtype=float)

    rec = {"n": len(notes)}
    if len(notes) < 8:
        return None

    ts = [t for t in order]
    ivs = np.diff(ts)
    rec["med_iv"] = float(np.median(ivs)) if len(ivs) else 0.2
    rec["dur"] = float(ts[-1] - ts[0])
    rec["density"] = len(notes) / max(rec["dur"], 1.0)

    # 单音序列步长
    if len(singles) >= 3:
        s = np.array(singles, dtype=float)
        dx = np.abs(np.diff(s))
        nz = dx[dx > 1e-6]
        rec["step_med"] = float(np.median(nz)) if len(nz) else 0.0
        rec["same_rate"] = float((dx < 20).mean())
        d = np.diff(s)
        d = d[np.abs(d) > 1e-6]
        if len(d) >= 2:
            flip = (np.sign(d[:-1]) * np.sign(d[1:]) < 0).mean()
            rec["flip"] = float(flip)
        else:
            rec["flip"] = 0.5
    else:
        rec["step_med"] = 0.0; rec["same_rate"] = 0.0; rec["flip"] = 0.5

    # 全音横向分布
    rec["x_std"] = float(xs.std())
    rec["absx_med"] = float(np.median(np.abs(xs)))
    rec["left"] = float((xs < -100).mean())
    rec["right"] = float((xs > 100).mean())
    rec["edge"] = float((np.abs(xs) > 400).mean())

    # 占用格位 / 单格最多
    q = np.clip(np.round((xs + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1).astype(int)
    u, c = np.unique(q, return_counts=True)
    rec["ncol_used"] = int(len(u))
    rec["col_max"] = int(c.max())

    # 多押宽度
    byk = defaultdict(list)
    for v in groups:
        if len(v) >= 2:
            byk[len(v)].append(max(v) - min(v))
    for k in (2, 3, 4):
        if byk[k]:
            rec[f"chord{k}_med"] = float(np.median(byk[k]))
            rec[f"chord{n}_n"] = len(byk[k]) if False else len(byk[k])
        else:
            rec[f"chord{k}_med"] = None
            rec[f"chord{k}_n"] = 0

    # 24 音窗口跨度
    if len(xs) >= 24:
        spans = [xs[j:j + 24].max() - xs[j:j + 24].min() for j in range(0, len(xs) - 24, 24)]
        rec["span24"] = float(np.median(spans))
    else:
        rec["span24"] = 0.0
    return rec


def pct(a, q):
    a = np.array([x for x in a if x is not None and np.isfinite(x)])
    if len(a) == 0:
        return None
    return {p: float(np.percentile(a, p)) for p in (25, 50, 75)}


def main():
    files = sorted(glob.glob(os.path.join(PARS, "*.json")))
    rows = []
    with open(OUT_JSONL, "w", encoding="utf-8") as fo:
        for f in files:
            try:
                d = json.load(open(f, encoding="utf-8"))
                st = song_stats(d["notes"])
                if st is None:
                    continue
                st["fid"] = os.path.basename(f)[:-5]
                rows.append(st)
                fo.write(json.dumps(st, ensure_ascii=False) + "\n")
            except Exception as e:
                print("skip", f, e, file=sys.stderr)

    print("有效谱面", len(rows))
    metrics = ["med_iv", "density", "step_med", "flip", "same_rate", "absx_med",
               "x_std", "left", "right", "edge", "ncol_used", "col_max", "span24",
               "chord2_med", "chord3_med", "chord4_med"]
    summary = {"n_songs": len(rows), "overall": {}, "by_density": {}}
    for m in metrics:
        summary["overall"][m] = pct([r.get(m) for r in rows], m)

    # 按密度三档
    dens = np.array([r["density"] for r in rows])
    q1, q2 = np.percentile(dens, [33, 66])
    bands = {"sparse": lambda d: d <= q1, "mid": lambda d: q1 < d <= q2, "dense": lambda d: d > q2}
    for b, fn in bands.items():
        sub = [r for r in rows if fn(r["density"])]
        summary["by_density"][b] = {"n": len(sub), "q1_density": float(q1), "q2_density": float(q2)}
        for m in metrics:
            summary["by_density"][b][m] = pct([r.get(m) for r in sub], m)

    json.dump(summary, open(OUT_SUM, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("分层阈值 密度", round(q1, 2), round(q2, 2))
    for b in ("sparse", "mid", "dense"):
        s = summary["by_density"][b]
        def g(m, k):
            v = s.get(m)
            return v[k] if v else 0.0
        print(b, "n=%d 密度中位%.2f 步长%.0f 换向%.2f 同列%.1f%% 多押2押%.0f" % (
            s["n"], g("density", "50"), g("step_med", "50"), g("flip", "50"),
            g("same_rate", "50") * 100, g("chord2_med", "50")))


if __name__ == "__main__":
    main()
