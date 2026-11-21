#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量谱面 drag(type=3) 排布规律统计。
扫 phira_all/parsed/*.json，统计 drag 的空间位置、同拍共押情况、
连续 drag 段（是否成滑动线）、段内走向/间隔一致性/是否同判定线。
输出 drag_summary.json + drag_per_song.jsonl
"""
import json, glob, os, sys
import numpy as np
from collections import defaultdict, Counter

PARS = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/phira_all/parsed"
OUT_JSONL = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/drag_per_song.jsonl"
OUT_SUM = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/drag_summary.json"

GAP = 1e-3  # 同拍判定容差


def pct(a, qs=(25, 50, 75)):
    a = np.array([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    if len(a) == 0:
        return None
    return {str(p): float(np.percentile(a, p)) for p in qs}


def song_drag_stats(notes):
    notes = sorted(notes, key=lambda n: n['t'])
    if len(notes) < 8:
        return None
    g = defaultdict(list)
    for n in notes:
        g[round(n['t'], 3)].append(n)
    order = sorted(g)
    groups = [g[t] for t in order]

    drags = [n for n in notes if n['type'] == 3]
    rec = {'n': len(notes), 'n_drag': len(drags), 'drag_ratio': len(drags) / len(notes)}

    if not drags:
        return rec

    # ---- 1. drag 出现时同拍共押情况 ----
    alone = tap_together = hold_together = other_together = drag_pair = 0
    for t in order:
        grp = g[t]
        ds = [n for n in grp if n['type'] == 3]
        if not ds:
            continue
        if len(grp) == 1:
            alone += 1
        else:
            has_tap = any(n['type'] == 1 for n in grp)
            has_hold = any(n['type'] == 2 for n in grp)
            if len(ds) >= 2:
                drag_pair += 1
            if has_tap:
                tap_together += 1
            elif has_hold:
                hold_together += 1
            else:
                other_together += 1
    rec['drag_alone_ratio'] = alone / max(1, alone + tap_together + hold_together + other_together)
    rec['drag_tap_ratio'] = tap_together / max(1, alone + tap_together + hold_together + other_together)
    rec['drag_pair_ratio'] = drag_pair / max(1, len([1 for t in order if any(n['type'] == 3 for n in g[t])]))

    # ---- 2. drag 空间分布（排除 x==0 的 rel 假象）----
    xs = np.array([n['x'] for n in drags], dtype=float)
    xz = xs[xs != 0]
    rec['drag_x0_ratio'] = float((xs == 0).mean())
    if len(xz) >= 3:
        rec['drag_absx_med'] = float(np.median(np.abs(xz)))
        rec['drag_left'] = float((xz < -100).mean())
        rec['drag_right'] = float((xz > 100).mean())
        rec['drag_center'] = float(((xz >= -100) & (xz <= 100)).mean())

    # ---- 3. 连续 drag 段 ----
    # 段定义：时间上相邻的两颗 drag 之间没有非 drag 音符插入
    drag_ts = sorted(set(round(n['t'], 3) for n in drags))
    all_ts = set(order)
    runs = []
    cur = []
    for t in order:
        if any(n['type'] == 3 for n in g[t]):
            cur.append(t)
        else:
            if cur:
                runs.append(cur); cur = []
    if cur:
        runs.append(cur)

    # 只保留段长>=2的段做走向分析
    long_runs = [r for r in runs if len(r) >= 2]
    rec['drag_run_n'] = len(runs)
    rec['drag_run_median_len'] = float(np.median([len(r) for r in runs])) if runs else 0.0
    rec['drag_run_long_ratio'] = len(long_runs) / max(1, len(runs))
    rec['drag_run_max_len'] = int(max((len(r) for r in runs), default=0))

    mono_rates, cv_iv, single_line = [], [], []
    for r in long_runs:
        pts = []
        for t in r:
            # 每拍取第一颗 drag
            d = [n for n in g[t] if n['type'] == 3][0]
            pts.append((t, d['x'], d['line_id']))
        # 去掉 x==0 假象后不够就不算
        ps = [p for p in pts if p[1] != 0]
        if len(ps) >= 3:
            xs_ = np.array([p[1] for p in ps]); dx = np.diff(xs_)
            nz = dx[np.abs(dx) > 1e-6]
            if len(nz) >= 2:
                s = np.sign(nz)
                mono_rates.append(float((s[:-1] * s[1:] > 0).mean()))
            iv = np.diff([p[0] for p in ps])
            if iv.mean() > 1e-6:
                cv_iv.append(float(iv.std() / iv.mean()))
        lids = set(p[2] for p in pts)
        single_line.append(1.0 if len(lids) == 1 else 0.0)
    rec['drag_run_mono_rate'] = float(np.median(mono_rates)) if mono_rates else None
    rec['drag_run_iv_cv'] = float(np.median(cv_iv)) if cv_iv else None
    rec['drag_run_single_line'] = float(np.mean(single_line)) if single_line else None

    # ---- 4. drag 与前一音符（任意类型）关系 ----
    if len(drags) >= 3:
        allpts = [(n['t'], n['x'], n['type']) for n in notes]
        allpts.sort()
        steps, same, dts = [], [], []
        prev = None
        for t, x, typ in allpts:
            if typ == 3 and prev is not None:
                dts.append(t - prev[0])
                if x != 0 and prev[1] != 0:
                    d = abs(x - prev[1])
                    steps.append(d)
                    same.append(1.0 if d < 20 else 0.0)
            prev = (t, x)
        if steps:
            rec['drag_step_med'] = float(np.median(steps))
            rec['drag_same_rate'] = float(np.mean(same))
        if dts:
            rec['drag_dt_med'] = float(np.median(dts))

    # ---- 5. drag 外侧程度 & 同拍时相对 tap 的偏移 ----
    if len(xz) >= 3:
        rec['drag_outside_ratio'] = float((np.abs(xz) > 400).mean())
    offs = []
    for t in order:
        grp = g[t]
        ds = [n for n in grp if n['type'] == 3 and n['x'] != 0]
        ts_ = [n for n in grp if n['type'] == 1 and n['x'] != 0]
        if ds and ts_:
            offs.append(np.mean([abs(d['x']) for d in ds]) - np.mean([abs(t['x']) for t in ts_]))
    if offs:
        rec['drag_vs_tap_abs_off'] = float(np.median(offs))
    return rec


def main():
    files = sorted(glob.glob(os.path.join(PARS, "*.json")))
    rows = []
    with open(OUT_JSONL, "w", encoding="utf-8") as fo:
        for f in files:
            try:
                d = json.load(open(f, encoding="utf-8"))
                st = song_drag_stats(d["notes"])
                if st is None:
                    continue
                st["fid"] = os.path.basename(f)[:-5]
                rows.append(st)
                fo.write(json.dumps(st, ensure_ascii=False) + "\n")
            except Exception as e:
                print("skip", f, e, file=sys.stderr)

    print("有效谱面", len(rows))
    has_drag = [r for r in rows if r.get('n_drag', 0) > 0]
    print("含drag谱面", len(has_drag), "占比 %.1f%%" % (100 * len(has_drag) / len(rows)))

    metrics = ['n', 'n_drag', 'drag_ratio', 'drag_alone_ratio', 'drag_tap_ratio', 'drag_pair_ratio',
               'drag_x0_ratio', 'drag_absx_med', 'drag_left', 'drag_right', 'drag_center',
               'drag_outside_ratio', 'drag_vs_tap_abs_off', 'drag_dt_med',
               'drag_run_n', 'drag_run_median_len', 'drag_run_long_ratio', 'drag_run_max_len',
               'drag_run_mono_rate', 'drag_run_iv_cv', 'drag_run_single_line',
               'drag_step_med', 'drag_same_rate']
    summary = {"n_songs": len(rows), "n_with_drag": len(has_drag), "overall": {}, "by_drag_density": {}}
    for m in metrics:
        summary["overall"][m] = pct([r.get(m) for r in has_drag])

    # 按 drag 占比分层
    ratios = np.array([r['drag_ratio'] for r in has_drag])
    q1, q2 = np.percentile(ratios, [33, 66])
    bands = {"low": lambda x: x <= q1, "mid": lambda x: q1 < x <= q2, "high": lambda x: x > q2}
    for b, fn in bands.items():
        sub = [r for r in has_drag if fn(r['drag_ratio'])]
        summary["by_drag_density"][b] = {"n_songs": len(sub), "q1": float(q1), "q2": float(q2)}
        for m in metrics:
            summary["by_drag_density"][b][m] = pct([r.get(m) for r in sub])

    json.dump(summary, open(OUT_SUM, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n=== 整体（含 drag 的谱面）===")
    for m in ['drag_ratio', 'drag_alone_ratio', 'drag_tap_ratio', 'drag_pair_ratio',
              'drag_x0_ratio', 'drag_absx_med', 'drag_run_median_len', 'drag_run_long_ratio',
              'drag_run_mono_rate', 'drag_run_iv_cv', 'drag_run_single_line',
              'drag_step_med', 'drag_same_rate']:
        v = summary["overall"][m]
        if v:
            print("%-24s 25%%=%.3f  50%%=%.3f  75%%=%.3f" % (m, v['25'], v['50'], v['75']))

    print("\n=== 按 drag 占比分层 ===")
    for b in ("low", "mid", "high"):
        s = summary["by_drag_density"][b]
        def g(m, k):
            v = s.get(m); return v[k] if v else 0.0
        print("%s n=%d drag占比中位%.3f 独立drag%.2f 与tap同拍%.2f 连续段中位%.0f 段长≥2比例%.2f 同向%.2f 同线%.2f |x|%.0f" % (
            b, s['n_songs'], g('drag_ratio', '50'), g('drag_alone_ratio', '50'), g('drag_tap_ratio', '50'),
            g('drag_run_median_len', '50'), g('drag_run_long_ratio', '50'),
            g('drag_run_mono_rate', '50') or 0, g('drag_run_single_line', '50') or 0,
            g('drag_absx_med', '50')))


if __name__ == "__main__":
    main()
