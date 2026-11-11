#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双押评估：v5_low / chord 推理结果 vs 原谱。
用法: python3 chord_eval.py <fid> <infer.json>
指标: 双押MAE / 单音MAE / 双押间距中位数(预测&原谱) / 双押组数 / 总音符数 / 双押占比
坐标: 原谱 parsed/{fid}.json 的 x 与推理输出 x 均为中心0坐标系（已实测原谱 x≈±450~500），
      直接比较，无需 +675 换算。
"""
import json, sys, os
import numpy as np
from collections import defaultdict

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
KEEP = (1, 2)  # tap / hold


def main():
    fid = sys.argv[1]
    pred_path = sys.argv[2]
    orig_path = os.path.join(BASE, "parsed", f"{fid}.json")

    orig = [n for n in json.load(open(orig_path))["notes"] if n.get("type", 1) in KEEP]
    pred = [n for n in json.load(open(pred_path))["notes"] if n.get("type", 1) in KEEP]

    ox = [n["x"] for n in orig]
    px = [n["x"] for n in pred]
    print(f"[坐标] 原谱 x min/max = {min(ox):.1f} / {max(ox):.1f}")
    print(f"[坐标] 预测 x min/max = {min(px):.1f} / {max(px):.1f}")

    # 推理按 (line_id, t) 顺序逐条线 1:1 替换 x。
    # 注意：双押组内多音符可能共享同一 (line_id, t, type)（parsed 里 line_id 不是道宽），
    # 不能用三元组 dict 匹配（会互相覆盖）。按线分组后按 t 稳定排序、按名次配对。
    def by_line(notes):
        d = defaultdict(list)
        for n in notes:
            d[n["line_id"]].append(n)
        return {lid: sorted(lst, key=lambda n: (float(n["t"]), n.get("type", 1))) for lid, lst in d.items()}

    om, pm = by_line(orig), by_line(pred)
    pairs = []
    for lid in om:
        ol, pl = om[lid], pm.get(lid, [])
        if len(ol) != len(pl):
            print(f"[警告] line {lid} 数量不一致: orig {len(ol)} vs pred {len(pl)}")
        for o, p in zip(ol, pl):
            pairs.append((o, p))
    print(f"[匹配] 配对 {len(pairs)} 组 | 原谱 {len(orig)} | 预测 {len(pred)}")

    # 按 round(t,2) 分双押组
    grp = defaultdict(list)
    for o, p in pairs:
        grp[round(float(o["t"]), 2)].append((o, p))

    chord_pairs, single_pairs = [], []
    spac_pred, spac_orig = [], []
    ngroups = 0
    max_within_dt = 0.0
    for t, ps in grp.items():
        tspan = max(float(o["t"]) for o, p in ps) - min(float(o["t"]) for o, p in ps)
        max_within_dt = max(max_within_dt, tspan)
        if len(ps) >= 2:
            ngroups += 1
            chord_pairs.extend(ps)
            pxs = sorted(float(p["x"]) for o, p in ps)
            oxs = sorted(float(o["x"]) for o, p in ps)
            spac_pred += [abs(pxs[i + 1] - pxs[i]) for i in range(len(pxs) - 1)]
            spac_orig += [abs(oxs[i + 1] - oxs[i]) for i in range(len(oxs) - 1)]
        else:
            single_pairs.extend(ps)

    chord_mae = float(np.mean([abs(float(o["x"]) - float(p["x"])) for o, p in chord_pairs])) if chord_pairs else float("nan")
    single_mae = float(np.mean([abs(float(o["x"]) - float(p["x"])) for o, p in single_pairs])) if single_pairs else float("nan")
    total = len(pairs)
    n_chord = len(chord_pairs)

    print("=" * 46)
    print(f"双押 MAE:            {chord_mae:.1f} px")
    print(f"单音 MAE:            {single_mae:.1f} px")
    print(f"双押间距中位数(预测): {float(np.median(spac_pred)):.1f} px")
    print(f"双押间距中位数(原谱): {float(np.median(spac_orig)):.1f} px")
    print(f"双押组数:            {ngroups}")
    print(f"总音符数:            {total}")
    print(f"双押占比:            {n_chord / total * 100:.1f}%  ({n_chord}/{total})")
    print(f"[校验] 组内最大时间差: {max_within_dt:.4f}s (应≤0.03s)")


if __name__ == "__main__":
    main()
