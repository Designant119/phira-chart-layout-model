#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立对照验证：官方语义（easing 表 + 多 layer 累加 + 父线叠加） vs 修复版管道输出"""
import json, math, zipfile, sys
sys.path.insert(0, "/home/user/Doubao/chats/38445055719679234/phira_charts_dl")
from easing_lib import line_state_at, _beat

def beat_to_sec(segs, b):
    """segs: [(startBeat, bpm, startSec)] 绝对拍分段"""
    for i, (sb, bpm, ss) in enumerate(segs):
        if i == len(segs) - 1:
            return ss + (b - sb) * 60.0 / bpm
        nxt = segs[i+1][0]
        if b < nxt:
            return ss + (b - sb) * 60.0 / bpm
    return 0.0

def build_segs(bpm_list):
    segs = []
    base = 0.0
    for i, e in enumerate(bpm_list):
        sb = _beat(e["startTime"])
        bpm = e["bpm"]
        if i == len(bpm_list) - 1:
            segs.append((sb, bpm, base))
        else:
            nxt = _beat(bpm_list[i+1]["startTime"])
            segs.append((sb, bpm, base))
            base += (nxt - sb) * 60.0 / bpm
    return segs

def parse_indep(pez_path):
    zf = zipfile.ZipFile(pez_path)
    names = zf.namelist()
    jn = [n for n in names if n.lower().endswith(".json") and "AutoSave" not in n]
    raw = zf.read(jn[0]).decode("utf-8", errors="replace")
    chart = json.loads(raw)
    segs = build_segs(chart.get("BPMList", []))
    out = []
    for jl in chart.get("judgeLineList", []):
        for n in jl.get("notes", []) or []:
            if n.get("isFake"):
                continue
            if n.get("type", 1) not in (1, 2, 3, 4):
                continue
            b = _beat(n["startTime"])
            px, py, theta = line_state_at(jl, chart, b)
            if abs(theta) > 10.0:
                continue
            t = beat_to_sec(segs, b)
            gx = px + (n.get("positionX", 0.0) or 0.0) * math.cos(math.radians(theta))
            dur = 0.0
            et = n.get("endTime")
            if et is not None:
                d = beat_to_sec(segs, _beat(et)) - t
                dur = d if d > 0 else 0.0
            out.append({"t": round(t, 6), "x": round(gx, 3), "type": n.get("type", 1), "dur": round(dur, 6)})
    out.sort(key=lambda k: (k["t"], k["x"]))
    return out

# 旧线性 parsed（bug#2 前产物，作对照参考）
old_path = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl/parsed/3200.json"
old = None
try:
    with open(old_path) as f:
        old = json.load(f)
except Exception:
    pass

ind = parse_indep("subsets/unstable_0002/3200_Fickle Mind.pez")
print(f"独立官方语义: {len(ind)} 音符")
if old:
    print(f"旧线性解析:   {len(old.get('notes', []))} 音符")
    old_notes = sorted(old.get("notes", []), key=lambda k: (k["t"], k["x"]))
    # x 差异统计
    diffs = 0
    maxdiff = 0.0
    for a, b in zip(ind, old_notes):
        if abs(a["x"] - b["x"]) > 0.5:
            diffs += 1
            maxdiff = max(maxdiff, abs(a["x"] - b["x"]))
    print(f"x 差异>0.5 的音符数: {diffs}/{len(ind)}，最大差异 {maxdiff:.2f}")
# 保存独立结果
with open("verify_3200_official.json", "w") as f:
    json.dump({"fid": "3200", "count": len(ind), "notes": ind}, f, ensure_ascii=False)
print("已保存 verify_3200_official.json")
# 抽样展示前 8 个
for n in ind[:8]:
    print(n)
