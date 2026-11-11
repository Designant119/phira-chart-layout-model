#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lost Requiem：采音候选 → 单线分线 → 排键输入。
BPM segs 从谱面 BPMList 提取（20 段变速）。
"""
import json, os
from collections import defaultdict

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
SRC = "/home/user/Doubao/chats/38447255748799234/audio_notes_lr.json"
FID = "lost_requiem"
LRJSON = "/home/user/Doubao/chats/38447255748799234/lr_song/2143114633218669.json"

chart = json.load(open(LRJSON))
bpm_list = chart["BPMList"]
segs = [[int(b["startTime"][0]), float(b["bpm"])] for b in bpm_list]

notes = json.load(open(SRC))["notes"]
notes.sort(key=lambda n: n["t"])
lines = defaultdict(list)
for n in notes:
    t = round(n["t"], 4)
    cnt = 2 if n["chord"] == 2 else 1
    for _ in range(cnt):
        lines[0].append(t)

all_notes = []
for lid, ts in lines.items():
    for t in sorted(ts):
        all_notes.append({"t": t, "x": 0.0, "type": 1, "dur": 0.0, "line_id": lid})
all_notes.sort(key=lambda n: (n["line_id"], n["t"]))

os.makedirs(os.path.join(BASE, "parsed"), exist_ok=True)
os.makedirs(os.path.join(BASE, "bpm"), exist_ok=True)
json.dump({"fid": FID, "notes": all_notes}, open(os.path.join(BASE, "parsed", f"{FID}.json"), "w"), ensure_ascii=False)
json.dump({"segs": segs}, open(os.path.join(BASE, "bpm", f"{FID}.json"), "w"))

# level_cache 补 LR（IN Lv.16）
lc_path = os.path.join(BASE, "level_cache.json")
lc = json.load(open(lc_path)) if os.path.exists(lc_path) else {}
if FID not in lc:
    lc[FID] = 16.0
    json.dump(lc, open(lc_path, "w"))

print(f"总音符 {len(all_notes)} | BPM segs {len(segs)} 段 | 每线 {len(lines[0])}")
