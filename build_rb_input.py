#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采音候选 → 4 条固定判定线分线 → 生成排键模型输入（parsed 格式）。
- 双押时刻：2 音对称分配（交替 线(0,3) / 线(1,2)）
- 单押时刻：1 音轮转 4 线（跳过同刻已占用）
输出 parsed/87039153.json + bpm/87039153.json（供 infer_chord 推理）
"""
import json, os
from collections import defaultdict

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
SRC = "/home/user/Doubao/chats/38447255748799234/audio_notes.json"

notes = json.load(open(SRC))["notes"]
notes.sort(key=lambda n: n["t"])

# 用户要求：全部音符一条判定线（不分线）。双押时刻 = 同刻 2 音（同一线上不同 x）
lines = defaultdict(list)
for n in notes:
    t = round(n["t"], 4)
    cnt = 2 if n["chord"] == 2 else 1
    for _ in range(cnt):
        lines[0].append(t)

# 每线音符排序，写 parsed 格式
all_notes = []
for lid, ts in lines.items():
    for t in sorted(ts):
        all_notes.append({"t": t, "x": 0.0, "type": 1, "dur": 0.0, "line_id": lid})
all_notes.sort(key=lambda n: (n["line_id"], n["t"]))

os.makedirs(os.path.join(BASE, "parsed"), exist_ok=True)
os.makedirs(os.path.join(BASE, "bpm"), exist_ok=True)
json.dump({"fid": "87039153", "notes": all_notes},
          open(os.path.join(BASE, "parsed", "87039153.json"), "w"), ensure_ascii=False)
json.dump({"segs": [[0.0, 133.0]]}, open(os.path.join(BASE, "bpm", "87039153.json"), "w"))

per_line = {lid: len(ts) for lid, ts in lines.items()}
print(f"总音符 {len(all_notes)} | 每线 {per_line}")
