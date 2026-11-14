#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 osu.direct 拉 mania 谱（4K/7K），只存 .osu 文本 + 元数据。"""
import json, os, sys, time, re
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "osu"); os.makedirs(OUT, exist_ok=True)
META = os.path.join(ROOT, "meta.json")
S = requests.Session(); S.headers.update({"User-Agent": "mania-layout-research/1.0"})
MAX_CHARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
MAX_OFFSET = int(sys.argv[2]) if len(sys.argv) > 2 else 3000


def get(url, timeout=30):
    for _ in range(3):
        try:
            r = S.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
        time.sleep(0.8)
    return None


cands = []
for off in range(0, MAX_OFFSET + 1, 50):
    txt = get(f"https://osu.direct/api/v2/search?query=&offset={off}&mode=3")
    if not txt:
        continue
    try:
        arr = json.loads(txt)
    except Exception:
        continue
    for s in arr:
        for b in s.get("beatmaps", []):
            if b.get("mode_int") != 3:
                continue
            keys = int(b.get("cs") or 0)
            if keys not in (4, 7):
                continue
            n = int(b.get("count_circles") or 0)
            if n < 200 or n > 2500:
                continue
            version = b.get("version") or ""
            if re.search(r"\[\d+(\.\d+)?x Rate\]", version):
                continue
            cands.append((b["id"], keys, n, s["id"], s["title"], version, b.get("difficulty_rating")))
print("候选", len(cands), flush=True)
seen = set(); picked = []
for c in cands:
    if c[0] in seen: continue
    seen.add(c[0]); picked.append(c)
picked = picked[:MAX_CHARTS]
print("下载", len(picked), flush=True)
meta = {}
if os.path.exists(META):
    meta = json.load(open(META))


def one(c):
    bid, keys, n, sid, title, version, star = c
    p = os.path.join(OUT, f"{bid}.osu")
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return bid, "skip"
    txt = get(f"https://osu.direct/api/osu/{bid}", timeout=60)
    if not txt or "[HitObjects]" not in txt:
        return bid, "fail"
    open(p, "w").write(txt)
    meta[str(bid)] = {"bid": bid, "keys": keys, "notes": n, "set_id": sid, "title": title,
                      "version": version, "star": star}
    return bid, "ok"


with ThreadPoolExecutor(max_workers=4) as ex:
    for i, (bid, st) in enumerate(ex.map(one, picked)):
        if (i + 1) % 50 == 0:
            print(i + 1, st, flush=True)
json.dump(meta, open(META, "w"), ensure_ascii=False, indent=1)
print("完成", len(meta), flush=True)
