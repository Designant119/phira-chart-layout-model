#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 api.phira.cn 拿谱面元数据里的 file 包，只抽音频到 data/audio/<fid>.<ext>。"""
import json, os, sys, time, zipfile, io, glob
from concurrent.futures import ThreadPoolExecutor
import requests

DATA = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/data"
AUD = os.path.join(DATA, "audio")
os.makedirs(AUD, exist_ok=True)
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (chart-audio/1.0)"})
AUDIO_EXT = (".mp3", ".ogg", ".wav", ".m4a", ".flac")


def meta(fid):
    for _ in range(3):
        try:
            r = S.get(f"https://api.phira.cn/chart/{fid}", timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1)
    return None


def one(fid):
    for ext in AUDIO_EXT:
        p = os.path.join(AUD, fid + ext)
        if os.path.exists(p) and os.path.getsize(p) > 100000:
            return "skip"
    try:
        m = meta(fid)
        if not m or not m.get("file"):
            return "nometa"
        r = S.get(m["file"], timeout=180)
        if r.status_code != 200:
            return f"http{r.status_code}"
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = [n for n in z.namelist() if n.lower().endswith(AUDIO_EXT)]
        if not names:
            return "noaudio"
        names.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
        n = names[0]
        ext = os.path.splitext(n)[1].lower()
        data = z.read(n)
        open(os.path.join(AUD, fid + ext), "wb").write(data)
        return "ok"
    except Exception as e:
        return "err:" + type(e).__name__


fids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(DATA, "parsed", "*.json")))
print("fids", len(fids), flush=True)
res = {}
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, (fid, st) in enumerate(zip(fids, ex.map(one, fids))):
        res[fid] = st
        if (i + 1) % 10 == 0:
            print(i + 1, st, flush=True)
print({k: list(res.values()).count(k) for k in set(res.values())}, flush=True)
json.dump(res, open(os.path.join(DATA, "audio_status.json"), "w"), ensure_ascii=False, indent=1)
