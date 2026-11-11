#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 parsed 的每个 fid 补读 pez 内 RPE json 的 BPMList，存 bpm/{fid}.json"""
import json, os, sys, zipfile, glob
from multiprocessing import Pool

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
BPM_DIR = os.path.join(BASE, "bpm")
os.makedirs(BPM_DIR, exist_ok=True)

manifest = json.load(open(os.path.join(BASE, "manifest.json")))


def beat_of(ts):
    """RPE 拍点 [bar, beat, tick] -> beat（1 小节 4 拍，1 拍 192 tick）"""
    bar, bt, tk = ts
    return bar * 4 + bt + tk / 192.0


def extract_one(fid):
    out_path = os.path.join(BPM_DIR, f"{fid}.json")
    if os.path.exists(out_path):
        return (fid, "exists", None)
    ent = manifest.get(fid)
    path = ent.get("path") if isinstance(ent, dict) else ent
    if not path or not os.path.exists(path):
        return (fid, "no_pez", None)
    try:
        with zipfile.ZipFile(path) as z:
            jname = None
            for n in z.namelist():
                if n.lower().endswith(".json"):
                    jname = n
                    break
            if jname is None:
                return (fid, "no_json", None)
            data = json.loads(z.read(jname))
        bl = data.get("BPMList") or data.get("bpmList")
        if not isinstance(bl, list) or len(bl) == 0:
            return (fid, "no_bpmlist", None)
        segs = []
        for s in bl:
            try:
                bpm = float(s.get("bpm", s.get("BPM", 0)))
                st = s.get("startTime", s.get("starttime", [0, 0, 0]))
                if not isinstance(st, (list, tuple)) or len(st) < 3:
                    st = [0, 0, 0]
                segs.append((beat_of(st), bpm))
            except Exception:
                continue
        if not segs:
            return (fid, "empty_segs", None)
        segs.sort(key=lambda x: x[0])
        # 规范化：起始补 0（若第一段不从 0 开始，前面按第一段 bpm 计）
        first_beat = segs[0][0]
        if first_beat > 0:
            segs.insert(0, (0.0, segs[0][1]))
        out = {
            "fid": fid,
            "segs": [[round(b, 6), bpm] for b, bpm in segs],
            "n_segs": len(segs),
            "note": "segs = [[start_beat, bpm]...]，beat→秒：累计积分；相位=floor(beat*4)%16",
        }
        with open(out_path, "w") as f:
            json.dump(out, f, ensure_ascii=False)
        return (fid, "ok", len(segs))
    except Exception as e:
        return (fid, f"err:{type(e).__name__}:{str(e)[:80]}", None)


def main():
    fids = []
    for p in sorted(glob.glob(os.path.join(BASE, "parsed", "*.json"))):
        fid = os.path.basename(p)[:-5]
        if not os.path.exists(os.path.join(BPM_DIR, f"{fid}.json")):
            fids.append(fid)
    done = len(glob.glob(os.path.join(BPM_DIR, "*.json")))
    print(f"待提取 BPM: {len(fids)}，已有 {done}", flush=True)
    if not fids:
        print("全部已有，跳过", flush=True)
        return
    ncpu = min(os.cpu_count() or 4, 8)
    with Pool(ncpu) as pool:
        results = pool.imap_unordered(extract_one, fids, chunksize=32)
        stat = {}
        n = 0
        for fid, status, extra in results:
            stat[status] = stat.get(status, 0) + 1
            n += 1
            if n % 500 == 0:
                print(f"进度 {n}/{len(fids)} {stat}", flush=True)
    print("完成:", stat, flush=True)


if __name__ == "__main__":
    main()
