#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Phira 列表抓高质量谱面到 corpus/parsed + bpm。
只抓 ranked/stable 池，按难度和 rating 筛；谱面文件不入库，脚本入库。
用法：python3 phira_corpus.py --outdir /path/corpus --max-charts 100
"""
import os, json, argparse, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from phira_fetch import fetch_chart

API = "https://api.phira.cn/chart"


def api(url):
    for i in range(3):
        r = subprocess.run(["curl", "-s", "-L", "--max-time", "30", "-H",
                            "User-Agent: Mozilla/5.0 (chart-corpus/1.0)", url],
                           capture_output=True)
        if r.returncode == 0 and r.stdout.startswith(b"{"):
            try:
                return json.loads(r.stdout)
            except Exception:
                pass
        time.sleep(1.5)
    return {}


def crawl(source, max_pages=40):
    first = api(f"{API}?{source}=true&page=1")
    rs = first.get("results", [])
    if not rs:
        return []
    total = int(first.get("count", 0) or 0)
    pages = min(max_pages, max(1, (total + 29) // 30))
    print(f"  {source}: total {total}, pages {pages}", flush=True)
    out = list(rs)
    def one(pg):
        return api(f"{API}?{source}=true&page={pg}").get("results", [])
    with ThreadPoolExecutor(max_workers=6) as ex:
        for part in ex.map(one, range(2, pages + 1)):
            out += part
    print(f"  {source}: got {len(out)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/home/user/Doubao/chats/38453392661469954/_run_real_traj/phira_corpus")
    ap.add_argument("--min-diff", type=float, default=11.0)
    ap.add_argument("--max-diff", type=float, default=16.5)
    ap.add_argument("--max-charts", type=int, default=100)
    ap.add_argument("--min-rating", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    all_charts = {}
    for src in ("ranked", "stable"):
        print(f"抓 {src} 列表", flush=True)
        for c in crawl(src):
            all_charts[c["id"]] = c
    print("去重后", len(all_charts), "张", flush=True)
    cand = []
    for c in all_charts.values():
        try:
            diff = float(c.get("difficulty", 0))
            rating = float(c.get("rating", 0) or 0)
        except Exception:
            continue
        if not (a.min_diff <= diff <= a.max_diff):
            continue
        if rating < a.min_rating:
            continue
        tags = set(c.get("tags") or [])
        if tags & {"joke", "troll", "meme", "shitpost"}:
            continue
        cand.append(c)
    cand.sort(key=lambda c: (float(c.get("rating", 0) or 0), float(c.get("ratingCount", 0) or 0)), reverse=True)
    print(f"难度 {a.min_diff}-{a.max_diff} 候选 {len(cand)}，取前 {a.max_charts}", flush=True)
    selected = cand[:a.max_charts]
    json.dump({"selected": selected, "candidates": len(cand)}, open(os.path.join(a.outdir, "index.json"), "w"),
              ensure_ascii=False, indent=1)
    os.makedirs(os.path.join(a.outdir, "parsed"), exist_ok=True)
    def one(c):
        fp = os.path.join(a.outdir, "parsed", f"{c['id']}.json")
        if os.path.exists(fp):
            return "skip"
        try:
            good, msg = fetch_chart(str(c["id"]), a.outdir)
            return "ok" if good else f"FAIL {msg}"
        except Exception as e:
            return f"ERR {repr(e)[:80]}"
    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, res in enumerate(ex.map(one, selected)):
            if res == "skip": skip += 1
            elif res == "ok": ok += 1
            else:
                fail += 1; print(res, flush=True)
            if (i + 1) % 10 == 0:
                print(f"进度 {i+1}/{len(selected)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"完成 ok={ok} skip={skip} fail={fail} → {a.outdir}", flush=True)


if __name__ == "__main__":
    main()
