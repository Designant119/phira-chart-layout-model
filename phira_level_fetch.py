#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step1 并发版：phira chart page=1..333 -> {id(str): difficulty(float)}。
6 线程并发；每页 curl+8118代理+重试3次指数退避；失败页记入 fetch_level_fail.txt。
断点续跑：成功页集合存 fetched_pages.txt，cache 存 level_cache_phira.json。
"""
import json, os, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
OUT_DIR = os.path.join(BASE, "phira_full")
API = "https://api.phira.cn/chart?page={}"
PROXY = "http://127.0.0.1:8118"
FAIL_LOG = os.path.join(OUT_DIR, "fetch_level_fail.txt")
CACHE_OUT = os.path.join(OUT_DIR, "level_cache_phira.json")
DONE_PAGES = os.path.join(OUT_DIR, "fetched_pages.txt")
N_PAGE = 333
WORKERS = 6
lock = threading.Lock()


def curl_json(url, retries=3, timeout=30):
    for i in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-s", "-L", "--max-time", str(timeout), "--proxy", PROXY,
                 "-H", "User-Agent: Mozilla/5.0 (level-fetch/1.0)", url],
                capture_output=True, timeout=timeout + 15)
            if out.returncode == 0 and out.stdout:
                try:
                    return json.loads(out.stdout)
                except json.JSONDecodeError:
                    pass
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2.0 * (2 ** i))
    return None


def fetch_page(page):
    data = curl_json(API.format(page))
    if data is None or "results" not in data:
        return page, None
    recs = {}
    for rec in data["results"]:
        cid, diff = rec.get("id"), rec.get("difficulty")
        if cid is not None and diff is not None:
            recs[str(cid)] = float(diff)
    return page, recs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = {}
    if os.path.exists(CACHE_OUT):
        try:
            cache = json.load(open(CACHE_OUT))
        except Exception:
            cache = {}
    done = set()
    if os.path.exists(DONE_PAGES):
        done = set(int(x) for x in open(DONE_PAGES).read().split() if x.strip())
    pending = [p for p in range(1, N_PAGE + 1) if p not in done]
    print(f"续跑：cache={len(cache)} 已完成页={len(done)} 待抓={len(pending)}", flush=True)

    failed = []
    ok_count = len(done)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_page, p): p for p in pending}
        for fu in as_completed(futs):
            page, recs = fu.result()
            if recs is None:
                failed.append(page)
            else:
                with lock:
                    cache.update(recs)
                    done.add(page)
                    ok_count += 1
            if (ok_count + len(failed)) % 10 == 0:
                json.dump(cache, open(CACHE_OUT + ".tmp", "w"))
                os.replace(CACHE_OUT + ".tmp", CACHE_OUT)
                open(DONE_PAGES, "w").write(" ".join(str(x) for x in sorted(done)))
                el = time.time() - t0
                print(f"进度 完成={ok_count} 失败={len(failed)} cache={len(cache)} elapsed={el:.0f}s", flush=True)

    json.dump(cache, open(CACHE_OUT + ".tmp", "w"))
    os.replace(CACHE_OUT + ".tmp", CACHE_OUT)
    open(DONE_PAGES, "w").write(" ".join(str(x) for x in sorted(done)))
    with open(FAIL_LOG, "w") as f:
        f.write("\n".join(str(p) for p in sorted(failed)) + ("\n" if failed else ""))
    print("=" * 60, flush=True)
    print(f"成功页数: {ok_count} / {N_PAGE}", flush=True)
    print(f"失败页数: {len(failed)}", flush=True)
    print(f"cache 条数: {len(cache)}", flush=True)
    print(f"失败页: {sorted(failed)}", flush=True)
    print(f"DONE -> {CACHE_OUT}", flush=True)


if __name__ == "__main__":
    main()
