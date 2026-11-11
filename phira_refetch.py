#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量校验+重抓：拉遍全部页列表，逐个验证本地文件，不合格重抓。"""
import json, os, subprocess, time
from phira_fetch import fetch_chart, API

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
OUT = os.path.join(BASE, "phira_full")


def curl_json(url, retries=3):
    for i in range(retries):
        r = subprocess.run(["curl", "-s", "-L", "--max-time", "45", url],
                           capture_output=True, timeout=60)
        if r.returncode == 0 and r.stdout[:1] in (b"{", b"["):
            return json.loads(r.stdout)
        time.sleep(2 * (i + 1))
    return None


def validate(fid):
    """检查本地 parsed+fiam 文件是否完好。返回 (ok, reason)。"""
    p = os.path.join(OUT, "parsed", f"{fid}.json")
    b = os.path.join(OUT, "bpm", f"{fid}.json")
    if not os.path.exists(p) or not os.path.exists(b):
        return False, "missing"
    try:
        d = json.load(open(p))
        n = d.get("notes", [])
        if len(n) < 10:
            return False, f"few_notes({len(n)})"
        if not all("t" in x and "x" in x for x in n[:20]):
            return False, "bad_fields"
        bpm = json.load(open(b))
        if not bpm.get("segs"):
            return False, "no_bpm"
        return True, "ok"
    except Exception as e:
        return False, f"corrupt({str(e)[:60]})"


def main():
    # 1) 拉全部页
    all_ids = []
    page = 1
    empty_pages = 0
    while True:
        d = curl_json(f"{API}?page={page}")
        if d is None:
            print(f"page {page}: 列表拉取失败，跳过", flush=True)
            page += 1
            empty_pages += 1
            if empty_pages > 5:
                break
            continue
        rs = d.get("results", [])
        if not rs:
            break
        all_ids += [str(x["id"]) for x in rs]
        if page % 25 == 0:
            print(f"列表已拉 {page} 页 / {len(all_ids)} 谱面", flush=True)
        page += 1
        time.sleep(0.3)
    all_ids = sorted(set(all_ids))
    print(f"全量列表共 {len(all_ids)} 谱面", flush=True)

    # 2) 逐个校验
    bad, ok = [], 0
    for fid in all_ids:
        good, reason = validate(fid)
        if good:
            ok += 1
        else:
            bad.append({"fid": fid, "reason": reason})
    print(f"校验完成: 完好 {ok} / 待重抓 {len(bad)}", flush=True)

    # 3) 重抓不合格
    refetch_ok, refetch_fail = [], []
    for i, item in enumerate(bad):
        fid = item["fid"]
        try:
            s, msg = fetch_chart(fid, OUT)
            (refetch_ok if s else refetch_fail).append({"fid": fid, "msg": msg})
            print(f"[重抓 {i+1}/{len(bad)}] {fid}({item['reason']}): {'OK' if s else 'FAIL '+msg}", flush=True)
        except Exception as e:
            refetch_fail.append({"fid": fid, "msg": str(e)[:100]})
        time.sleep(0.4)

    # 4) 终验
    final_ok = sum(1 for fid in all_ids if validate(fid)[0])
    json.dump({"total_listed": len(all_ids), "initial_ok": ok,
               "refetch_attempted": len(bad), "refetch_ok": refetch_ok,
               "refetch_fail": refetch_fail, "final_ok": final_ok},
              open(os.path.join(OUT, "refetch_report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"完成: 列表{len(all_ids)} / 初验完好{ok} / 重抓{len(bad)} / 终验完好{final_ok}", flush=True)


if __name__ == "__main__":
    main()
