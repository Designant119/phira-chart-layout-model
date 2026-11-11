#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 Phira 列表 API（1-325 页），建立 fid → file 直链映射。
列表 API 稳定，详情 API 有故障窗口——用直链绕开详情端点。
输出：phira_full/chart_list.json {"fid": "https://phira.5wyxi.com/files/<uuid>"}
"""
import json, sys, time
sys.path.insert(0, '.')
import phira_fetch as pf

OUT = "phira_full/chart_list.json"
MAX_PAGE = 325
mapping = {}
page_fail = []
for page in range(1, MAX_PAGE + 1):
    url = f"https://api.phira.cn/chart?page={page}"
    try:
        raw = pf.http_get(url, timeout=30)
        data = json.loads(raw)
        results = data.get("results") or []
        if not results:
            page_fail.append(page)
            continue
        for r in results:
            fid = str(r["id"])
            if r.get("file"):
                mapping[fid] = r["file"]
    except Exception as e:
        page_fail.append(page)
        print(f"page {page} FAIL: {type(e).__name__} {str(e)[:50]}", flush=True)
    time.sleep(0.5)
    if page % 50 == 0:
        print(f"page {page}/{MAX_PAGE} mapped={len(mapping)}", flush=True)

json.dump(mapping, open(OUT, "w"), ensure_ascii=False)
print(f"DONE mapped={len(mapping)} page_fail={page_fail}", flush=True)
