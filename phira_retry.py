#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补抓 failed.json 里的 missing 谱面。幂等。"""
import json, os, sys, time, subprocess
from phira_fetch import fetch_chart
BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
OUT = os.path.join(BASE, "phira_full")

def main():
    f = json.load(open(os.path.join(OUT, "failed.json")))
    missing = f.get("missing", [])
    fids = [m["fid"] if isinstance(m, dict) else m for m in missing]
    ok, fail = [], []
    for i, fid in enumerate(fids):
        p = os.path.join(OUT, "parsed", f"{fid}.json")
        if os.path.exists(p):
            ok.append(fid); continue
        try:
            s, msg = fetch_chart(fid, OUT)
            (ok if s else fail).append((fid, msg))
            print(f"[{i+1}/{len(fids)}] {fid}: {'OK' if s else 'FAIL'} {msg}", flush=True)
        except Exception as e:
            fail.append((fid, str(e)[:100]))
            print(f"[{i+1}/{len(fids)}] {fid}: EXC {str(e)[:100]}", flush=True)
        time.sleep(0.4)
    json.dump({"retry_ok": ok, "retry_fail": fail}, open(os.path.join(OUT, "retry_result.json"), "w"), ensure_ascii=False, indent=1)
    print(f"补抓完成: OK {len(ok)} / FAIL {len(fail)} → phira_full/retry_result.json")

if __name__ == "__main__":
    main()
