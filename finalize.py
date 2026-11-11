#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phira 采集收尾：全量列表对比 + 统计汇总 + 抽样校验。
1. 重拉 333 页列表，得全量 fid 集合与元信息
2. 与 phira_full/parsed 对比 → 缺失谱面写 failed.json
3. 汇总统计（总数/成功/失败/总音符数）→ report.json + 打印
4. 抽样 20 个 parsed 校验：notes 非空、t 单调不减、x∈[-1200,1200]、type∈{1,2,3,4}、dur≥0、bpm segs 有序且 bpm∈[60,300]
"""
import json, os, sys, random, collections

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
OUT = os.path.join(BASE, "phira_full")
sys.path.insert(0, BASE)
import phira_fetch as pf

API_LIST = "https://api.phira.cn/chart"
MAX_PAGE = 325  # 已验证：库总数 9748，30/页 → 有效最后一页 325（326-333 返回空 results）


def fetch_all_ids():
    """拉全部列表页，返回 {fid: item}。"""
    all_items = {}
    page_fails = []
    for page in range(1, MAX_PAGE + 1):
        try:
            raw = pf.http_get(f"{API_LIST}?page={page}", retries=3, timeout=60)
            data = json.loads(raw)
            for it in data.get("results", []):
                all_items[str(it["id"])] = it
        except Exception as e:
            page_fails.append(page)
        if page % 50 == 0:
            print(f"list page {page}/{MAX_PAGE} ...", flush=True)
    return all_items, page_fails


def check_notes(notes):
    """校验 notes 列表，返回 (ok, msg)。"""
    if not notes:
        return False, "notes为空"
    prev_t = -1e9
    for i, n in enumerate(notes):
        t, x, typ, dur = n["t"], n["x"], n["type"], n["dur"]
        if t < prev_t:
            return False, f"t不递增 idx={i} {prev_t}->{t}"
        if not (-1200.0 <= x <= 1200.0):
            return False, f"x越界 idx={i} x={x}"
        if typ not in (1, 2, 3, 4):
            return False, f"type非法 idx={i} type={typ}"
        if dur < 0:
            return False, f"dur为负 idx={i} dur={dur}"
        prev_t = t
    return True, f"{len(notes)}音 t/x/type/dur全部通过"


def check_bpm(segs):
    """校验 bpm segs，返回 (ok, msg)。segs: [[start_beat, bpm], ...]"""
    if not segs:
        return True, "无BPM段(合法)"
    prev_b = -1e9
    for i, (b, bpm) in enumerate(segs):
        if b < prev_b:
            return False, f"BPM段start_beat不递增 idx={i}"
        if not (60.0 <= float(bpm) <= 300.0):
            return False, f"bpm越界 idx={i} bpm={bpm}"
        prev_b = b
    return True, f"{len(segs)}段BPM通过"


def main():
    random.seed(20261015)
    print("== 1/3 重拉全量列表 ==", flush=True)
    all_items, page_fails = fetch_all_ids()
    print(f"列表共 {len(all_items)} 个谱面, 列表页失败 {page_fails}", flush=True)

    parsed_dir = os.path.join(OUT, "parsed")
    bpm_dir = os.path.join(OUT, "bpm")
    parsed_fids = set(f[:-5] for f in os.listdir(parsed_dir) if f.endswith(".json"))
    bpm_fids = set(f[:-5] for f in os.listdir(bpm_dir) if f.endswith(".json"))

    # 缺失 = 列表里有但 parsed 没有
    missing = sorted(all_items.keys() - parsed_fids)
    missing_items = []
    for fid in missing:
        it = all_items.get(fid, {})
        missing_items.append({"fid": fid, "name": it.get("name", ""),
                              "level": it.get("level", ""), "difficulty": it.get("difficulty")})
    # bpm 缺但 parsed 有（异常情况）
    bpm_missing = sorted(parsed_fids - bpm_fids)

    print(f"== 2/3 统计 == parsed={len(parsed_fids)} bpm={len(bpm_fids)} "
          f"缺失={len(missing)} bpm缺失={len(bpm_missing)}", flush=True)

    # 总音符数
    total_notes = 0
    note_counts = {}
    for fid in parsed_fids:
        try:
            d = json.load(open(os.path.join(parsed_dir, f"{fid}.json")))
            total_notes += len(d.get("notes", []))
        except Exception:
            pass

    # 抽样校验 20 个
    sample = random.sample(sorted(parsed_fids), min(20, len(parsed_fids)))
    sample_results = []
    all_ok = True
    for fid in sample:
        try:
            d = json.load(open(os.path.join(parsed_dir, f"{fid}.json")))
            b = json.load(open(os.path.join(bpm_dir, f"{fid}.json")))
            ok1, m1 = check_notes(d.get("notes", []))
            ok2, m2 = check_bpm(b.get("segs", []))
            ok = ok1 and ok2
            if not ok:
                all_ok = False
            sample_results.append({"fid": fid, "ok": ok, "notes_check": m1, "bpm_check": m2})
        except Exception as e:
            all_ok = False
            sample_results.append({"fid": fid, "ok": False, "error": str(e)[:100]})
    ok_cnt = sum(1 for r in sample_results if r["ok"])

    report = {
        "list_count": len(all_items),
        "list_page_fails": page_fails,
        "parsed_count": len(parsed_fids),
        "bpm_count": len(bpm_fids),
        "missing_count": len(missing),
        "bpm_missing_fids": bpm_missing[:50],
        "total_notes": total_notes,
        "success_rate": round(len(parsed_fids) / len(all_items), 4) if all_items else 0,
        "sample_checked": len(sample_results),
        "sample_ok": ok_cnt,
        "sample_all_pass": all_ok,
        "sample_details": sample_results,
    }
    with open(os.path.join(OUT, "failed.json"), "w") as f:
        json.dump({"missing": missing_items, "list_page_fails": page_fails}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"== 3/3 汇总 == 列表{len(all_items)} 成功{len(parsed_fids)} 缺失{len(missing)} "
          f"总音符{total_notes} 抽样{ok_cnt}/{len(sample_results)}通过", flush=True)
    if missing:
        print("缺失 fid 前 30:", missing[:30], flush=True)
    if not all_ok:
        print("抽样存在未通过项:", [r for r in sample_results if not r["ok"]], flush=True)
    print("report.json / failed.json 已写入", flush=True)


if __name__ == "__main__":
    main()
