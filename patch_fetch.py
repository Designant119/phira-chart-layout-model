#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补采 v2：重试主采集中失败/漏采的谱面。
改进（相对 v1）：
1) .pec 支持：zip 内无 RPE json 时用 pec_parse 解析 Phigros legacy 文本格式
2) 限流防护：每条间隔 1.0s；连续 8 条失败 → 暂停 60s；Extra data(限流迹象) 首次失败后等 30s 再试 1 次
3) 顺序：先 PAGE_FAIL 整页漏采（列表正常，成功率高），再 FAIL fid
幂等：parsed/{fid}.json 已存在则跳过；仍失败写 failed_patch.txt
用法：python3 patch_fetch.py
"""
import json, os, sys, re, time, zipfile, io

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
sys.path.insert(0, BASE)
import phira_fetch as pf
import pec_parse

OUT = os.path.join(BASE, "phira_full")
API_LIST = "https://api.phira.cn/chart"
ITEM_INTERVAL = 1.5     # 每条间隔（正常窗口内保持节奏）
THROTTLE_AT = 25        # 每处理 THROTTLE_AT 条强制休息
THROTTLE_SEC = 45
CONSEC_SLEEP_AT = 5     # 连续失败阈值（快速识别故障窗口）
CONSEC_SLEEP_SEC = 600  # 连续失败暂停（故障窗口 10min）
EXTRA_RETRY_WAIT = 30   # Extra data 类二次尝试等待
ABORT_FAIL_AT = 15      # 一轮内累计失败达此数且零成功 → 判定故障窗口，退出本轮


def fetch_chart_v2(chart_id, outdir, file_url=None):
    """file直链(zip) → RPE json 或 Pec 文本 → parsed/{fid}.json + bpm/{fid}.json。
    file_url 来自列表 API 映射（绕开有故障窗口的详情 API）；缺省时回退详情 API。"""
    if not file_url:
        data = json.loads(pf.http_get(f"{pf.API}/{chart_id}", timeout=60))
        file_url = data.get("file")
    if not file_url:
        return False, "无file字段"
    zbytes = pf.http_get(file_url, timeout=60)
    zf = zipfile.ZipFile(io.BytesIO(zbytes))
    names = zf.namelist()
    # 遍历候选 chart 文件：先试 RPE json，失败（或内容非 json）再试 Pec 文本
    notes, segs, src = None, None, None
    for nm in names:
        base = nm.rsplit("/", 1)[-1].lower()
        if not (base.endswith(".json") or base.endswith(".pec")):
            continue
        if base == "info.json" or base == "info.pec":
            continue
        raw = zf.read(nm)
        head = raw.lstrip()[:8]
        try:
            if head.startswith(b"{"):
                chart_json = json.loads(raw)
                notes, segs = pf.parse_chart(chart_json)
                src = "rpe"
            else:
                notes, segs = pec_parse.parse_pec(raw.decode("utf-8", errors="replace"))
                src = "pec"
        except Exception:
            continue
        if notes:
            break
    if notes is None:
        return False, f"zip内无chart json/pec: {names[:8]}"
    if not notes:
        return False, "解析后无notes"
    fid = str(chart_id)
    pd = os.path.join(outdir, "parsed")
    bd = os.path.join(outdir, "bpm")
    os.makedirs(pd, exist_ok=True)
    os.makedirs(bd, exist_ok=True)
    json.dump({"fid": fid, "notes": notes}, open(os.path.join(pd, f"{fid}.json"), "w"),
              ensure_ascii=False)
    json.dump({"segs": segs}, open(os.path.join(bd, f"{fid}.json"), "w"), ensure_ascii=False)
    return True, f"{src} {len(notes)}音/{len(segs)}段BPM"


def collect_failed():
    """返回 (fail_fids, page_fail_pages)。跳过 failed_permanent.txt 中的永久失败 fid。"""
    fids, pages = [], []
    seen = set()
    perm = set()
    perm_path = os.path.join(OUT, "failed_permanent.txt")
    if os.path.exists(perm_path):
        perm = set(l.strip() for l in open(perm_path) if l.strip())
    logp = os.path.join(OUT, "failed_log.txt")
    if not os.path.exists(logp):
        return fids, pages
    for line in open(logp, encoding="utf-8"):
        m = re.search(r"FAIL fid=(\d+)", line)
        if m and m.group(1) not in seen and m.group(1) not in perm:
            seen.add(m.group(1))
            fids.append(m.group(1))
        m = re.search(r"PAGE_FAIL page=(\d+)", line)
        if m:
            p = int(m.group(1))
            if p not in pages:
                pages.append(p)
    return fids, pages


PERMANENT_MARKS = ("解析后无notes", "zip内无chart json/pec", "zip内无chart json")


def fids_missing_on_pages(pages):
    """重拉这些页的列表，返回页内 parsed 缺失的 fid。"""
    out = []
    for p in sorted(pages):
        try:
            raw = pf.http_get(f"{API_LIST}?page={p}", retries=3, timeout=60)
            data = json.loads(raw)
            for it in data.get("results", []):
                fid = str(it["id"])
                if not os.path.exists(os.path.join(OUT, "parsed", f"{fid}.json")):
                    out.append(fid)
        except Exception as e:
            print(f"PAGE_FAIL 重拉 page={p} 仍失败: {str(e)[:100]}", flush=True)
        time.sleep(0.8)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runtime", type=int, default=0,
                    help="单轮最长运行秒数，超时正常退出（配合幂等续跑，抵御沙箱重启）")
    args = ap.parse_args()

    fail_fids, page_fails = collect_failed()
    print(f"failed_log: FAIL fid {len(fail_fids)} 个, PAGE_FAIL 页 {len(page_fails)} 个", flush=True)
    # 列表 API 映射（fid→file 直链），绕开详情 API 故障窗口
    file_map = {}
    map_path = os.path.join(OUT, "chart_list.json")
    if os.path.exists(map_path):
        file_map = json.load(open(map_path))
        print(f"chart_list.json 映射 {len(file_map)} 个 fid→file", flush=True)
    page_fids = fids_missing_on_pages(page_fails)
    print(f"PAGE_FAIL 页内 parsed 缺失 {len(page_fids)} 个", flush=True)
    fids, seen = [], set()
    for f in page_fids + fail_fids:  # PAGE_FAIL 页优先
        if f not in seen:
            seen.add(f)
            fids.append(f)
    print(f"合计待补采 {len(fids)} 个（先 PAGE_FAIL 后 FAIL）", flush=True)

    started = time.time()
    ok, fail, skip = 0, 0, 0
    consec_fail = 0
    still_fail = []
    runtime_hit = False
    with open(os.path.join(OUT, "failed_patch.txt"), "w") as fl:
        for i, fid in enumerate(fids, 1):
            if args.max_runtime and time.time() - started > args.max_runtime:
                runtime_hit = True
                print(f"RUNTIME_LIMIT 到达（{args.max_runtime}s），正常退出待续跑", flush=True)
                break
            if os.path.exists(os.path.join(OUT, "parsed", f"{fid}.json")):
                skip += 1
                continue
            ok_flag, msg = False, ""
            # 尝试 1（优先列表直链，零详情请求）
            try:
                ok_flag, msg = fetch_chart_v2(fid, OUT, file_url=file_map.get(fid))
            except json.JSONDecodeError as e:
                # 限流迹象：等 30s 再试 1 次
                time.sleep(EXTRA_RETRY_WAIT)
                try:
                    ok_flag, msg = fetch_chart_v2(fid, OUT, file_url=file_map.get(fid))
                    if ok_flag:
                        msg = f"(重试后){msg}"
                except Exception as e2:
                    msg = f"限流类两次失败 {type(e2).__name__}:{str(e2)[:100]}"
            except Exception as e:
                msg = f"{type(e).__name__}:{str(e)[:120]}"
            if ok_flag:
                ok += 1
                consec_fail = 0
            else:
                fail += 1
                still_fail.append(fid)
                fl.write(f"FAIL fid={fid} err={msg}\n")
                fl.flush()
                # 数据本身问题（空谱/无谱面文件）→ 永久失败，不计入连续失败（不触发暂停）
                permanent = any(mark in msg for mark in PERMANENT_MARKS)
                if permanent:
                    with open(os.path.join(OUT, "failed_permanent.txt"), "a") as pf_:
                        pf_.write(f"{fid} {msg}\n")
                    continue
                consec_fail += 1
                if consec_fail >= CONSEC_SLEEP_AT:
                    print(f"  连续 {consec_fail} 条失败，暂停 {CONSEC_SLEEP_SEC}s 让 API 喘息", flush=True)
                    time.sleep(CONSEC_SLEEP_SEC)
                    consec_fail = 0
                if fail >= ABORT_FAIL_AT and ok == 0:
                    print(f"  本轮已连续失败 {fail} 条且零成功，判定故障窗口，退出本轮待下轮", flush=True)
                    runtime_hit = True
                    break
            if i % 10 == 0:
                print(f"  {i}/{len(fids)} ok={ok} fail={fail} skip={skip}", flush=True)
            time.sleep(ITEM_INTERVAL)
            if i % THROTTLE_AT == 0:
                time.sleep(THROTTLE_SEC)
    tag = "PARTIAL" if runtime_hit else "FINISH"
    print(f"PATCH {tag}: 重试{len(fids)} ok={ok} fail={fail} skip={skip}", flush=True)
    print(f"仍失败 {len(still_fail)} 个:", still_fail, flush=True)


if __name__ == "__main__":
    main()
