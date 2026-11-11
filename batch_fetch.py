#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phira 批量采集：页段 [start_page, end_page] → 列表 → 逐条下载 zip → 解析 → 写文件。
- 列表条目已含 file 直链（https://phira.5wyxi.com/files/<uuid>），直接下载，跳过详情请求；
  列表 file 缺失或下载失败时回退 phira_fetch.fetch_chart（详情 API 拿最新 file）。
- 幂等：parsed/{fid}.json 已存在则跳过（支持断点续跑，重跑同一命令即可）。
- 失败重试由 phira_fetch.http_get 内置（3 次退避）；仍失败记 failed_log.txt 继续，不中断。
- 礼貌爬取：item_interval（每条谱面间隔）与 interval（每页间隔），默认 0.5s / 0.8s。

用法：python3 batch_fetch.py --start-page 1 --end-page 84 [--outdir ...] [--item-interval 0.5]
"""
import json, os, sys, time, zipfile, io, argparse

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
sys.path.insert(0, BASE)
import phira_fetch as pf  # 复用已验证的 http_get / parse_chart / fetch_chart（转换逻辑不动）

API_LIST = "https://api.phira.cn/chart"


def fetch_chart_from_file(chart_id, file_url, outdir):
    """与 phira_fetch.fetch_chart 逻辑一致，仅跳过详情请求，直接用列表里的 file 直链。"""
    if not file_url:
        return False, "列表无file字段"
    zbytes = pf.http_get(file_url, timeout=60)
    zf = zipfile.ZipFile(io.BytesIO(zbytes))
    names = zf.namelist()
    # 遍历候选 chart 文件：先试 RPE json，内容非 json 再试 Pec 文本（.pec 常伪装成 .json）
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
                import pec_parse
                notes, segs = pec_parse.parse_pec(raw.decode("utf-8", errors="replace"))
                src = "pec"
        except Exception:
            continue
        if notes:
            break
    if notes is None:
        return False, f"zip内无chart json/pec: {names[:10]}"
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
    return True, f"{len(notes)}音/{len(segs)}段BPM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-page", type=int, required=True)
    ap.add_argument("--end-page", type=int, required=True)
    ap.add_argument("--outdir", type=str, default=os.path.join(BASE, "phira_full"))
    ap.add_argument("--interval", type=float, default=0.8, help="列表页之间间隔(秒)")
    ap.add_argument("--item-interval", type=float, default=0.5, help="每条谱面之间间隔(秒)")
    args = ap.parse_args()

    fail_log = os.path.join(args.outdir, "failed_log.txt")
    ok, fail, skipped, page_fail = 0, 0, 0, 0
    started = time.time()

    with open(fail_log, "a") as fl:
        for page in range(args.start_page, args.end_page + 1):
            url = f"{API_LIST}?page={page}"
            data = None
            try:
                raw = pf.http_get(url, retries=3, timeout=60)  # 内部 3 次重试退避
                data = json.loads(raw)
            except Exception as e:
                page_fail += 1
                fl.write(f"PAGE_FAIL page={page} err={type(e).__name__}:{str(e)[:150]}\n")
                fl.flush()
                print(f"[page {page}] PAGE FAIL: {type(e).__name__} {str(e)[:100]}", flush=True)
                time.sleep(args.interval)
                continue

            results = data.get("results", [])
            for item in results:
                cid = str(item.get("id"))
                if os.path.exists(os.path.join(args.outdir, "parsed", f"{cid}.json")):
                    skipped += 1
                    continue
                ok_flag, msg = False, ""
                # 尝试 1：列表 file 直链
                try:
                    ok_flag, msg = fetch_chart_from_file(cid, item.get("file"), args.outdir)
                except Exception as e:
                    msg = f"file直链失败 {type(e).__name__}:{str(e)[:100]}"
                # 尝试 2：回退详情 API
                if not ok_flag:
                    try:
                        ok2, msg2 = pf.fetch_chart(cid, args.outdir)
                        if ok2:
                            ok_flag = True
                            msg = f"回退详情OK({msg2})"
                        else:
                            msg += f" | 详情失败:{msg2}"
                    except Exception as e:
                        msg += f" | 详情异常 {type(e).__name__}:{str(e)[:100]}"
                if ok_flag:
                    ok += 1
                else:
                    fail += 1
                    fl.write(f"FAIL fid={cid} page={page} err={msg}\n")
                    fl.flush()
                time.sleep(args.item_interval)

            if page % 5 == 0 or page == args.end_page:
                el = int(time.time() - started)
                print(f"[page {page}/{args.end_page}] {el}s ok={ok} fail={fail} skip={skipped}",
                      flush=True)
            time.sleep(args.interval)

    el = int(time.time() - started)
    summary = {"start_page": args.start_page, "end_page": args.end_page,
               "ok": ok, "fail": fail, "skipped": skipped, "page_fail": page_fail,
               "elapsed_sec": el}
    with open(os.path.join(args.outdir, f"summary_{args.start_page}_{args.end_page}.json"),
              "w") as f:
        json.dump(summary, f, indent=1)
    print(f"FINISH pages {args.start_page}-{args.end_page}: ok={ok} fail={fail} "
          f"skip={skipped} page_fail={page_fail} elapsed={el}s", flush=True)


if __name__ == "__main__":
    main()
