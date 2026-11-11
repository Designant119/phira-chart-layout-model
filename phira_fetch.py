#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phira 谱面抓取+解析：chart id → RPE zip → parsed/{fid}.json + bpm/{fid}.json
用法：python3 phira_fetch.py <chart_id> [--outdir <dir>]   （单谱面烟测）
      python3 phira_fetch.py --fetch-list <page>           （拉一页 /chart 列表，供批量）
"""
import json, os, sys, time, zipfile, io, argparse, math, subprocess

API = "https://api.phira.cn/chart"
BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"


def http_get(url, retries=3, timeout=60, proxy="http://127.0.0.1:8118"):
    """用 curl 下载（显式走本地 8118 tinyproxy，实测连续请求稳定；默认 vortex 代理在
    高频请求下会间歇返回污染响应导致 JSONDecodeError）。返回 bytes。"""
    for i in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-s", "-L", "--max-time", str(timeout), "--proxy", proxy, "-H",
                 "User-Agent: Mozilla/5.0 (chart-dl/1.0)", url],
                capture_output=True, timeout=timeout + 15)
            if out.returncode == 0 and len(out.stdout) > 0:
                return out.stdout
            err = out.stderr.decode()[:200] if out.stderr else "empty"
            if i == retries - 1:
                raise RuntimeError(f"curl fail rc={out.returncode}: {err}")
        except subprocess.TimeoutExpired:
            if i == retries - 1:
                raise RuntimeError(f"curl timeout {url}")
        time.sleep(1.5 * (i + 1))


def rpe_time_to_beat(t, grid_den):
    """RPE time [bar, beat, grid] → 拍。grid 分母由数据推断（4 或 192）。"""
    bar, beat, grid = t[0], t[1], t[2]
    return bar * 4 + beat + grid / float(grid_den)


def parse_chart(chart_json):
    """RPE chart json → (notes[], bpm_segs[])。返回 notes 列表 [{t,x,type,dur,line_id}]。
    t: 秒；x: RPE px 原值；type: 1Tap 2Hold 3Drag 4Flick；dur: 秒。"""
    # BPM segs：[[start_beat, bpm], ...]
    bpm_raw = chart_json.get("BPMList", [])
    # 推断 grid 分母：统计所有 startTime 第 3 位的最大值
    grids = []
    for b in bpm_raw:
        grids += [b["startTime"][2]] if "startTime" in b else []
    for jl in chart_json.get("judgeLineList", []):
        for n in jl.get("notes", []):
            grids += [n["startTime"][2]] if "startTime" in n else []
            if "endTime" in n and isinstance(n["endTime"], list):
                grids += [n["endTime"][2]]
    mx = max(grids) if grids else 3
    # RPE 第3位是格：0-3 四分格 / 0-15 十六分格 / 0-191 192分格
    grid_den = 192 if mx >= 16 else (16 if mx >= 8 else 4)
    segs = []
    for b in bpm_raw:
        beat = rpe_time_to_beat(b["startTime"], grid_den)
        segs.append([beat, float(b["bpm"])])
    segs = sorted(segs, key=lambda s: s[0])

    def beat_to_sec(beat):
        if not segs:
            return beat * 0.5
        # 逐段累计秒
        total_sec = 0.0
        for i, (sb, bpm) in enumerate(segs):
            if i + 1 < len(segs):
                nb = segs[i + 1][0]
                if beat < nb:
                    return total_sec + (beat - sb) * 60.0 / bpm
                total_sec += (nb - sb) * 60.0 / bpm
            else:
                return total_sec + (beat - sb) * 60.0 / bpm
        return total_sec

    notes = []
    for li, jl in enumerate(chart_json.get("judgeLineList", [])):
        for n in jl.get("notes", []):
            if n.get("isFake", 0) == 1:
                continue
            typ = int(n.get("type", 1))
            t_beat = rpe_time_to_beat(n["startTime"], grid_den)
            t = beat_to_sec(t_beat)
            dur = 0.0
            if typ == 2 and "endTime" in n and isinstance(n["endTime"], list):
                e_beat = rpe_time_to_beat(n["endTime"], grid_den)
                dur = max(0.0, beat_to_sec(e_beat) - t)
            notes.append({"t": round(t, 6), "x": float(n.get("positionX", 0.0)),
                          "type": typ, "dur": round(dur, 6), "line_id": li})
    notes.sort(key=lambda n: n["t"])
    return notes, segs


def fetch_chart(chart_id, outdir):
    """下载 zip → 解析 → 写 parsed/{fid}.json + bpm/{fid}.json。fid = chart_id。"""
    # 先从 API 拿该谱面详情（含 file URL）
    data = json.loads(http_get(f"{API}/{chart_id}"))
    file_url = data.get("file")
    if not file_url:
        return False, "无file字段"
    zbytes = http_get(file_url, timeout=60)
    zf = zipfile.ZipFile(io.BytesIO(zbytes))
    # 遍历候选 chart 文件：先试 RPE json，内容非 json 再试 Pec 文本（.pec 常伪装成 .json）
    names = zf.namelist()
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
                notes, segs = parse_chart(chart_json)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chart_id", type=str)
    ap.add_argument("--outdir", type=str, default=os.path.join(BASE, "phira_full"))
    args = ap.parse_args()
    ok, msg = fetch_chart(args.chart_id, args.outdir)
    print(f"chart {args.chart_id}: {'OK' if ok else 'FAIL'} {msg}")
