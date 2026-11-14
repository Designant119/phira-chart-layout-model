#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理结果"明显问题"可计数判据（阈值显式可调，不改仓库代码）。

输入：infer_traj.py 产出的预测 json（目录 <fid>.json 或单文件）；
     --root 给数据根时额外对齐原谱 orig 做"交叉"判定，缺省退化为纯预测单调性检查。

判据定义（单位 px；x 合法域 [-675, 675]）：
(a) 左右乱跳（per line，按时间排序）
    - 方向反转次数：相邻两段位移 dx_i = x_i - x_{i-1}，sign(dx_i) 与 sign(dx_{i-1})
      均非零且相反，记一次反转（来回横跳）。
    - 最大跳变距离：max |dx_i|，并给出对应相邻音符。
(b) 贴边堆叠
    - 贴边音符数：|x| >= --bound-px（默认 660，即距 675 边界 15px 以内）。
    - 堆叠对数：同一 line_id 内时间相邻、|Δx| <= --stack-px（默认 2px）的音对。
(c) 同线交叉
    - 同一 line_id 内按时间相邻的两个音符，预测 x 相对原谱 x 的走向反序：
      orig dx 与 pred dx 均非零且符号相反，记一次交叉。
    - 无 orig 时退化为：pred dx < 0（相对前一音左移）即记一次反序，结果在
      `degraded_mode` 中标注，仅供粗筛。

每谱输出计数，并给 1~2 个具体示例（fid、line_id、时间点、前后 x 数值）。

用法示例：
  python3 problem_metrics.py --pred-dir /path/preds --fids /path/split.json \
      --root /path/real_root --out /path/problems.json
  python3 problem_metrics.py --pred-dir /tmp/traj_smoke_pred --fids smoke02 \
      --out /path/problems.json --bound-px 660 --stack-px 2 --examples 2
"""
import argparse
import json
import os
from collections import defaultdict

R3 = lambda t: round(float(t), 3)


def sign(v):
    if v > 1e-9:
        return 1
    if v < -1e-9:
        return -1
    return 0


def load_notes(p):
    return json.load(open(p, encoding="utf-8"))["notes"]


def orig_index(root, fid):
    """orig -> {(t3, line_id): x}；缺文件返回 None。"""
    if not root:
        return None
    p = os.path.join(root, "parsed", f"{fid}.json")
    if not os.path.exists(p):
        return None
    idx = {}
    for n in load_notes(p):
        idx[(R3(n["t"]), n.get("line_id", -1))] = float(n["x"])
    return idx


def analyze_line(lnotes, orig_idx, bound_px, stack_px):
    """单条 line 的判据统计。lnotes 已按 t 排序。"""
    res = {
        "reversal_count": 0,
        "max_jump_px": 0.0,
        "max_jump_pair": None,
        "boundary_notes": 0,
        "stack_pairs": 0,
        "cross_count": 0,
        "events": {"jump": [], "boundary": [], "stack": [], "cross": []},
    }
    prev_sign = 0
    for i, n in enumerate(lnotes):
        x = float(n["x"])
        if abs(x) >= bound_px:
            res["boundary_notes"] += 1
            res["events"]["boundary"].append({
                "t": n["t"], "x": round(x, 2),
                "note": {"type": n.get("type", 1), "line_id": n.get("line_id", -1)},
            })
        if i == 0:
            continue
        p = lnotes[i - 1]
        px = float(p["x"])
        dx = x - px
        adx = abs(dx)

        if adx > res["max_jump_px"]:
            res["max_jump_px"] = adx
            res["max_jump_pair"] = {
                "t_prev": p["t"], "t": n["t"],
                "x_prev": round(px, 2), "x": round(x, 2), "dx": round(dx, 2),
            }
        s = sign(dx)
        if prev_sign != 0 and s != 0 and s != prev_sign:
            res["reversal_count"] += 1
            res["events"]["jump"].append({
                "t_prev": p["t"], "t": n["t"],
                "x_prev": round(px, 2), "x": round(x, 2),
                "prev_dx_sign": prev_sign, "dx_sign": s,
            })
        prev_sign = s

        if adx <= stack_px:
            res["stack_pairs"] += 1
            res["events"]["stack"].append({
                "t_prev": p["t"], "t": n["t"],
                "x_prev": round(px, 2), "x": round(x, 2), "dx": round(dx, 2),
            })

        # 同线交叉：对齐 orig 走向
        if orig_idx is not None:
            ox_prev = orig_idx.get((R3(p["t"]), p.get("line_id", -1)))
            ox = orig_idx.get((R3(n["t"]), n.get("line_id", -1)))
            if ox_prev is not None and ox is not None:
                odx = ox - ox_prev
                if sign(odx) != 0 and sign(dx) != 0 and sign(odx) != sign(dx):
                    res["cross_count"] += 1
                    res["events"]["cross"].append({
                        "t_prev": p["t"], "t": n["t"],
                        "pred_x_prev": round(px, 2), "pred_x": round(x, 2),
                        "orig_x_prev": round(ox_prev, 2), "orig_x": round(ox, 2),
                    })
        else:
            # 退化模式：pred 左移即记反序
            if dx < -1e-9:
                res["cross_count"] += 1
                res["events"]["cross"].append({
                    "t_prev": p["t"], "t": n["t"],
                    "pred_x_prev": round(px, 2), "pred_x": round(x, 2),
                    "note": "degraded: 无 orig，按 pred 左移计数",
                })
    return res


def main():
    ap = argparse.ArgumentParser(description="推理结果问题判据统计")
    ap.add_argument("--pred-dir", required=True, help="预测 json 目录（<fid>.json）")
    ap.add_argument("--fids", required=True, help="fid 逗号列表或 split.json 路径")
    ap.add_argument("--root", default=None, help="原谱数据根（可选，用于交叉判定）")
    ap.add_argument("--split-key", default="test")
    ap.add_argument("--out", required=True, help="输出 json（同名 .md 侧车）")
    ap.add_argument("--bound-px", type=float, default=660.0,
                    help="贴边阈值 |x|>=该值（默认 660，合法域 ±675）")
    ap.add_argument("--stack-px", type=float, default=2.0,
                    help="堆叠阈值：相邻 |Δx|<=该值（px，默认 2）")
    ap.add_argument("--examples", type=int, default=2, help="每类示例最多保留几条")
    args = ap.parse_args()

    if os.path.isfile(args.fids):
        split = json.load(open(args.fids, encoding="utf-8"))
        fids = list(split[args.split_key])
    else:
        fids = [f.strip() for f in args.fids.split(",") if f.strip()]

    per_fid = []
    totals = defaultdict(float)
    degraded = False

    for fid in fids:
        pred_p = os.path.join(args.pred_dir, f"{fid}.json")
        notes = load_notes(pred_p)
        orig_idx = orig_index(args.root, fid)
        if orig_idx is None:
            degraded = True

        by_line = defaultdict(list)
        for n in notes:
            by_line[n.get("line_id", -1)].append(n)

        fid_res = {"fid": fid, "n_notes": len(notes), "n_lines": len(by_line),
                   "reversal_count": 0, "max_jump_px": 0.0, "max_jump_pair": None,
                   "boundary_notes": 0, "stack_pairs": 0, "cross_count": 0,
                   "examples": {}}
        all_events = defaultdict(list)

        for lid, lnotes in by_line.items():
            lnotes.sort(key=lambda n: n["t"])
            r = analyze_line(lnotes, orig_idx, args.bound_px, args.stack_px)
            fid_res["reversal_count"] += r["reversal_count"]
            fid_res["boundary_notes"] += r["boundary_notes"]
            fid_res["stack_pairs"] += r["stack_pairs"]
            fid_res["cross_count"] += r["cross_count"]
            if r["max_jump_px"] > fid_res["max_jump_px"]:
                fid_res["max_jump_px"] = r["max_jump_px"]
                fid_res["max_jump_pair"] = r["max_jump_pair"]
            for k, evs in r["events"].items():
                all_events[k].extend(evs)

        for k in ("jump", "boundary", "stack", "cross"):
            fid_res["examples"][k] = all_events[k][: args.examples]

        for k in ("reversal_count", "boundary_notes", "stack_pairs", "cross_count"):
            totals[k] += fid_res[k]
        totals["max_jump_px"] = max(totals["max_jump_px"], fid_res["max_jump_px"])
        per_fid.append(fid_res)

    summary = {
        "thresholds": {"bound_px": args.bound_px, "stack_px": args.stack_px},
        "degraded_mode_no_orig": degraded,
        "totals": dict(totals),
    }
    result = {"fids": fids, "per_fid": per_fid, "summary": summary}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md = render_md(per_fid, summary)
    md_p = os.path.splitext(args.out)[0] + ".md"
    with open(md_p, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\n[json] {args.out}\n[md]   {md_p}")


def render_md(per_fid, summary):
    t = summary["totals"]
    lines = ["| fid | 音符数 | 方向反转 | 最大跳变px | 贴边音符 | 堆叠对 | 交叉次数 |",
             "|---|---|---|---|---|---|---|"]
    for r in per_fid:
        lines.append(
            f"| {r['fid']} | {r['n_notes']} | {r['reversal_count']} "
            f"| {r['max_jump_px']:.2f} | {r['boundary_notes']} "
            f"| {r['stack_pairs']} | {r['cross_count']} |")
    lines.append("")
    lines.append(f"## 汇总（阈值 贴边≥{summary['thresholds']['bound_px']}px，"
                 f"堆叠≤{summary['thresholds']['stack_px']}px）")
    lines.append(f"- 方向反转 {t['reversal_count']}；最大跳变 {t['max_jump_px']:.2f}px；"
                 f"贴边 {t['boundary_notes']}；堆叠对 {t['stack_pairs']}；交叉 {t['cross_count']}")
    if summary["degraded_mode_no_orig"]:
        lines.append("- 注意：未对齐 orig，交叉计数为退化模式（pred 左移粗筛）。")
    # 示例摘录
    lines.append("\n## 示例（每谱前 N 条，详见 json）")
    for r in per_fid:
        ex = r["examples"]
        if ex["cross"]:
            c = ex["cross"][0]
            lines.append(f"- {r['fid']} 交叉例: t={c.get('t_prev')}->{c.get('t')}, "
                         f"pred {c.get('pred_x_prev')}->{c.get('pred_x')}, "
                         f"orig {c.get('orig_x_prev')}->{c.get('orig_x')}")
        if ex["stack"]:
            s = ex["stack"][0]
            lines.append(f"- {r['fid']} 堆叠例: t={s['t_prev']}->{s['t']}, "
                         f"x {s['x_prev']}->{s['x']}, Δx={s['dx']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
