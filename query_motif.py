#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按节奏窗口检索 Phira motif：给一个同 fid/line/start 的窗口，找最像的人类 pattern。"""
import os, json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--fid", required=True)
    ap.add_argument("--line", type=int, default=-1)
    ap.add_argument("--start", type=int, default=-1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = np.load(os.path.join(a.index, "motifs.npz"), allow_pickle=True)
    Q, X, R, T = d["Q"], d["X"], d["R"], d["T"]
    fids, lines, starts = d["fid"], d["line"], d["start"]
    idx = np.where(fids == a.fid)[0]
    if a.line >= 0: idx = idx[lines[idx] == a.line]
    if a.start >= 0: idx = idx[starts[idx] == a.start]
    if not len(idx):
        raise SystemExit("没有匹配的 motif，检查 fid/line/start")
    qi = int(idx[len(idx) // 2])  # 没指定就取中间那段
    if a.start >= 0: qi = int(idx[0])
    q = Q[qi] / (np.linalg.norm(Q[qi]) + 1e-8)
    sim = Q @ q / (np.linalg.norm(Q, axis=1) + 1e-8)
    sim[qi] = -9
    top = np.argsort(-sim)[:a.topk]
    rows = []
    for j in top:
        rows.append({
            "fid": str(fids[j]), "line": int(lines[j]), "start": int(starts[j]),
            "sim": round(float(sim[j]), 4),
            "x": [round(float(v * 675.0), 1) for v in X[j]],
            "types": [int(v) for v in T[j]],
        })
    out = {"query": {"fid": a.fid, "line": int(lines[qi]), "start": int(starts[qi])},
           "matches": rows}
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
