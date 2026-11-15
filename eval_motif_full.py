#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量 motif 检索评估：节奏粗筛 + 接缝/手位/类型重排，对比零基线。"""
import os, json, glob, argparse, itertools, time
import numpy as np
from build_motif_index import line_motifs, L

X_SCALE = 675.0

def build_targets(root, split, fids, rng, limit):
    out = []
    for fid in fids:
        p = os.path.join(root, "data_fixed", "parsed", f"{fid}.json")
        if not os.path.exists(p): continue
        notes = json.load(open(p)).get("notes", [])
        if not notes: continue
        for lid in sorted({int(n.get("line_id", 0)) for n in notes}):
            ns = [n for n in notes if int(n.get("line_id", 0)) == lid and int(n.get("type", 1)) in (1,2,3,4)]
            ns.sort(key=lambda n: n["t"])
            if len(ns) < L: continue
            x = np.clip(np.array([q["x"] for q in ns], dtype=np.float32) / X_SCALE, -1.2, 1.2)
            for m in line_motifs(notes, lid, fid):
                s = m[7]; n = len(ns)
                out.append({
                    "fid": str(fid), "line": int(lid), "start": int(s),
                    "q": m[0], "x": m[1], "r": m[2], "t": m[3],
                    "prev": float(x[s-1]) if s > 0 else np.nan,
                    "next": float(x[s+L]) if s + L < n else np.nan,
                    "ctx_med": float(np.median(x[max(0,s-16):s])) if s > 0 else float(x[s]),
                })
    if limit and len(out) > limit:
        idx = rng.choice(len(out), limit, replace=False)
        out = [out[i] for i in idx]
    return out

def gather_candidates(parts, targets, mt=100, mp=100, mf=600, ctxw=4.0):
    """按节奏取每张 target 的候选，返回每个 target 的 components。"""
    res = []
    tq = np.stack([t["q"] for t in targets]).astype(np.float32)
    tq /= np.linalg.norm(tq, axis=1, keepdims=True) + 1e-8
    for b0 in range(0, len(targets), mt):
        b1 = min(len(targets), b0 + mt)
        cand = [[] for _ in range(b1 - b0)]  # dict arrays per target
        for pf in parts:
            d = np.load(pf, allow_pickle=True)
            Q = d["Q"].astype(np.float32)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8
            sims = Q @ tq[b0:b1].T  # (n, b)
            fid_arr = d["fid"].astype(str)
            x0 = d["X"][:, 0]; mx = d["X"].mean(axis=1)
            for j in range(b1 - b0):
                tf = targets[b0 + j]["fid"]
                valid = (fid_arr != tf)
                t = targets[b0 + j]
                ctx = np.abs(mx - t["ctx_med"]).astype(np.float32)
                sv = sims[:, j].copy() - ctxw * ctx
                sv[~valid] = -9
                k = min(mp, int(valid.sum()))
                if k <= 0: continue
                idx = np.argpartition(-sv, k-1)[:k]
                cand[j].append({
                    "sim": sims[idx, j].astype(np.float32),
                    "rank": sv[idx].astype(np.float32),
                    "ctx": ctx[idx].astype(np.float32),
                    "R": d["R"][idx].astype(np.float32),
                    "X": d["X"][idx].astype(np.float32),
                    "T": d["T"][idx].astype(np.int32),
                    "fid": d["fid"][idx].astype(str),
                })
        for j in range(b1 - b0):
            cs = cand[j]
            if not cs: continue
            sim = np.concatenate([x["sim"] for x in cs])
            rank = np.concatenate([x["rank"] for x in cs])
            ctx = np.concatenate([x["ctx"] for x in cs])
            R = np.concatenate([x["R"] for x in cs])
            X = np.concatenate([x["X"] for x in cs])
            T = np.concatenate([x["T"] for x in cs])
            fids = np.concatenate([x["fid"] for x in cs])
            # 全局按节奏取前 mf
            if len(sim) > mf:
                idx = np.argpartition(-rank, mf-1)[:mf]
                sim, ctx, R, X, T, fids = sim[idx], ctx[idx], R[idx], X[idx], T[idx], fids[idx]
            t = targets[b0 + j]
            prev, nxt = t["prev"], t["next"]
            r0, rl = R[:, 0], R[:, -1]
            anchor = t["ctx_med"] - R.mean(axis=1)
            pred = R + anchor[:, None]
            seam = ctx
            jump = np.mean(np.abs(np.diff(pred, axis=1)), axis=1)
            span = np.max(pred, axis=1) - np.min(pred, axis=1)
            type_pen = np.mean(T != t["t"][None, :], axis=1)
            res.append({"sim": sim, "R": R, "X": X, "T": T, "pred": pred, "ctx": ctx,
                        "seam": seam, "jump": jump, "span": span, "type_pen": type_pen,
                        "target": t})
        print(f"  candidates {b1}/{len(targets)}", flush=True)
    return res

def score_one(c, w):
    return (w[0] * c["sim"] - w[1] * c["type_pen"] - w[2] * c["ctx"]
            - w[3] * c["jump"] - w[4] * c["span"])

def eval_rows(rows, w, topn=1):
    maes = []; abses = []; wins = 0
    for c in rows:
        t = c["target"]; score = score_one(c, w)
        if topn == 1:
            j = int(np.argmax(score)); pred = c["pred"][j]
        else:
            js = np.argsort(-score)[:topn]
            pred = np.mean([c["pred"][j] for j in js], axis=0)
        rel = float(np.mean(np.abs((pred - pred[0]) - t["r"])))
        ab = float(np.mean(np.abs(pred - t["x"])))
        base = float(np.mean(np.abs(t["r"])))
        maes.append(rel); abses.append(ab); wins += rel < base
    return np.mean(maes), np.mean(abses), wins / max(len(maes), 1), np.mean([np.mean(np.abs(c["target"]["r"])) for c in rows])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--root", default="/home/user/Doubao/chats/38453392661469954/_run_real_traj")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--ctxw", type=float, default=4.0)
    a = ap.parse_args()
    parts = sorted(glob.glob(os.path.join(a.parts, "part_*.npz")))
    split = json.load(open(os.path.join(a.root, "train_data_traj_fixed", "split.json")))
    rng = np.random.default_rng(0)
    val = build_targets(a.root, "val", split["val"], rng, a.n_val)
    test = build_targets(a.root, "test", split["test"], rng, a.n_test)
    print("targets", len(val), len(test), flush=True)
    t0 = time.time()
    vrows = gather_candidates(parts, val, ctxw=a.ctxw)
    print("val candidates", time.time()-t0, flush=True)
    grid = list(itertools.product([0.5, 1.0, 1.5, 2.0], [0, 0.5, 1.0, 2.0],
                                  [0, 1, 2, 4, 8], [0, 0.1, 0.3, 0.8, 1.5], [0, 0.1, 0.3]))
    best = None
    for w in grid:
        res = eval_rows(vrows, w)
        mae = res[1]
        if best is None or mae < best[0]: best = (mae, w)
    print("best val", best, flush=True)
    t1 = time.time()
    trows = gather_candidates(parts, test, ctxw=a.ctxw)
    print("test candidates", time.time()-t1, flush=True)
    for topn in (1, 3):
        print("TEST top", topn, eval_rows(trows, best[1], topn), flush=True)
    json.dump({"weights": best[1], "val": best[0],
               "test": {"top1": eval_rows(trows, best[1], 1), "top3": eval_rows(trows, best[1], 3)}},
              open(os.path.join(a.parts, "eval_full.json"), "w"), ensure_ascii=False, indent=1, default=float)

if __name__ == "__main__": main()
