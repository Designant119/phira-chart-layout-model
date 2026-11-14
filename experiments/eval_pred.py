#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对已有 pred 目录算双押/单音逐谱 MAE + 覆盖率（与 eval_any 同口径）。"""
import json, os, sys
import numpy as np
REPO = "/home/user/Doubao/chats/38453392661469954/phira-chart-layout-model"
RUN = "/home/user/Doubao/chats/38453392661469954/_run_real_traj"
tag, pred_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
src = open(os.path.join(REPO, "eval_traj_full.py")).read().split("def show(")[0]
ns = {}; exec(compile(src, "eval_traj_full.py", "exec"), ns); mc = ns["mae_compare"]
test = json.load(open(os.path.join(RUN, "npz_prevx/split.json")))["test"]
rows = []
for fid in test:
    rows.append(mc(fid, os.path.join(pred_dir, f"{fid}.json"), os.path.join(RUN, "data/parsed", f"{fid}.json")))


def agg(k):
    v = [r[k] for r in rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return float(np.mean(v)), float(np.median(v)), float(np.percentile(v, 90))


print(tag, "双押 %.2f/%.2f/%.2f" % agg("dbl_mae"))
print(tag, "单音 %.2f/%.2f/%.2f" % agg("sgl_mae"))
tot = sum(json.load(open(os.path.join(pred_dir, f"{f}.json")))["n_notes"] for f in test)
mod = sum(json.load(open(os.path.join(pred_dir, f"{f}.json")))["n_model"] for f in test)
print(tag, "覆盖率 %.2f%%" % (mod / tot * 100))
json.dump({r["fid"]: {k: v for k, v in r.items() if k != "by_src"} for r in rows},
          open(os.path.join(RUN, f"report/eval_{tag}.json"), "w"), indent=1)
