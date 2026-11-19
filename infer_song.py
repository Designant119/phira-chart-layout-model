#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整曲 SongNet 推理：读 angle_parsed 谱面，按训练口径算 17 维特征，整曲前向出 x。
开卷版：prev/同伴特征先用真值 x 算，验证模型信息齐全时的上限手感。
输出 {fid, notes:[{t,type,x(px,中心0),line_id,dur}]}。"""
import json, os, argparse, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_song as T
from build_song_data import parse_chart, feats, X_SCALE

PARSED = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/angle_all/angle_parsed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fid")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy", action="store_true")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    net = T.SongNet(nf=ck.get("nf", 17), hidden=ck.get("hidden", 64), layers=ck.get("layers", 1))
    net.load_state_dict(ck["model"]); net.eval()

    d = json.load(open(os.path.join(PARSED, f"{a.fid}.json")))
    keep = parse_chart(d)
    if keep is None:
        keep = sorted(d["notes"], key=lambda n: float(n["t"]))[:2048]
    f, x_true = feats(keep)
    ft = torch.from_numpy(f.astype(np.float32)).unsqueeze(0)
    lens = torch.tensor([len(keep)])
    with torch.no_grad():
        lg = net(ft, lens)[0].numpy()
    pred = lg.argmax(-1)
    x_pred = pred / (T.NB - 1) * 2 - 1

    out = []
    for j, n in enumerate(keep):
        out.append({"t": float(n["t"]), "type": int(n.get("type", 1)),
                    "x": float(x_pred[j] * X_SCALE), "line_id": int(n.get("line_id", 0)),
                    "dur": float(n.get("dur", 0.0))})
    out.sort(key=lambda n: (n["line_id"], n["t"]))
    json.dump({"fid": a.fid, "notes": out}, open(a.out, "w"), ensure_ascii=False)
    print("notes", len(out), "x 范围 %.0f ~ %.0f" % (min(o["x"] for o in out), max(o["x"] for o in out)))


if __name__ == "__main__":
    main()
