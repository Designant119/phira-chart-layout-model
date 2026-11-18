#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从角度过滤后的原始训练谱面里随机抽样，画排键预览图。"""
import os, glob, json, random, argparse, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    1: "#42a5f5",  # Tap 蓝
    2: "#4dd0e1",  # Hold 青
    3: "#ffd54f",  # Drag 黄
    4: "#f06292",  # Flick 粉
}
NAMES = {1: "Tap", 2: "Hold", 3: "Drag", 4: "Flick"}


def load_title_map(zips):
    m = {}
    for z in zips:
        b = os.path.basename(z)
        fid = b.split("_", 1)[0]
        if fid.isdigit():
            m[fid] = b.rsplit(".zip", 1)[0]
    return m


def candidate(path, mainline=None):
    d = json.load(open(path, encoding="utf-8"))
    fid = d.get("fid") or os.path.splitext(os.path.basename(path))[0]
    notes = d.get("notes", [])
    if mainline and fid in mainline:
        lid = int(mainline[fid])
        notes = [n for n in notes if int(n.get("line_id", 0)) == lid]
    d["notes"] = notes
    if len(notes) < 150 or len(notes) > 1500:
        return None
    c = collections.Counter(int(n["type"]) for n in notes)
    if c[1] < 30 or c[2] < 3 or c[3] < 3 or c[4] < 3:
        return None
    if max(float(n["t"]) for n in notes) < 30:
        return None
    return d, c


def plot_chart(ax, d, title, counts):
    notes = sorted(d["notes"], key=lambda n: n["t"])
    xs = [float(n["x"]) for n in notes]
    ts = [float(n["t"]) for n in notes]
    ax.set_xlim(0, max(ts) + 2)
    ax.set_ylim(-700, 700)
    ax.axhline(0, color="#888888", lw=0.8, alpha=0.5)

    for n in notes:
        t, x, typ, dur = float(n["t"]), float(n["x"]), int(n["type"]), float(n.get("dur", 0))
        if typ == 2:
            ax.hlines(x, t, t + max(dur, 0.05), color=COLORS[2], lw=3, alpha=0.85)
        elif typ == 1:
            ax.scatter(t, x, color=COLORS[1], s=12, alpha=0.85)
        elif typ == 3:
            ax.scatter(t, x, color=COLORS[3], marker="D", s=13, alpha=0.9)
        elif typ == 4:
            ax.scatter(t, x, color=COLORS[4], marker="^", s=16, alpha=0.95)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("时间/秒", fontsize=8)
    ax.set_ylabel("相对位置", fontsize=8)
    ax.tick_params(labelsize=7)
    info = " ".join(f"{NAMES[k]}:{counts[k]}" for k in [1, 2, 3, 4])
    ax.text(0.01, 0.03, info, transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--zips", required=True)
    ap.add_argument("--mainline", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    title_map = load_title_map(glob.glob(os.path.join(a.zips, "**", "*.zip"), recursive=True))
    mainline = json.load(open(a.mainline, encoding="utf-8")) if a.mainline else None
    files = sorted(glob.glob(os.path.join(a.parsed, "*.json")))

    pool = []
    for p in files:
        got = candidate(p, mainline)
        if got:
            pool.append((p, *got))

    random.seed(a.seed)
    random.shuffle(pool)
    picked = pool[:a.n]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.ravel()
    meta = []
    for i, (p, d, counts) in enumerate(picked):
        fid = d.get("fid") or os.path.splitext(os.path.basename(p))[0]
        title = title_map.get(fid, fid)
        short = title if len(title) <= 46 else title[:45] + "…"
        ax = axes[i]
        plot_chart(ax, d, short, counts)
        json.dump(d, open(os.path.join(a.out, f"{fid}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        meta.append({"fid": fid, "title": title, "counts": dict(counts),
                     "notes": len(d["notes"]), "duration": max(float(n["t"]) for n in d["notes"])})

    for j in range(len(picked), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    sheet = os.path.join(a.out, "training_sample_contact.jpg")
    fig.savefig(sheet, dpi=150, bbox_inches="tight")
    json.dump(meta, open(os.path.join(a.out, "sample_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("pool", len(pool), "picked", len(picked), "sheet", sheet)


if __name__ == "__main__":
    main()
