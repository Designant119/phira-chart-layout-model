#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量排键图（RPE 风格，不依赖 pez）：python3 viz_plain.py <fid> <infer.json> <tag>
用 bpm/{fid}.json 首段 BPM；取前 N 小节窗口。"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
FONT = font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
NOTE_W = 178.0  # 音符条宽 px（定稿样式）
XMAX = 675.0


def main():
    fid, infer_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    notes = json.load(open(infer_path))['notes']
    bpm = json.load(open(os.path.join(BASE, 'bpm', f'{fid}.json')))['segs'][0][1]
    bps = bpm / 60.0
    n_bars = 3
    win = n_bars * 4.0 / bps
    beat_end = max(n['t'] for n in notes) * bps
    # 自动选最密 3 小节窗口
    best_n, t0 = -1, 0.0
    g = 0.0
    while g < beat_end:
        a, b = g / bps, g / bps + win
        c = sum(1 for n in notes if a <= n['t'] < b)
        if c > best_n:
            best_n, t0 = c, a
        g += 0.25
    t1 = t0 + win
    sel = [n for n in notes if t0 <= n['t'] < t1]
    total = int((t1 - t0) * bps * 2.0)  # 格数 = 24
    fig, ax = plt.subplots(figsize=(9, 16), dpi=150, facecolor='#0b0b18')
    ax.set_facecolor('#14142e')
    for gi in range(0, total + 1):
        y = gi
        if gi % 8 == 0:
            ax.axhline(y, color='#4a4a96', lw=2.4, zorder=1)
        elif gi % 4 == 0:
            ax.axhline(y, color='#37377a', lw=1.6, zorder=1)
        else:
            ax.axhline(y, color='#2a2a5c', lw=1.2, zorder=1)
        ax.axhline(y + 0.5, color='#1e1e44', lw=0.8, zorder=1)
    for mi in range(n_bars):
        ax.text(-640, (mi + 1) * 8.0 - 2, f'{mi + 1}', color='#8899cc', fontsize=22,
                ha='left', va='center', fontproperties=FONT, zorder=2)
    ax.axvline(0, color='#5a5aaa', lw=1.2, ls='--', zorder=1)
    ax.text(0, total + 0.6, 'x=0', color='#667', fontsize=13, ha='center', fontproperties=FONT, zorder=2)
    cmap = {1: '#3f83d9', 2: '#9fd4f2'}
    for n in sel:
        tp = n.get('type', 1)
        if tp not in cmap:
            continue
        gy = (n['t'] - t0) * bps * 2.0
        if tp == 2 and n.get('dur', 0) > 0:
            hg = n['dur'] * bps * 1.4
        else:
            hg = 0.35
        ax.add_patch(plt.Rectangle((n['x'] - NOTE_W / 2, gy - hg / 2), NOTE_W, hg,
                                   facecolor=cmap[tp], edgecolor='none', zorder=3))
    ax.set_xlim(-XMAX - 40, XMAX + 40)
    ax.set_ylim(-0.5, total + 1.5)
    ax.set_xticks(np.arange(-600, 601, 200))
    ax.set_xticklabels([f'{v}' for v in np.arange(-600, 601, 200)], color='#8899cc', fontsize=13, fontproperties=FONT)
    ax.set_yticks([])
    ax.text(0, -1.6, '判定线x（RPE px，-675~+675）', color='#667', fontsize=13,
            ha='center', fontproperties=FONT)
    ax.text(40, 2.0, '1/8主网格+1/16细线', color='#556', fontsize=11, fontproperties=FONT)
    ax.set_title(f'编辑器风格 · {fid} · 前{n_bars}小节 · BPM{bpm:.0f} · 音符{len(sel)} · 宽={NOTE_W:.0f}px',
                 color='#ccd6ff', fontsize=17, fontproperties=FONT, pad=10)
    out = f'{BASE}/viz_plain_{fid}_{tag}.png'
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == '__main__':
    main()
