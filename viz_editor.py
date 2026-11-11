#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编辑器风格排键图（参考 RPE/Phigros 编辑器布局）：
- 竖版 9:16，时间轴从上往下（早->晚），左侧标注小节号
- x 轴 = 判定线 x（-675 ~ +675），y 轴 = 16 分网格（每小节 16 格）
- 小节强线 / 1/8 中线 / 1/16 弱线
- tap = 1 格高的横条（宽=真实178px），hold = 时长拉伸，flick/drag 不画（只 Tap/Hold）
"""
import zipfile, glob, json, io, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc')
NOTE_W = 178.0
BASE = '/home/user/Doubao/chats/38445055719679234/phira_charts_dl'

def segs_from_pez(d):
    out = []
    for it in (d.get('BPMList') or []):
        b = it['startTime'][0] + it['startTime'][1] / max(it['startTime'][2], 1)
        out.append([b, it['bpm']])
    if not out: out = [[0.0, 120.0]]
    return out

def beat_sec(triple, segs):
    b = triple[0] + triple[1] / max(triple[2], 1)
    if b <= 0: return 0.0
    n = len(segs); ss = [0.0]*n
    for i in range(1, n):
        b0, bpm0 = segs[i-1]; b1 = segs[i][0]
        ss[i] = ss[i-1] + (b1 - b0) * 60.0 / bpm0
    idx = n - 1
    for i in range(n-1, -1, -1):
        if b >= segs[i][0]: idx = i; break
    return ss[idx] + (b - segs[idx][0]) * 60.0 / segs[idx][1]


def sec_to_beat(t, segs):
    if t <= 0: return 0.0
    n = len(segs); ss = [0.0]*n
    for i in range(1, n):
        b0, bpm0 = segs[i-1]; b1 = segs[i][0]
        ss[i] = ss[i-1] + (b1 - b0) * 60.0 / bpm0
    idx = n - 1
    for i in range(n-1, -1, -1):
        if t >= ss[i]: idx = i; break
    return segs[idx][0] + (t - ss[idx]) * segs[idx][1] / 60.0

def snap_grid(t, segs, grid=0.25):
    b = sec_to_beat(t, segs)
    b2 = round(b / grid) * grid
    return beat_sec([b2, 0, 1], segs)

def find_matching_pez(fid, infer_notes):
    infer_keys = {(round(n['t'], 3), n['type']) for n in infer_notes}
    best, best_score = None, -1
    for pez in glob.glob(f'{BASE}/subsets/*/{fid}_*.pez'):
        try:
            z = zipfile.ZipFile(pez)
            jn = [n for n in z.namelist() if n.endswith('.json')][0]
            d = json.load(io.BytesIO(z.read(jn)))
            s2 = segs_from_pez(d)
            keys = set()
            for l in (d.get('judgeLineList') or []):
                for n in (l.get('notes') or []):
                    if n.get('type') in (1, 2):
                        keys.add((round(beat_sec(n['startTime'], s2), 3), n['type']))
            score = len(infer_keys & keys)
            if score > best_score: best_score, best = score, pez
        except Exception:
            continue
    return best, best_score

def main(fid, bar_start=None, infer_path=None, tag=''):
    infer = json.load(open(infer_path or f'{BASE}/infer_th_{fid}.json'))
    notes = infer['notes']
    best, bs = find_matching_pez(fid, notes)
    z = zipfile.ZipFile(best); jn = [n for n in z.namelist() if n.endswith('.json')][0]
    d = json.load(io.BytesIO(z.read(jn)))
    segs = segs_from_pez(d)
    bpm0 = segs[0][1]
    bps = bpm0 / 60.0
    # 窗口：默认最密 3 小节；bar_start 指定则从该小节起
    beat_end = max(n['t'] for n in notes) * bps
    win = 12.0 / bps   # 3 小节（网格间距更宽）
    if bar_start is None:
        best_n, t0, t1 = -1, 0.0, 0.0
        g = 0.0
        while g < beat_end:
            a, b = g / bps, g / bps + win
            c = sum(1 for n in notes if a <= n['t'] < b)
            if c > best_n: best_n, t0, t1 = c, a, b
            g += 0.25
    else:
        t0 = bar_start * 4.0 / bps
        t1 = min(t0 + win, beat_end / bps + win)
    sel = [n for n in notes if t0 <= n['t'] < t1]
    # 钉格：音符时间钉回最近1/16格（0.25拍），hold起点同钉
    for n in sel:
        n['t_snap'] = snap_grid(n['t'], segs)
    # 从选中 pez 建音符 size/speed 查找表（按 (t,type)）
    lines = d.get('judgeLineList') or []
    line_ns = {}
    for lid, l in enumerate(lines):
        tbl = {}
        for n in (l.get('notes') or []):
            if n.get('type') in (1, 2):
                tbl[(round(beat_sec(n['startTime'], segs), 3), n.get('type'))] = (n.get('speed', 1.0), n.get('size', 1.0))
        line_ns[lid] = tbl
    # 格坐标：1/8格 = 每拍2格 = 每小节8格；每格占图高3.3%（参考图主网格）
    def grid_of(t):
        return (t - t0) * bps * 2.0
    bar0 = int(t0 * bps / 4)
    total = int((t1 - t0) * bps * 2.0)  # 总格数 = 24（3小节）
    # 画图
    fig, ax = plt.subplots(figsize=(9, 16), dpi=150, facecolor='#0b0b18')
    ax.set_facecolor('#14142e')
    # 网格线
    for gi in range(0, total + 1):
        y = gi
        pos = gi % 8
        if pos == 0: ax.axhline(y, color='#4a4a96', lw=2.4, zorder=1)
        elif gi % 4 == 0: ax.axhline(y, color='#37377a', lw=1.6, zorder=1)
        else: ax.axhline(y, color='#2a2a5c', lw=1.2, zorder=1)
        ax.axhline(y + 0.5, color='#1e1e44', lw=0.8, zorder=1)  # 1/16 细线
    # 小节号标注（左侧）
    for mi in range(3):
        yy = (mi + 1) * 8.0
        ax.text(-640, yy - 2, f'{bar0 + mi}', color='#8899cc', fontsize=22,
                ha='left', va='center', fontproperties=FONT, zorder=2)
    ax.axvline(0, color='#5a5aaa', lw=1.2, ls='--', zorder=1)
    ax.text(0, total + 0.6, 'x=0', color='#667', fontsize=13, ha='center', fontproperties=FONT, zorder=2)
    # 音符：tap=深蓝横条(厚0.35格) / hold=浅蓝长条（时长拉伸），均中心对齐网格线
    cmap = {1: '#3f83d9', 2: '#9fd4f2'}   # tap 深蓝 / hold 浅蓝
    for n in sel:
        tp = n['type']
        if tp not in cmap: continue
        gy = grid_of(n.get('t_snap', n['t']))
        sp, sz = line_ns.get(n['line_id'], {}).get((round(n['t'], 3), tp), (1.0, 1.0))
        nw = NOTE_W * max(sz, 0.1)
        if tp == 2 and n.get('dur', 0) > 0:
            hg = n['dur'] * bps * 1.4   # hold 条高（格）
        else:
            hg = 0.35                   # tap 厚度（格）
        ax.add_patch(Rectangle((n['x'] - nw / 2, gy - hg / 2), nw, hg,
                               facecolor=cmap[tp], alpha=0.95, edgecolor='none', zorder=3))
    ax.set_xlim(-675, 675); ax.set_ylim(0, total)
    ax.invert_yaxis()   # 时间从上往下
    ax.set_xlabel('判定线 x（RPE px，-675 ~ +675）', fontproperties=FONT, fontsize=20, color='#ccc')
    ax.set_ylabel('1/8主网格（每小节8格·间距4.2%）+1/16细线', fontproperties=FONT, fontsize=20, color='#ccc')
    ax.set_title(f'编辑器风格 · fid {fid} · 第{bar0}-{bar0 + 3}小节 · BPM{bpm0:.0f} · 音符{len(sel)} · 宽=178px',
                 fontproperties=FONT, fontsize=18, color='white', pad=10)
    ax.set_yticks([i * 8 for i in range(0, total // 8 + 1)])
    ax.set_yticklabels([f'{bar0 + i}' for i in range(0, total // 8 + 1)],
                       fontsize=14, color='#6677aa')
    ax.tick_params(axis='x', colors='#889', labelsize=15)
    for s in ax.spines.values(): s.set_color('#33336a')
    fig.tight_layout()
    out = f'{BASE}/viz_edit_{fid}{tag}{"" if bar_start is None else f"_b{bar0}"}.png'
    fig.savefig(out, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f'{fid}: 第{bar0}-{bar0 + 3}小节 音符{len(sel)} BPM{bpm0:.0f} 匹配{bs} -> {out}')

if __name__ == '__main__':
    for fid in ['37977', '21483', '69319', '34756', '25262']:
        try: main(fid)
        except Exception:
            import traceback; traceback.print_exc()
