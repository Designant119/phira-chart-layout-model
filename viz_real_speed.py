#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phira 真实流速排键图（快照视角）：
y 轴 = 音符相对自己判定线的距离（RPE px），当前时刻取窗口末 t1。
  height(t) = ∫ speed(τ)·SPEED_RATIO dτ （prpr rpe.rs/line.rs）
  y = 450 × (height(t1) - height(t)) × note.speed
  → 0px = 已落到判定线；越大 = 离判定线越远（上方）。
BPM 一律用选中 pez 自带 BPMList（多谱同源）。
网格线：段内平均 speed 的参考 16 分格（近似，非逐线精确）。
"""
import zipfile, glob, json, io, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc')
SPEED_RATIO = 10.0 / 45.0 / 0.83175
PX_PER_SPEED_SEC = 450.0 * SPEED_RATIO   # ≈120.2
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

def speed_pieces(events, segs, t_end):
    if not events:
        return [(0.0, 0.0)]
    evs = sorted(events, key=lambda e: beat_sec(e['startTime'], segs))
    pts = [(0.0, 0.0)]; h = 0.0; cursor = 0.0; last = 0.0
    for e in evs:
        s = beat_sec(e['startTime'], segs); en = beat_sec(e['endTime'], segs)
        s = max(s, cursor); en = max(en, s)
        if s > cursor + 1e-9:
            h += last * (s - cursor)
            pts.append((s, h))
        v0 = e.get('start', 0.0) * SPEED_RATIO; v1 = e.get('end', 0.0) * SPEED_RATIO
        if en > s + 1e-9:
            h += (v0 + v1) / 2.0 * (en - s)
            pts.append((en, h))
        cursor = en; last = v1
    if cursor < t_end - 1e-9:
        h += last * (t_end - cursor); pts.append((t_end, h))
    return pts

def height_at(pts, t):
    if t <= pts[0][0]: return pts[0][1]
    for i in range(1, len(pts)):
        if t <= pts[i][0]:
            (t0, h0), (t1, h1) = pts[i-1], pts[i]
            return h0 + (h1 - h0) * (t - t0) / max(t1 - t0, 1e-9)
    return pts[-1][1]

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

def main(fid):
    infer = json.load(open(f'{BASE}/infer_th_{fid}.json'))
    notes = infer['notes']
    best, bs = find_matching_pez(fid, notes)
    z = zipfile.ZipFile(best); jn = [n for n in z.namelist() if n.endswith('.json')][0]
    d = json.load(io.BytesIO(z.read(jn)))
    segs = segs_from_pez(d)
    lines = d.get('judgeLineList') or []
    line_ev, line_ns = {}, {}
    for lid, l in enumerate(lines):
        ev = []
        for el in (l.get('eventLayers') or []):
            if isinstance(el, dict): ev += el.get('speedEvents') or []
        line_ev[lid] = ev
        tbl = {}
        for n in (l.get('notes') or []):
            tbl[(round(beat_sec(n['startTime'], segs), 3), n.get('type'))] = n.get('speed', 1.0)
        line_ns[lid] = tbl
    t_max = max(n['t'] for n in notes) + 5.0
    line_pts = {lid: speed_pieces(ev, segs, t_max) for lid, ev in line_ev.items()}
    items = []
    for n in notes:
        lid = n['line_id']
        sp = line_ns.get(lid, {}).get((round(n['t'], 3), n['type']), 1.0)
        items.append((n['t'], height_at(line_pts.get(lid, [(0.0, 0.0)]), n['t']),
                      n['x'], n['type'], n.get('dur', 0.0), lid, sp))
    bps = segs[0][1] / 60.0
    beat_end = max(n['t'] for n in notes) * bps
    win = 8.0 / bps
    best_n, t0, t1 = -1, 0.0, 0.0
    g = 0.0
    while g < beat_end:
        a, b = g / bps, g / bps + win
        c = sum(1 for it in items if a <= it[0] < b)
        if c > best_n: best_n, t0, t1 = c, a, b
        g += 1.0 / 4.0
    sel = [it for it in items if t0 <= it[0] < t1]
    sel2 = []
    for t, h, x, tp, dur, lid, sp in sel:
        h1 = height_at(line_pts.get(lid, [(0.0, 0.0)]), t1)
        y = 450.0 * (h1 - h) * max(sp, 0.05)
        sel2.append((t, y, x, tp, dur, lid, sp))
    ymin = min(s[1] for s in sel2); ymax = max(s[1] for s in sel2)
    span = max(ymax - ymin, 300.0)
    avg_speeds = []
    for lid, ev in line_ev.items():
        if ev: avg_speeds.append(ev[0].get('start', 1.0))
    avg_speed = sorted(avg_speeds)[len(avg_speeds)//2] if avg_speeds else 1.0
    def grid_y(tt):
        return 450.0 * avg_speed * SPEED_RATIO * (t1 - tt)
    # 画图
    fig, ax = plt.subplots(figsize=(9, 16), dpi=150, facecolor='#0b0b18')
    ax.set_facecolor('#111128')
    g = math.floor(t0 * bps * 4.0) - 1
    while True:
        tt = g / 4.0 / bps
        if tt > t1: break
        if tt >= t0:
            yv = grid_y(tt)
            if -0.02 * span <= yv <= ymax + 0.02 * span:
                if g % 16 == 0: ax.axhline(yv, color='#3d3d85', lw=2.6, zorder=1)
                elif g % 4 == 0: ax.axhline(yv, color='#2c2c60', lw=1.8, zorder=1)
                else: ax.axhline(yv, color='#1c1c3e', lw=1.2, zorder=1)
        g += 1
    ax.axvline(0, color='#4a4a8a', lw=1.2, ls='--', zorder=1)
    ax.text(0, ymax + 0.015 * span, 'x=0', color='#667', fontsize=12, ha='center', fontproperties=FONT, zorder=2)
    cmap = {1: '#ffb74d', 2: '#7fd4ff'}
    for t, y, x, tp, dur, lid, sp in sel2:
        if tp not in cmap: continue
        if tp == 2 and dur > 0:
            ev = line_ev.get(lid, [])
            s_now = ev[0].get('start', 1.0) if ev else 1.0
            hg = dur * 450.0 * s_now * SPEED_RATIO
        else:
            hg = NOTE_W * 0.9
        ax.add_patch(Rectangle((x - NOTE_W / 2, y - hg / 2), NOTE_W, hg,
                               facecolor=cmap[tp], alpha=0.92, edgecolor='none', zorder=3))
    ax.set_xlim(-675, 675); ax.set_ylim(-0.05 * span, ymax + 0.05 * span)
    ax.set_xlabel('判定线 x（RPE px，-675 ~ +675）', fontproperties=FONT, fontsize=17, color='#bbb')
    ax.set_ylabel('音符距自己判定线的真实距离（RPE px，0=已落线）', fontproperties=FONT, fontsize=17, color='#bbb')
    ax.set_title(f'真实流速(快照) · fid {fid} · 第{int(t0 * bps / 4)}-{int(t1 * bps / 4)}小节(最密) · 音符宽178px · 中位speed={avg_speed:.1f} · 匹配{bs}音符',
                 fontproperties=FONT, fontsize=19, color='white', pad=14)
    ticks = [span * i / 4 for i in range(5)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{int(v)}px' for v in ticks], fontsize=13, color='#889')
    ax.tick_params(axis='x', colors='#889', labelsize=13)
    for s in ax.spines.values(): s.set_color('#33336a')
    fig.tight_layout()
    out = f'{BASE}/viz_spd_{fid}.png'
    fig.savefig(out, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f'{fid}: 匹配{bs} 段[{t0:.2f},{t1:.2f}]s 音符{len(sel2)} 跨度{span:.0f}px 中位speed={avg_speed:.1f}')

if __name__ == '__main__':
    for fid in ['37977', '21483', '69319', '34756', '25262']:
        try: main(fid)
        except Exception:
            import traceback; traceback.print_exc()
