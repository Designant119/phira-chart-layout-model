#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规则排键 v4：整首一套对齐中心的等间距网格，段内只在这套列位上选。
- 基础网格：间距 5 格、13 列、关于屏心对称；少数段换间距/列数但仍对齐中心
- 列位按距中心加权：中列高频、边缘低频
- 单音：左右交替为主，允许同列连敲、同侧连走 2 步
- 双押对称配对，三押两端+中，四押均分
用法: python3 rule_charter4.py <chart.json> <out.json> [--seed N]"""
import os, sys, json, argparse, random, math
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phira_fetch import rpe_time_to_beat, parse_chart

NB = 81
HALF = 675.0
SEG = 24
SAME_CELL_P = 0.28      # 基准：原地同列概率（0.183s 间隔谱面口径，运行时按密度自适应）
FLIP_P = 0.53           # 基准：单步换向概率（运行时按密度自适应）
# 非零步长的跳列数权重（按真谱 15831 实测计数）
STEP_W = [15, 63, 128, 285, 90, 183, 72, 109, 23, 11, 0, 23]
STEP_IDX = list(range(1, 13))


def g2x(c):
    return -HALF + c / (NB - 1) * 2 * HALF


def wchoice(rng, idx, w):
    w = np.array(w, dtype=np.float64)
    w = w / w.sum()
    return idx[int(np.searchsorted(w.cumsum(), rng.random()))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--use-model', action='store_true', help='单音位置改用段级模型')
    ap.add_argument('--use-model2', action='store_true', help='用 v2 段级模型，一拍多颗')
    a = ap.parse_args()

    rng = random.Random(a.seed)
    d = json.load(open(a.chart, encoding='utf-8'))
    # 原谱的速度曲线要在被清空之前存下来：hold 长度和音符纵向间距都靠它
    orig_speed = []
    for _jl in d['judgeLineList']:
        for _lyr in (_jl.get('eventLayers') or []):
            for e in (_lyr or {}).get('speedEvents', []) or []:
                if isinstance(e, dict) and e.get('startTime'):
                    orig_speed.append((tuple(e['startTime'][:3]), float(e.get('start', 10.0))))
    _, psegs = parse_chart(d)

    def b2s(beat):
        total = 0.0
        for i, (sb, bpm) in enumerate(psegs):
            if i + 1 < len(psegs):
                nb = psegs[i + 1][0]
                if beat < nb:
                    return total + (beat - sb) * 60.0 / bpm
                total += (nb - sb) * 60.0 / bpm
            else:
                return total + (beat - sb) * 60.0 / bpm
        return beat * 0.5

    notes = []
    for li, jl in enumerate(d['judgeLineList']):
        for ni, n in enumerate(jl.get('notes', [])):
            if n.get('isFake', 0) == 1:
                continue
            beat = rpe_time_to_beat(n['startTime'])
            notes.append({'b': round(beat, 4), 't': b2s(beat),
                          'li': li, 'ni': ni, 'out': 0.0,
                          'x': float(n.get('positionX', 0.0)),
                          'type': int(n.get('type', 1))})
    mpos = {}
    mpos2 = {}
    if getattr(a, 'use_model2', False):
        import model_pos2
        mpos2 = model_pos2.build(notes, seed=a.seed)
        print('# v2 模型给出 %d 拍（每拍多颗）' % len(mpos2), flush=True)
    if a.use_model:
        import model_pos
        mpos = model_pos.build(notes, seed=a.seed)
        print('# 段级模型给出 %d 拍位置' % len(mpos), flush=True)

    groups = defaultdict(list)
    for r in notes:
        groups[r['b']].append(r)
    order = sorted(groups)

    # ---- drag 段识别：时间相邻、整拍全是 drag 的连续段（真谱 drag 扫线都是单拍单 drag）----
    drag_run = {}          # 拍 -> (run_id, idx_in_run, run_len)
    run_state = {}         # run_id -> 扫线状态
    rid = -1
    cur_run = []
    def _flush_run():
        nonlocal rid
        if len(cur_run) >= 1:
            rid += 1
            for idx, b in enumerate(cur_run):
                drag_run[b] = (rid, idx, len(cur_run))
        cur_run.clear()
    for b in order:
        grp = groups[b]
        all_drag = all(x['type'] == 3 for x in grp)
        if all_drag:
            cur_run.append(b)
        else:
            _flush_run()
    _flush_run()

    # 切段
    segs, cur, cnt = [], [], 0
    for b in order:
        k = len(groups[b])
        if cnt + k > SEG and cur:
            segs.append(cur); cur = []; cnt = 0
        cur.append(b); cnt += k
    if cur:
        segs.append(cur)

    # 全格位游走：候选列位放开到全部 81 格，不再锁十几列等间距网格
    ncol = NB
    cols = list(range(NB))
    half = (NB - 1) / 2.0
    h = int(round(half))
    COL_STEP = 4  # 原基准列宽换算格步用，跳列权重 → 格步

    # 密度自适应：基准取全量 9746 首谱面分层中位（full_baseline.py）
    # 换向全量恒定 0.70；步长/同列率随密度略变；|x| 中位恒定 225px
    ts = sorted(set(r['t'] for r in notes))
    ivs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    med_iv = float(np.median(ivs)) if ivs else 0.13
    tgt_px = min(max(263.0 + 245.0 * (med_iv - 0.094), 250.0), 300.0)
    step_scale = tgt_px / 270.0
    scp = min(max(0.15 + 0.47 * (med_iv - 0.094), 0.13), 0.22)
    flip_p = 0.70
    chord_follow_p = min(max((0.13 - med_iv) * 4.0, 0.0), 0.8)  # 密集谱多押中心跟随游走
    # 多押半宽（px）按全量分层中位插值：2押随密度略收，3/4押随密度加宽
    # 锚点 稀疏 iv=0.200 / 密集 iv=0.083，全宽中位 → 半宽 px
    td = min(max((0.200 - med_iv) / (0.200 - 0.083), 0.0), 1.0)
    colw = 2 * HALF / (NB - 1)  # 每格的 RPE 坐标宽度
    # phira 源码 NOTE_WIDTH_RATIO_BASE=0.13175016，note 画出来的宽是屏宽 13.17%：
    # 480px 屏下 63px，换算到 RPE 是 177 坐标，也就是 10.5 格。以前把 16.875 当 px 用，导致所有 px 目标缩水 2.8 倍
    PX_PER_COL = 2 * HALF / (NB - 1) * 0.3556  # 1 格 = 6.0px
    NOTE_PX = 0.13175016 * 480.0 * 1.0        # 一个 note 的实际宽度（px）
    # 本曲横向尺度：取本曲 tap 的 |x| 中位（真谱里多押半宽基本就等于本曲音符的典型 |x|）
    tap_xs = [abs(r['x']) for r in notes if r.get('type') == 1 and abs(r.get('x', 0)) > 1]
    tap_med = float(np.median(tap_xs)) if len(tap_xs) >= 20 else 270.0
    # 真谱 15831 实测：2押跨度中位180px(半宽90)、3押270(半宽135)、4押180(半宽90)
    w2_px = 90.0 * (tap_med / 253.0) * (1.0 - 0.12 * td)
    w3_px = 135.0 * (tap_med / 253.0) * (1.0 + 0.10 * td)
    w4_px = 90.0 * (tap_med / 253.0) * (1.0 + 0.10 * td)
    mag_c = tgt_px / colw  # drag 横向尺度的格数：本曲目标步长换算成格，drag 就落在这个量级附近
    print(f'# 间隔中位 {med_iv:.3f}s 目标步长 {tgt_px:.0f}px 缩放 {step_scale:.2f} 同列概率 {scp:.2f} 换向 {flip_p:.2f} 密度档 {td:.2f}')

    for seg in segs:
        # 单音走位移状态机游走；多押按押型独立选对称宽度
        i = h
        dirn = 1
        last_use = [-1e9] * ncol  # 每列最近一次落点时间，用于生成端避让
        L = R = 0                # 段内左右落点计数，平衡拉力用

        for bt in seg:
            items = groups[bt]
            m2 = mpos2.get(round(bt, 3)) if mpos2 else None
            k = len(items)
            cur_t = items[0]['t']
            cool = 1.0 * med_iv

            def avoid(c):
                # 目标列刚被占用过，就在邻近列里挑冷却最久的，生成时自然错开
                if last_use[c] > cur_t - cool:
                    lo, hi = max(0, c - 2), min(ncol - 1, c + 2)
                    cand = list(range(lo, hi + 1))
                    return min(cand, key=lambda q: last_use[q])
                return c

            # 多押中心偏移：一侧偏多时整组往偏少侧挪
            if L - R >= 2:
                bias = rng.choice([0, 1, 2])
            elif R - L >= 2:
                bias = rng.choice([-2, -1, 0])
            else:
                bias = rng.choice([-2, -1, 0, 0, 1, 2])

            # drag 扫线推进：每拍都推进（与同拍几颗无关），撞界反弹，步长严格等距
            runinfo = drag_run.get(bt)
            sweep_col = None
            if runinfo and runinfo[2] >= 4:
                r_id, r_idx, r_len = runinfo
                if r_id not in run_state:
                    # 扫线幅度跟着本曲横向尺度走；步长取半幅的 0.28~0.5，一次横穿约 3 拍，
                    # 和真谱往复扫（如 31572 那串 180px 步长、半幅 540px）一个量级
                    amp = rng.uniform(mag_c * 0.65, mag_c * 1.35)
                    stepc = amp * rng.uniform(0.28, 0.48)
                    run_state[r_id] = {
                        'amp': amp,
                        'step': stepc,
                        'col': h + rng.uniform(-amp * 0.55, amp * 0.55),
                        'dirn': rng.choice([-1, 1]),
                        'last': 0}
                st = run_state[r_id]
                adv = r_idx - st['last']
                for _ in range(max(0, adv)):
                    st['col'] += st['dirn'] * st['step']
                    lo, hi = h - st['amp'], h + st['amp']
                    if st['col'] > hi:
                        st['col'] = 2 * hi - st['col']; st['dirn'] = -1
                    elif st['col'] < lo:
                        st['col'] = 2 * lo - st['col']; st['dirn'] = 1
                    st['col'] = float(np.clip(st['col'], 0, ncol - 1))
                st['last'] = r_idx
                sweep_col = int(np.clip(round(st['col']), 0, ncol - 1))
                if os.environ.get('DBG_SWEEP'):
                    print('  run%d idx%d adv%d step%.1f col%.1f->%d amp%.1f' % (
                        r_id, r_idx, adv, st['step'], st['col'], sweep_col, st['amp']))

            if k == 1:
                is_drag = items[0]['type'] == 3
                mpx = mpos.get(round(bt, 3)) if mpos else None
                if mpx is not None and sweep_col is None and not is_drag:
                    # 单音位置由段级模型决定（模型只看节奏指纹，不看原谱位置）
                    c0 = int(np.clip(round((mpx + HALF) / colw), 0, ncol - 1))
                    # 模型采样偏中间：真谱 21% 的音在 |x|>400，模型只有 11%，做一次径向拉伸把外侧拉开
                    u = (c0 - h) / half
                    u = np.sign(u) * (abs(u) ** 0.78)
                    c = int(np.clip(round(h + u * half), 0, ncol - 1))
                    i = c
                elif sweep_col is not None:
                    c = sweep_col
                elif is_drag:
                    # 孤立 drag：落在偏外但跟着本曲横向尺度走（真谱 drag |x| 与本曲 tap 基本同量级）
                    side = rng.choice([-1, 1])
                    mag = int(np.clip(round(rng.gauss(mag_c, mag_c * 0.45)), 3, 41))
                    c = int(np.clip(round(h + side * mag), 0, ncol - 1))
                elif rng.random() < scp:
                    c = i
                else:
                    step = wchoice(rng, STEP_IDX, STEP_W)
                    step = max(1, int(round(step * COL_STEP * step_scale)))
                    # 平衡拉力：本段一侧落太多，就把方向掰去偏少侧
                    if i <= h and L - R >= 2:
                        dirn = 1
                    elif i >= h and R - L >= 2:
                        dirn = -1
                    elif rng.random() < flip_p:
                        dirn = -dirn
                    nc = i + dirn * step
                    if nc < 0:
                        nc = -nc; dirn = 1
                    elif nc > ncol - 1:
                        nc = 2 * (ncol - 1) - nc; dirn = -1
                    c = int(np.clip(nc, 0, ncol - 1))
                    if c == h and rng.random() < 0.6:
                        c = int(np.clip(h + dirn, 0, ncol - 1))
                    c = avoid(c)
                i = c
                if c < h:
                    L += 1
                elif c > h:
                    R += 1
                pos = [c]
            elif k == 2:
                types2 = [x['type'] for x in items]
                if sweep_col is not None:
                    # 扫线段内的双 drag：围绕扫线列做窄对称，保持扫线连续性
                    pos = [int(np.clip(sweep_col - 5, 0, ncol - 1)),
                           int(np.clip(sweep_col + 5, 0, ncol - 1))]
                    i = sweep_col
                elif 3 in types2 and types2 != [3, 3]:
                    # drag 与 tap 同拍：位置贴着 tap（真谱 15831 里 drag 与 tap 的 |x| 中位完全相同）
                    # 46.4% 比 tap 更靠外，27.9% 更靠内，25.8% 差不多
                    tap_col = int(np.clip(round(i), 0, ncol - 1))
                    if rng.random() < 0.45:
                        off = rng.randint(11, 20)
                        sgn = 1
                    else:
                        # drag 图标约 5 格宽（84px），比 tap 的 2 格宽得多，所以最近也要隔 5 格
                        off = rng.randint(11, 16)
                        sgn = -1
                    out_dir = np.sign(tap_col - h) or rng.choice([-1, 1])
                    cd = int(np.clip(round(tap_col + sgn * out_dir * off), 0, ncol - 1))
                    # 真谱里少数 drag 才落到中间带（13%），落到中带就多半往外推一档
                    if abs(cd - h) < 9 and rng.random() < 0.65:
                        cd = int(np.clip(round(tap_col + out_dir * rng.randint(11, 18)), 0, ncol - 1))
                    pos = [cd if t == 3 else tap_col for t in types2]
                    i = tap_col
                else:
                    w2 = max(11, int(round(w2_px / PX_PER_COL + rng.choice([-0.5, 0, 0, 0.5]))))
                    ch_raw = (i if rng.random() < chord_follow_p else h) + bias
                    ch = int(np.clip(ch_raw, w2, ncol - 1 - w2))
                    ch = avoid(ch)
                    l = int(np.clip(ch - w2, 0, ncol - 1)); r = int(np.clip(ch + w2, 0, ncol - 1))
                    # 真谱双押有 10% 是两颗压在同一列（当成一个重音点）
                    if rng.random() < 0.10:
                        pos = [ch, ch]
                    else:
                        pos = [l, r]
                    i = r if dirn > 0 else l
            elif k == 3:
                w3 = max(11, int(round(w3_px / PX_PER_COL + rng.choice([-0.5, 0, 0, 0.5]))))
                ch_raw = (i if rng.random() < chord_follow_p else h) + bias
                ch = int(np.clip(ch_raw, w3, ncol - 1 - w3))
                ch = avoid(ch)
                l = int(np.clip(ch - w3, 0, ncol - 1)); r = int(np.clip(ch + w3, 0, ncol - 1))
                mid = int(np.clip(ch + rng.choice([-1, 0, 0, 1]), 0, ncol - 1))
                pos = [l, mid, r]; i = h
            else:
                w4 = max(11, int(round(w4_px / PX_PER_COL + rng.choice([-0.5, 0, 0, 0.5]))))
                ch_raw = (i if rng.random() < chord_follow_p else h) + bias
                ch = int(np.clip(ch_raw, w4, ncol - 1 - w4))
                ch = avoid(ch)
                l = int(np.clip(ch - w4, 0, ncol - 1)); r = int(np.clip(ch + w4, 0, ncol - 1))
                # 真谱四押 58 拍里 50 拍是左右各两颗完全重合（视觉只有两个点），只有 14% 摊开成四颗
                gap4 = max(11, w4 // 3)
                if rng.random() < 0.14:
                    pos = [l, int(np.clip(l + gap4, 0, ncol - 1)),
                           int(np.clip(r - gap4, 0, ncol - 1)), r]
                else:
                    pos = [l, l, r, r]
                pos = pos[:k]; i = r if dirn > 0 else l
            if m2 and sweep_col is None and not any(t == 3 for t in [x['type'] for x in items]):
                # v2 模型直接给每颗的位置，颗数不够就用最后一个位置叠着（重合本来就是合法写法）
                pos = [int(np.clip(round((m2[min(j, len(m2) - 1)] + HALF) / colw), 0, ncol - 1))
                       for j in range(k)]
            seen, clean = set(), []
            # 真谱里同列重合是常规写法（四押左右各两颗、双押压成一点、三押偶尔塌成一点），不能被去重逻辑推开一格
            allow_dup = (k >= 2)
            for p in pos:
                p = int(np.clip(p, 0, ncol - 1))
                if p in seen and not allow_dup:
                    for dd in range(1, ncol):
                        if p - dd >= 0 and p - dd not in seen:
                            p = p - dd; break
                        if p + dd < ncol and p + dd not in seen:
                            p = p + dd; break
                seen.add(p); clean.append(p)
            for p in clean:
                last_use[p] = cur_t
                if p < h:
                    L += 1
                elif p > h:
                    R += 1
            for kk, r in enumerate(items):
                c = clean[kk] if kk < len(clean) else clean[-1]
                r['out'] = g2x(cols[c])

    # hold 重叠避让：时间交叠的长押分居对侧，间距不小于阈值
    from gen_chart_from_model import collect
    rec = collect(d)
    out_map = {(r['li'], r['ni']): r['out'] for r in notes}
    holds = [r for r in rec if r['type'] == 2 and r['dur'] > 0.1]
    holds.sort(key=lambda r: r['t'])

    def ov(A, B):
        return B['t'] < A['t'] + A['dur'] and A['t'] < B['t'] + B['dur']

    MIN_HOLD_GAP = 336.0
    for i in range(len(holds)):
        for j in range(i + 1, len(holds)):
            A, B = holds[i], holds[j]
            if not ov(A, B):
                continue
            x1 = out_map[(A['li'], A['ni'])]
            x2 = out_map[(B['li'], B['ni'])]
            if abs(x1 - x2) < MIN_HOLD_GAP:
                x2 = -x1
                if abs(x2 - x1) < MIN_HOLD_GAP:
                    x2 = -np.sign(x1 if x1 != 0 else 1) * 600.0
                out_map[(B['li'], B['ni'])] = float(np.clip(x2, -HALF, HALF))
    for r in notes:
        r['out'] = out_map[(r['li'], r['ni'])]

    for r in notes:
        d['judgeLineList'][r['li']]['notes'][r['ni']]['positionX'] = round(
            float(np.clip(r['out'], -HALF, HALF)), 2)

    # 判定线全部静止水平：清掉移动/旋转/速度动画与位移控制，不保留原谱线动效
    static_pos = [{"easing": 1, "pos": 1.0, "x": 0.0}, {"easing": 1, "pos": 1.0, "x": 9999999.0}]
    static_y = [{"easing": 1, "x": 0.0, "y": 1.0}, {"easing": 1, "x": 9999999.0, "y": 1.0}]
    static_size = [{"easing": 1, "size": 1.0, "x": 0.0}, {"easing": 1, "size": 1.0, "x": 9999999.0}]
    static_skew = [{"easing": 1, "skew": 0.0, "x": 0.0}, {"easing": 1, "skew": 0.0, "x": 9999999.0}]
    def _static_ev(val):
        return [{"bezier": 0, "bezierPoints": [0, 0, 0, 0], "easingLeft": 0.0,
                 "easingRight": 1.0, "easingType": 1, "end": val, "endTime": [1, 0, 1],
                 "linkgroup": 0, "start": val, "startTime": [0, 0, 1]}]
    # 判定线不再全锁平：真谱 15831 每条线都有 rotate/moveY 动效，音符沿倾斜的线铺开
    # 音最多的那条线保持水平当主轴，其余线按闲置顺序给固定倾角
    def _bt(b):
        b = float(max(0.0, b))
        return [int(b), int(round((b - int(b)) * 4)), 4]

    def _ev(st, en, t0, t1):
        return {"bezier": 0, "bezierPoints": [0, 0, 0, 0], "easingLeft": 0.0,
                "easingRight": 1.0, "easingType": 1, "start": st, "end": en,
                "startTime": t0, "endTime": t1, "linkgroup": 0}

    # 每条线只在"自己有音符的那段时间"显示，避免同一时刻一堆线织成网格
    line_span = {}
    for r in notes:
        li = r['li']; b = r['b']
        if li not in line_span:
            line_span[li] = [b, b]
        else:
            line_span[li][0] = min(line_span[li][0], b)
            line_span[li][1] = max(line_span[li][1], b)

    def alpha_track(li):
        b0, b1 = line_span.get(li, (0, 1))
        # 音符还在屏幕上方时线就该亮，所以窗口整体提前四拍
        return [_ev(0, 0, [0, 0, 1], _bt(b0 - 5.0)),
                _ev(0, 255, _bt(b0 - 5.0), _bt(b0 - 4.0)),
                _ev(255, 255, _bt(b0 - 4.0), _bt(b1 + 2.0)),
                _ev(255, 0, _bt(b1 + 2.0), _bt(b1 + 3.0))]

    ANGLE_POOL = [0, -28, 26, -18, 18, 33, -33]
    YOFF_POOL = [0, -420, 420, -210, 210, -560, 560]
    sound_lines = [i for i, jl in enumerate(d['judgeLineList']) if jl.get('notes')]
    sound_lines.sort(key=lambda i: -len(d['judgeLineList'][i]['notes']))
    for rank, li in enumerate(sound_lines):
        ang = ANGLE_POOL[rank % len(ANGLE_POOL)]
        yo = YOFF_POOL[rank % len(YOFF_POOL)]
        d['judgeLineList'][li]['eventLayers'] = [{
            "alphaEvents": alpha_track(li),
            "moveXEvents": _static_ev(0), "moveYEvents": _static_ev(yo),
            "rotateEvents": _static_ev(ang),
            "speedEvents": _static_ev(10.0)}]
    for li, l in enumerate(d['judgeLineList']):
        if li not in sound_lines:
            # 没音的线一律透明：真谱里它们本来就不显示，留着只会织成满屏网格
            l['eventLayers'] = [{
                "alphaEvents": _static_ev(0),
                "moveXEvents": _static_ev(0), "moveYEvents": _static_ev(0),
                "rotateEvents": _static_ev(0),
                "speedEvents": _static_ev(10.0)}]
        l['posControl'] = static_pos
        l['yControl'] = static_y
        l['sizeControl'] = static_size
        l['skewControl'] = static_skew
    # === 不沿用原谱判定线：自造两条线，按小段轮流挂音，避免所有音挤在一条带上 ===
    _jl = d['judgeLineList'][0]
    _all = []
    for _l in d['judgeLineList']:
        _all.extend(_l.get('notes', []) or [])
    _all.sort(key=lambda n: n['startTime'][0] + n['startTime'][1] / max(1, n['startTime'][2]))

    def _mkline(tmpl, yoff, ang):
        return {
            "Group": 0, "Name": "own", "Texture": "line.png",
            "alphaControl": [{"alpha": 1.0, "easing": 1, "x": 0.0}, {"alpha": 1.0, "easing": 1, "x": 9999999.0}],
            "bpmfactor": 1.0,
            "extended": tmpl.get('extended', {"inclineEvents": []}),
            "father": -1, "isCover": 1, "notes": [], "numOfNotes": 0,
            "eventLayers": [{
                "alphaEvents": [_ev(255, 255, [0, 0, 1], [9999999, 0, 1])],
                "moveXEvents": [_ev(0.0, 0.0, [0, 0, 1], [9999999, 0, 1])],
                "moveYEvents": [_ev(float(yoff), float(yoff), [0, 0, 1], [9999999, 0, 1])],
                "rotateEvents": [_ev(float(ang), float(ang), [0, 0, 1], [9999999, 0, 1])],
                "speedEvents": [_ev(10.0, 10.0, [0, 0, 1], [9999999, 0, 1])]}],
            "zOrder": tmpl.get('zOrder', 0),
            "posControl": [{"easing": 1, "pos": 1.0, "x": 0.0}, {"easing": 1, "pos": 1.0, "x": 9999999.0}],
            "sizeControl": [{"easing": 1, "size": 1.0, "x": 0.0}, {"easing": 1, "size": 1.0, "x": 9999999.0}],
            "skewControl": [{"easing": 1, "skew": 0.0, "x": 0.0}, {"easing": 1, "skew": 0.0, "x": 9999999.0}],
            "yControl": [{"easing": 1, "x": 0.0, "y": 1.0}, {"easing": 1, "x": 9999999.0, "y": 1.0}],
        }
    _LNS = [_mkline(_jl, 0.0, 0.0)]
    # 按"连续 10 拍"切小段轮流换线：同屏大约两条线都在，但每条各承担一半音，密度降一半
    _grp = 10
    for _i, _n in enumerate(_all):
        _t = _n['startTime']; _b = _t[0] + _t[1] / max(1, _t[2])
        _line = 0
        _LNS[_line]['notes'].append(_n)
    for _l in _LNS:
        _l['numOfNotes'] = len(_l['notes'])
    d['judgeLineList'] = _LNS
    json.dump(d, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)

    # 统计口径与 full_baseline 对齐：单音序列、非零步算换向；同列率也在单音序列上
    from collections import defaultdict as _dd
    _g = _dd(list)
    for r in notes:
        _g[round(r['b'], 3)].append(r['out'])
    _singles = [v[0] for v in _g.values() if len(v) == 1]
    if len(_singles) >= 3:
        _s = np.array(_singles, dtype=float)
        _dx = np.abs(np.diff(_s))
        same = float((_dx < 20).mean())
        _d = np.diff(_s); _d = _d[np.abs(_d) > 1e-6]
        alt = float((np.sign(_d[:-1]) * np.sign(_d[1:]) < 0).mean()) if len(_d) >= 2 else 0.5
        step_med = float(np.median(_dx[_dx > 1e-6])) if (_dx > 1e-6).any() else 0.0
    else:
        alt, same, step_med = 0.5, 0.0, 0.0
    xs = np.array([r['out'] for r in sorted(notes, key=lambda r: r['b'])])
    q = np.clip(np.round((xs + HALF) / (2 * HALF) * (NB - 1)), 0, NB - 1).astype(int)
    u, c2 = np.unique(q, return_counts=True)
    spans = [xs[j:j + SEG].max() - xs[j:j + SEG].min() for j in range(0, max(1, len(xs) - SEG), SEG)]
    print('段 %d 总音 %d' % (len(segs), len(notes)))
    print('步长中位 %.0f 交替 %.2f 占用 %d 格 单格最多 %d 同列率 %.1f%% 24音跨度中位 %.0f' % (
        step_med, alt, len(u), c2.max(), 100 * same, np.median(spans)))
    _byk = _dd(list)
    for v in _g.values():
        if len(v) >= 2:
            _byk[len(v)].append(max(v) - min(v))
    chord_str = ' '.join('%d押%.0f(n%d)' % (k, np.median(_byk[k]), len(_byk[k])) for k in (2, 3, 4) if _byk[k])
    _all = np.array([r['out'] for r in notes])
    lr = '左%.2f 右%.2f 边%.2f' % ((_all < -100).mean(), (_all > 100).mean(), (np.abs(_all) > 400).mean())
    print('多押', chord_str, '|', lr)
    _drags = np.array([abs(r['out']) for r in notes if r['type'] == 3])
    if len(_drags) >= 3:
        _runs = {}
        for b, (rid_, idx_, ln_) in drag_run.items():
            _runs[rid_] = ln_
        sweep_runs = sum(1 for v in _runs.values() if v >= 4)
        print('drag n=%d |x|中位%.0f 中间带135内 %.1f%% 外侧400上 %.1f%% 扫线段%d 最长%d' % (
            len(_drags), np.median(_drags),
            100 * float((_drags < 135).mean()), 100 * float((_drags > 400).mean()),
            sweep_runs, max(_runs.values()) if _runs else 0))


if __name__ == '__main__':
    main()
