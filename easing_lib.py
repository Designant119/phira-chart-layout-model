#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RPE 官方缓动函数库（Phira 官方文档 chart-standard/chart-format/rpe/extend.html 权威对照表）
easingType 1-based：1=Linear, 2=Out Sine, 3=In Sine, 4=Out Quad, 5=In Quad,
6=In Out Sine, 7=In Out Quad, 8=Out Cubic, 9=In Cubic, 10=Out Quart, 11=In Quart,
12=In Out Cubic, 13=In Out Quart, 14=Out Quint, 15=In Quint, 16=Out Expo, 17=In Expo,
18=Out Circ, 19=In Circ, 20=Out Back, 21=In Back, 22=In Out Circ, 23=In Out Back,
24=Out Elastic, 25=In Elastic, 26=Out Bounce, 27=In Bounce, 28=In Out Bounce, 29=In Out Elastic
"""
import math

def ease(t, easing_type):
    """t in [0,1], easing_type int(1-based) -> eased t'（Back/Elastic/Bounce 可越界）"""
    try:
        et = int(easing_type or 1)
    except Exception:
        et = 1
    if et <= 1:
        return t
    if et == 2:  # out sine
        return math.sin((t * math.pi) / 2)
    if et == 3:  # in sine
        return 1 - math.cos((t * math.pi) / 2)
    if et == 4:  # out quad
        return 1 - (1 - t) * (1 - t)
    if et == 5:  # in quad
        return t ** 2
    if et == 6:  # io sine
        return -(math.cos(math.pi * t) - 1) / 2
    if et == 7:  # io quad
        return 2 * (t ** 2) if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
    if et == 8:  # out cubic
        return 1 - (1 - t) ** 3
    if et == 9:  # in cubic
        return t ** 3
    if et == 10:  # out quart
        return 1 - (1 - t) ** 4
    if et == 11:  # in quart
        return t ** 4
    if et == 12:  # io cubic
        return 4 * (t ** 3) if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    if et == 13:  # io quart
        return 8 * (t ** 4) if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2
    if et == 14:  # out quint
        return 1 - (1 - t) ** 5
    if et == 15:  # in quint
        return t ** 5
    if et == 16:  # out expo
        return 1.0 if t == 1 else 1 - 2 ** (-10 * t)
    if et == 17:  # in expo
        return 0.0 if t == 0 else 2 ** (10 * t - 10)
    if et == 18:  # out circ
        return (1 - (t - 1) ** 2) ** 0.5
    if et == 19:  # in circ
        return 1 - (1 - t ** 2) ** 0.5
    if et == 20:  # out back (c1=1.70158, c3=c1+1=2.70158)
        return 1 + 2.70158 * ((t - 1) ** 3) + 1.70158 * ((t - 1) ** 2)
    if et == 21:  # in back
        return 2.70158 * (t ** 3) - 1.70158 * (t ** 2)
    if et == 22:  # io circ
        if t < 0.5:
            return (1 - (1 - (2 * t) ** 2) ** 0.5) / 2
        return (((1 - (-2 * t + 2) ** 2) ** 0.5) + 1) / 2
    if et == 23:  # io back (c2=2.5949095)
        if t < 0.5:
            return ((2 * t) ** 2 * ((2.5949095 + 1) * 2 * t - 2.5949095)) / 2
        return ((2 * t - 2) ** 2 * ((2.5949095 + 1) * (t * 2 - 2) + 2.5949095) + 2) / 2
    if et == 24:  # out elastic
        if t == 0: return 0.0
        if t == 1: return 1.0
        return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi / 3)) + 1
    if et == 25:  # in elastic
        if t == 0: return 0.0
        if t == 1: return 1.0
        return -2 ** (10 * t - 10) * math.sin((t * 10 - 10.75) * (2 * math.pi / 3))
    if et == 26:  # out bounce
        n1, d1 = 7.5625, 2.75
        if t < 1 / d1:
            return n1 * (t ** 2)
        if t < 2 / d1:
            t2 = t - 1.5 / d1
            return n1 * t2 * t2 + 0.75
        if t < 2.5 / d1:
            t2 = t - 2.25 / d1
            return n1 * t2 * t2 + 0.9375
        t2 = t - 2.625 / d1
        return n1 * t2 * t2 + 0.984375
    if et == 27:  # in bounce
        return 1 - ease(1 - t, 26)
    if et == 28:  # io bounce
        if t < 0.5:
            return (1 - ease(1 - 2 * t, 26)) / 2
        return (1 + ease(2 * t - 1, 26)) / 2
    if et == 29:  # io elastic
        if t == 0: return 0.0
        if t == 1: return 1.0
        if t < 0.5:
            return -2 ** (20 * t - 10) * math.sin((20 * t - 11.125) * ((2 * math.pi) / 4.5)) / 2
        return 2 ** (-20 * t + 10) * math.sin((20 * t - 11.125) * ((2 * math.pi) / 4.5)) / 2 + 1
    return t  # 未知编号：线性


def _beat(b):
    return b[0] + b[1] / b[2]


def _layer_events(layer, key):
    if not isinstance(layer, dict):
        return []
    return layer.get(key, []) or []


def interp_events(events, t_beat):
    """单层事件插值（拍）。语义：排序后，start<=t 的最晚事件；t 在事件区间内按 easing 插值，
    t 在区间外（gap）保持前事件 end 值。返回数值。"""
    if not events:
        return 0.0
    evs = []
    for e in events:
        if not isinstance(e, dict):
            continue
        s = e.get("startTime", [0, 0, 1])
        en = e.get("endTime", s)
        if not isinstance(s, (list, tuple)) or len(s) < 3:
            continue
        if not isinstance(en, (list, tuple)) or len(en) < 3:
            en = s
        sb, eb = _beat(s), _beat(en)
        try:
            sv = float(e.get("start", e.get("value", 0.0)))
            ev = float(e.get("end", e.get("endValue", sv)))
        except Exception:
            sv, ev = 0.0, 0.0
        evs.append((sb, eb, sv, ev, e.get("easingType", 1) or 1))
    if not evs:
        return 0.0
    evs.sort(key=lambda x: x[0])
    if t_beat <= evs[0][0]:
        return evs[0][2]
    if t_beat >= evs[-1][1]:
        return evs[-1][3]
    best = None
    for ev in evs:
        if ev[0] <= t_beat:
            best = ev
        else:
            break
    if best is None:
        return evs[0][2]
    sb, eb, sv, ev, et = best
    if t_beat >= eb or eb == sb:
        return ev
    u = (t_beat - sb) / (eb - sb)
    return sv + (ev - sv) * ease(u, et)


def line_state_at(jl, chart, t_beat, _cache=None):
    """判定线在 t_beat（拍）时刻的状态：返回 (px, py, theta)
    官方语义（Phira docs RPE 文档 Python 示例）：
    - 多 eventLayers 的值**累加**（每层独立插值后相加）
    - father != -1 时位置叠加父线位置（GetPos 无条件叠加）
    - rotate 只累加本线各层（官方示例未加父线旋转）
    """
    if _cache is None:
        _cache = {}
    key = (id(jl), t_beat)
    if key in _cache:
        return _cache[key]
    els = jl.get("eventLayers", []) or []
    if isinstance(els, dict):
        els = [els]
    px = py = theta = 0.0
    for layer in els:
        px += interp_events(_layer_events(layer, "moveXEvents"), t_beat)
        py += interp_events(_layer_events(layer, "moveYEvents"), t_beat)
        theta += interp_events(_layer_events(layer, "rotateEvents"), t_beat)
    # 父线位置叠加（旋转官方示例不叠加）
    father = jl.get("father", -1)
    if father != -1:
        jll = chart.get("judgeLineList", []) or []
        if 0 <= father < len(jll):
            fpx, fpy, _ = line_state_at(jll[father], chart, t_beat, _cache)
            px += fpx
            py += fpy
    _cache[key] = (px, py, theta)
    return px, py, theta
