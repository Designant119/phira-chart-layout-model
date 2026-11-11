"""独立验证：官方 GetPos 语义（每层插值累加+父线递归） vs easing_lib.line_state_at"""
import zipfile, json, math, glob, sys
sys.path.insert(0, "/home/user/Doubao/chats/38445055719679234/phira_charts_dl")
from easing_lib import _beat, interp_events, line_state_at, ease as ease_lib

# 独立官方 GetPos：easing 用官方独立实现（复用之前 verify 的 ease_official 逻辑——重新内联）
def ease_official(t, et):
    if et == 1: return t
    if et == 2: return math.sin((t * math.pi) / 2)
    if et == 3: return 1 - math.cos((t * math.pi) / 2)
    if et == 4: return 1 - (1 - t) * (1 - t)
    if et == 5: return t ** 2
    if et == 6: return -(math.cos(math.pi * t) - 1) / 2
    if et == 7: return 2 * (t ** 2) if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
    if et == 8: return 1 - (1 - t) ** 3
    if et == 9: return t ** 3
    if et == 10: return 1 - (1 - t) ** 4
    if et == 11: return t ** 4
    if et == 12: return 4 * (t ** 3) if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    if et == 13: return 8 * (t ** 4) if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2
    if et == 14: return 1 - (1 - t) ** 5
    if et == 15: return t ** 5
    if et == 16: return 1 if t == 1 else 1 - 2 ** (-10 * t)
    if et == 17: return 0 if t == 0 else 2 ** (10 * t - 10)
    if et == 18: return (1 - (t - 1) ** 2) ** 0.5
    if et == 19: return 1 - (1 - t ** 2) ** 0.5
    if et == 20: return 1 + 2.70158 * ((t - 1) ** 3) + 1.70158 * ((t - 1) ** 2)
    if et == 21: return 2.70158 * (t ** 3) - 1.70158 * (t ** 2)
    if et == 22:
        if t < 0.5: return (1 - (1 - (2 * t) ** 2) ** 0.5) / 2
        return (((1 - (-2 * t + 2) ** 2) ** 0.5) + 1) / 2
    if et == 23:
        if t < 0.5: return ((2 * t) ** 2 * ((2.5949095 + 1) * 2 * t - 2.5949095)) / 2
        return ((2 * t - 2) ** 2 * ((2.5949095 + 1) * (t * 2 - 2) + 2.5949095) + 2) / 2
    if et == 24:
        if t == 0: return 0
        if t == 1: return 1
        return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi / 3)) + 1
    if et == 25:
        if t == 0: return 0
        if t == 1: return 1
        return -2 ** (10 * t - 10) * math.sin((t * 10 - 10.75) * (2 * math.pi / 3))
    if et == 26:
        n1, d1 = 7.5625, 2.75
        if t < 1 / d1: return n1 * t * t
        if t < 2 / d1:
            t2 = t - 1.5 / d1; return n1 * t2 * t2 + 0.75
        if t < 2.5 / d1:
            t2 = t - 2.25 / d1; return n1 * t2 * t2 + 0.9375
        t2 = t - 2.625 / d1; return n1 * t2 * t2 + 0.984375
    if et == 27: return 1 - ease_official(1 - t, 26)
    if et == 28:
        if t < 0.5: return (1 - ease_official(1 - 2 * t, 26)) / 2
        return (1 + ease_official(2 * t - 1, 26)) / 2
    if et == 29:
        if t == 0: return 0
        if t == 1: return 1
        if t < 0.5: return -2 ** (20 * t - 10) * math.sin((20 * t - 11.125) * ((2 * math.pi) / 4.5)) / 2
        return 2 ** (-20 * t + 10) * math.sin((20 * t - 11.125) * ((2 * math.pi) / 4.5)) / 2 + 1
    return t

def ev_at(events, t):
    """官方 GetEventValue 语义（含 gap 保持）：排序后 start<=t 最晚事件，区间内 easing 插值"""
    evs = []
    for e in events:
        if not isinstance(e, dict): continue
        s = e.get("startTime", [0,0,1]); en = e.get("endTime", s)
        if not isinstance(s, (list,tuple)) or len(s)<3: continue
        if not isinstance(en, (list,tuple)) or len(en)<3: en = s
        sb, eb = _beat(s), _beat(en)
        try: sv = float(e.get("start", 0.0)); ev = float(e.get("end", sv))
        except Exception: sv = ev = 0.0
        evs.append((sb, eb, sv, ev, e.get("easingType", 1) or 1))
    if not evs: return 0.0
    evs.sort(key=lambda x: x[0])
    if t <= evs[0][0]: return evs[0][2]
    if t >= evs[-1][1]: return evs[-1][3]
    best = None
    for ev in evs:
        if ev[0] <= t: best = ev
        else: break
    sb, eb, sv, ev, et = best
    if t >= eb or eb == sb: return ev
    u = (t - sb) / (eb - sb)
    return sv + (ev - sv) * ease_official(u, et)

def official_getpos(jl, chart, t, cache):
    key = id(jl)
    if key in cache: return cache[key]
    px = py = 0.0
    els = jl.get("eventLayers", []) or []
    if isinstance(els, dict): els = [els]
    for layer in els:
        if isinstance(layer, dict):
            px += ev_at(layer.get("moveXEvents", []) or [], t)
            py += ev_at(layer.get("moveYEvents", []) or [], t)
    f = jl.get("father", -1)
    if f != -1:
        jll = chart.get("judgeLineList", []) or []
        if 0 <= f < len(jll):
            fpx, fpy = official_getpos(jll[f], chart, t, cache)
            px += fpx; py += fpy
    cache[key] = (px, py)
    return px, py

# 找 1000 那首 pez
import os
paths = glob.glob("subsets/unstable_000*/1000_*.pez")
path = paths[0] if paths else None
if not path:
    print("1000 pez 不存在")
    sys.exit(1)
zf = zipfile.ZipFile(path)
jn = [n for n in zf.namelist() if n.lower().endswith(".json") and "AutoSave" not in n]
chart = json.loads(zf.read(jn[0]).decode("utf-8", errors="replace"))
jll = chart.get("judgeLineList", [])

# 随机采样 500 个 (线, 时刻) 对比
import random
random.seed(42)
maxd = 0.0
bad = 0
multi_checked = father_checked = 0
for _ in range(500):
    jl = random.choice(jll)
    els = jl.get("eventLayers", []) or []
    if isinstance(els, dict): els = [els]
    # 采样时刻：用该线第一个音符附近
    tmax = 300.0
    t = random.uniform(0, tmax)
    opx, opy = official_getpos(jl, chart, t, {})
    # 单独 rotate 也对比
    lpx, lpy, lth = line_state_at(jl, chart, t)
    d = max(abs(opx - lpx), abs(opy - lpy))
    if d > 1e-6:
        bad += 1
        maxd = max(maxd, d)
    if len(els) >= 2: multi_checked += 1
    if jl.get("father", -1) != -1: father_checked += 1
print(f"对比 500 个随机样本: 差异 {bad}, 最大差异 {maxd:.2e}")
print(f"其中多layer样本 {multi_checked}, 父线样本 {father_checked}")
print("✅ 官方GetPos语义一致" if bad == 0 else "❌ 有差异")
