"""独立验证：官方 lambda 原样实现（不复用 easing_lib） vs easing_lib.ease
对 32425 全部事件的所有采样点逐一对比"""
import json, math, zipfile, sys
sys.path.insert(0, "/home/user/Doubao/chats/38445055719679234/phira_charts_dl")
from easing_lib import _beat

# ===== 官方 lambda 原样独立实现（Phira docs extend.html 逐字）=====
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

from easing_lib import ease as ease_lib

zf = zipfile.ZipFile("subsets/unstable_0005/32425_Squidward Finishes Those Errands.pez")
jn = [n for n in zf.namelist() if n.lower().endswith(".json") and "AutoSave" not in n]
chart = json.loads(zf.read(jn[0]).decode("utf-8", errors="replace"))

maxdiff = 0.0
bad = 0
total = 0
types_seen = set()
for jl in chart.get("judgeLineList", []):
    for layer in (jl.get("eventLayers", []) or []):
        if not isinstance(layer, dict):
            continue
        for key in ("rotateEvents", "moveXEvents", "moveYEvents"):
            for e in (layer.get(key) or []):
                et = e.get("easingType", 1) or 1
                s, en = _beat(e["startTime"]), _beat(e["endTime"])
                if en <= s:
                    continue
                types_seen.add(et)
                for k in range(11):  # u = 0.0, 0.1, ..., 1.0
                    u = k / 10.0
                    a = ease_official(u, et)
                    b = ease_lib(u, et)
                    total += 1
                    d = abs(a - b)
                    if d > 1e-9:
                        bad += 1
                        maxdiff = max(maxdiff, d)
                        if bad <= 5:
                            print(f"  差异 easingType={et} u={u:.1f}: 官方={a:.10f} 库={b:.10f}")
print(f"采样点总数: {total}, 差异点数: {bad}, 最大差异: {maxdiff:.2e}")
print(f"覆盖 easingType: {sorted(types_seen)}")
print("✅ 全部一致" if bad == 0 else "❌ 存在差异")
